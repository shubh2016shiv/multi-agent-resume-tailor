"""Stage 3 professional experience rewrite flow.

The LLM rewrites one role's bullets 1:1 into specific, recruiter-readable
accomplishments, mining only that role's own evidence. Code enforces two layers:

  * Truthfulness is non-negotiable. A rewrite must keep the source bullet count,
    preserve bullet identity/order, and introduce no unsupported numeric claim.
    Only a truth-safe version may ship; if none survives one repair, the source
    bullets do.
  * Substance is best-effort. A semantic rewrite review flags unsupported
    specificity, ownership inflation, vague accomplishments, brochure tone, or
    JD-keyword decoration. One repair tries to fix those issues before shipping.

Human-in-the-loop clarification is a real pause/resume step, not an offline
note. After rewrite plus one repair pass, a single semantic review decides which
bullets are still truthful but missing candidate-owned facts (result, artifact,
user, or scale) and phrases the candidate's question in the same pass. That whole
mechanism lives in src/hitl/professional_experience/; this module only
orchestrates rewrite, review, and the pause boundary before ATS assembly.

Role metadata is always rebuilt from the source object, so the LLM's writable
surface is the achievements text only.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from langgraph.types import interrupt

from src.agents.professional_experience import create_professional_experience_agent
from src.agents.professional_experience.models import (
    ExperienceRewriteProposal,
    OptimizedExperienceSection,
)
from src.core.logger import get_logger
from src.data_models.job import JobDescription
from src.data_models.resume import Experience, Resume
from src.data_models.strategy import AlignmentStrategy
from src.formatters.experience_optimizer_formatter import format_experience_optimizer_context
from src.hitl.professional_experience.answers import (
    answers_for_role,
    experience_with_candidate_answers,
)
from src.hitl.professional_experience.clarifications import build_bullet_clarifications
from src.hitl.professional_experience.models import (
    ExperienceBulletClarification,
    build_experience_bullet_id,
)
from src.orchestration.crew_task_execution import run_agent_task
from src.orchestration.state import ResumeEnhancementPipelineState
from src.tools.contracts import ReviewComment, ReviewResult, Severity
from src.tools.engines.resume_diagnostics import audit_experience_rewrite_quality
from src.tools.engines.truthfulness import detect_claim_inflation

logger = get_logger(__name__)


@dataclass(frozen=True)
class RoleRewriteDecision:
    """The rewrite proposal and review state that ultimately determined one role's output."""

    selected_rewrite_proposal: ExperienceRewriteProposal
    selected_quality_review: ReviewResult
    finalized_section: OptimizedExperienceSection


def optimize_experience(state: ResumeEnhancementPipelineState) -> dict:
    """Rewrite work-experience bullets with one role-scoped call per entry.

    Reads: resume, job_description, alignment_strategy, and (when the candidate
    answered a previous run's sheet) clarification_answers.
    Writes: optimized_experience and experience_clarifications.
    Returns: partial state with the merged section and the candidate questions.
    """
    start_time = time.monotonic()
    resume = state["resume"]
    job_description = state["job_description"]
    strategy = state["alignment_strategy"]
    clarification_answers = state.get("clarification_answers") or []
    logger.info(
        "pipeline_stage_started",
        stage="optimize_experience",
        run_id=state["run_id"],
        answered_clarifications=len(clarification_answers),
    )
    if resume is None or job_description is None or strategy is None:
        raise ValueError("resume, job_description, and alignment_strategy must be set.")

    optimized_experience, clarifications = _optimize_experience_entries(
        resume, job_description, strategy, clarification_answers, state["run_id"]
    )
    source_resume_for_downstream_review = _resume_with_candidate_answers_as_source(
        resume,
        clarification_answers,
    )
    duration_ms = round((time.monotonic() - start_time) * 1000)
    logger.info(
        "pipeline_stage_completed",
        stage="optimize_experience",
        run_id=state["run_id"],
        clarifications_requested=len(clarifications),
        duration_ms=duration_ms,
    )
    return {
        "resume": source_resume_for_downstream_review,
        "optimized_experience": optimized_experience,
        "experience_clarifications": clarifications,
        "clarification_answers": [],
    }


def _resume_with_candidate_answers_as_source(
    resume: Resume,
    clarification_answers: list[ExperienceBulletClarification],
) -> Resume:
    """Return the resume truth source after applying candidate-owned facts."""
    if not clarification_answers:
        return resume
    updated_experiences = [
        experience_with_candidate_answers(
            experience,
            answers_for_role(experience, clarification_answers),
        )
        for experience in resume.work_experience
    ]
    return resume.model_copy(update={"work_experience": updated_experiences})


def await_candidate_clarifications(state: ResumeEnhancementPipelineState):
    """Pause the graph when the experience stage needs candidate-owned bullet facts."""
    clarifications = state.get("experience_clarifications") or []
    clarification_answers = state.get("clarification_answers") or []
    if not clarifications:
        return {}
    if clarification_answers:
        logger.info(
            "candidate_clarifications_received",
            run_id=state["run_id"],
            answered_clarifications=len(clarification_answers),
        )
        return {}
    logger.info(
        "candidate_clarifications_requested",
        run_id=state["run_id"],
        clarification_count=len(clarifications),
    )
    interrupt(
        {
            "type": "candidate_clarifications_required",
            "questions": [clarification.model_dump(mode="json") for clarification in clarifications],
        }
    )
    return {}


def _optimize_experience_entries(
    resume: Resume,
    job_description: JobDescription,
    strategy: AlignmentStrategy,
    clarification_answers: list[ExperienceBulletClarification],
    run_id: str = "unknown",
) -> tuple[OptimizedExperienceSection, list[ExperienceBulletClarification]]:
    """Optimize every resume experience and merge the role-scoped results.

    Expects resume.work_experience to contain at least one entry.
    Returns the merged OptimizedExperienceSection for downstream ATS assembly,
    plus every question the roles raised for the candidate.
    """
    experiences = resume.work_experience
    if not experiences:
        raise ValueError("resume.work_experience must contain at least one entry.")

    role_outcomes = _run_experience_optimization_workers(
        resume=resume,
        job_description=job_description,
        strategy=strategy,
        experiences=experiences,
        clarification_answers=clarification_answers,
        run_id=run_id,
    )
    sections = [section for section, _ in role_outcomes]
    clarifications = [
        clarification for _, role_clarifications in role_outcomes for clarification in role_clarifications
    ]
    return _merge_optimized_experience_sections(sections), clarifications


def _build_resume_with_single_experience(resume: Resume, experience: Experience) -> Resume:
    """Copy resume context with only one work experience entry.

    Expects a validated Resume and one Experience from resume.work_experience.
    Returns a Resume copy whose work_experience list contains only that entry.
    """
    return resume.model_copy(update={"work_experience": [experience]})


def _rebuild_rewritten_role_from_proposal(
    rewrite_proposal: ExperienceRewriteProposal,
    original_experience: Experience,
) -> Experience:
    """Keep only the rewritten bullet text; every other field comes from the source.

    This is the containment boundary: metadata, description, and skills_used can
    never be changed by the LLM, whatever the rewrite proposal contains.
    """
    rewritten_achievements = [
        bullet_rewrite.rewritten_bullet
        for bullet_rewrite in rewrite_proposal.rewritten_bullets
    ]
    return original_experience.model_copy(update={"achievements": rewritten_achievements})


def _collect_rewrite_truthfulness_findings(
    source_role_resume: Resume,
    rewrite_proposal: ExperienceRewriteProposal,
    rewritten_role: Experience,
    original_experience: Experience,
) -> list[str]:
    """Return truthfulness findings for one rewritten role (empty list = safe to ship).

    Two deterministic checks, no LLM -- this is the non-negotiable floor a rewrite
    must clear to ship:
    1. Bullet count parity -- the 1:1 rewrite contract, a closed countable fact.
    2. detect_claim_inflation -- any figure present in the rewrite but absent from
       the source role. Numbers are the highest-risk, most verifiable fabrication.

    Non-numeric invention (a fake tool, team, or scale) is governed by the writer's
    prompt and temperature-0 decoding, not an LLM guard here: an LLM drift check on
    this path penalised the very context-mining that makes bullets specific, so it
    was removed for a net gain in quality without loosening the number floor.
    """
    source_count = len(original_experience.achievements)
    rewritten_count = len(rewritten_role.achievements)
    if rewritten_count != source_count:
        return [
            f"Bullet count changed: source role has {source_count} bullet(s), "
            f"rewrite has {rewritten_count}. Rewrite each source bullet one-for-one."
        ]

    expected_bullet_ids = [
        build_experience_bullet_id(original_experience, bullet_index)
        for bullet_index, _ in enumerate(original_experience.achievements)
    ]
    returned_bullet_ids = [
        bullet_rewrite.bullet_id
        for bullet_rewrite in rewrite_proposal.rewritten_bullets
    ]
    if returned_bullet_ids != expected_bullet_ids:
        return [
            "Bullet IDs changed or were returned out of source order. "
            "Copy each bullet_id exactly from the matching source bullet record."
        ]

    rewritten_role_resume = source_role_resume.model_copy(
        update={"work_experience": [rewritten_role]}
    )
    inflation = detect_claim_inflation(source_role_resume, rewritten_role_resume)
    return [f"{comment.message}. {comment.advice}" for comment in inflation.comments]


def _collect_rewrite_quality_review(
    source_experience: Experience,
    rewrite_proposal: ExperienceRewriteProposal,
) -> ReviewResult:
    """Review rewritten bullets for supported specificity, ownership, and recruiter tone."""
    return audit_experience_rewrite_quality(
        source_experience,
        rewrite_proposal.rewritten_bullets,
    )


def _split_repair_required_comments_from_follow_up_comments(
    quality_review: ReviewResult,
) -> tuple[list[ReviewComment], list[ReviewComment]]:
    """Separate must-fix rewrite findings from follow-up guidance.

    MAJOR/BLOCKER findings indicate unsupported specificity or ownership inflation
    and should not ship unresolved. MINOR/SUGGESTION findings may still ship if a
    truthful rewrite remains imperfect after one repair; those are surfaced as
    follow-up guidance for the candidate.
    """
    repair_required_comments = []
    follow_up_comments = []
    for review_comment in quality_review.comments:
        if review_comment.severity in {Severity.BLOCKER, Severity.MAJOR}:
            repair_required_comments.append(review_comment)
        else:
            follow_up_comments.append(review_comment)
    return repair_required_comments, follow_up_comments


def _build_candidate_follow_up_note(comments: list[ReviewComment]) -> str:
    """Surface bullets that stayed modest so the candidate can add real detail.

    These are not failures -- they are honest gaps the pipeline will not paper over
    with invented facts. Empty when every shipped bullet reads as a concrete
    accomplishment.
    """
    if not comments:
        return ""
    lines = [f'  - "{comment.quoted_text}": {comment.message}' for comment in comments]
    return (
        "\nBULLETS NEEDING YOUR INPUT (kept truthful but could not be made concrete "
        "without inventing facts -- add a specific result or detail):\n" + "\n".join(lines)
    )


def _accepted_section(revised_role: Experience, note: str) -> OptimizedExperienceSection:
    """Wrap an accepted (truth-verified) role for the downstream merge."""
    return OptimizedExperienceSection(
        optimized_experiences=[revised_role],
        optimization_notes=note,
    )


def _source_preserved_section(experience: Experience, reasons: list[str]) -> OptimizedExperienceSection:
    """Return the untouched source role after no truthful rewrite survived one repair."""
    return OptimizedExperienceSection(
        optimized_experiences=[experience],
        optimization_notes=(
            "Rewrite could not stay truthful after one repair; original bullets "
            "preserved. Findings: " + "; ".join(reasons)
        ),
    )


def _build_experience_repair_context(
    original_context: str,
    rewrite_proposal: ExperienceRewriteProposal,
    findings: list[str],
) -> str:
    """Add the previous proposal and the exact findings for the single repair call."""
    feedback = "\n".join(f"- {finding}" for finding in findings)
    return (
        f"{original_context}\n\n"
        f"PREVIOUS_EXPERIENCE_REWRITE_PROPOSAL_JSON:\n{rewrite_proposal.model_dump_json()}\n\n"
        f"EXPERIENCE_AUDIT_FEEDBACK:\n{feedback}\n\n"
        "Rewrite once more to fix every finding above. Preserve the improvements in "
        "bullets the findings did NOT flag. Fix only the flagged rewrite problems: "
        "unsupported specificity, ownership inflation, brochure tone, vague "
        "accomplishments, or JD keyword decoration. To fix unsupported specificity, "
        "remove or replace the shaky detail with supportable role evidence. To fix "
        "ownership inflation, restate the contribution at its true level in active "
        "phrasing ('Contributed to X', 'Supported Y'). To fix weak bullets, use this "
        "same role's description or skills_used only when the detail genuinely belongs "
        "to that bullet. If a flagged bullet still cannot state a concrete result or "
        "scope without inventing facts, keep it truthful and set its "
        "clarifying_question to exactly the missing piece. Keep the same bullet count "
        "and add no figure the source role does not state. Return only "
        "ExperienceRewriteProposal JSON."
    )


def _request_role_rewrite_proposal(
    context: str,
    run_id: str = "unknown",
) -> ExperienceRewriteProposal:
    """Ask the professional experience agent to rewrite one role's bullets.

    Expects TOON context for a single role.
    Returns an ExperienceRewriteProposal validated by CrewAI.
    """
    return run_agent_task(
        agent=create_professional_experience_agent(),
        task_name="optimize_experience_section_task",
        context=context,
        output_model=ExperienceRewriteProposal,
        run_id=run_id,
    )


def _render_quality_findings_for_repair(
    rewrite_comments: list[ReviewComment],
) -> list[str]:
    """Convert structured rewrite comments into concise repair instructions."""
    return [
        f"{review_comment.message}. {review_comment.advice}"
        for review_comment in rewrite_comments
    ]


def _decide_role_rewrite_outcome(
    rewrite_proposal: ExperienceRewriteProposal,
    context: str,
    source_role_resume: Resume,
    original_experience: Experience,
    run_id: str = "unknown",
) -> RoleRewriteDecision:
    """Choose the best rewrite to ship, repairing once when the first draft falls short.

    The two lines have different force. Truthfulness is non-negotiable: only a
    version that clears the truth floor may ship, and if none survives one repair
    the source bullets do. Substance is best-effort: thin bullets earn a repair,
    and whatever is still thin afterward is surfaced to the candidate, never
    invented away.
    """
    ####################################################
    # STEP 1: CHECK THE FIRST PROPOSAL -- TRUTH FLOOR, THEN SUBSTANCE#
    ####################################################
    first_rewritten_role = _rebuild_rewritten_role_from_proposal(
        rewrite_proposal,
        original_experience,
    )
    truthfulness_findings = _collect_rewrite_truthfulness_findings(
        source_role_resume,
        rewrite_proposal,
        first_rewritten_role,
        original_experience,
    )
    # The quality review judges against the EVIDENCE role (which may carry the
    # candidate's clarification answers), not the bare original -- otherwise
    # answer-sourced specifics would be flagged as unsupported.
    evidence_experience = source_role_resume.work_experience[0]
    first_quality_review = (
        ReviewResult(comments=[], summary="", score=None)
        if truthfulness_findings
        else _collect_rewrite_quality_review(evidence_experience, rewrite_proposal)
    )
    repair_required_comments, candidate_follow_up_comments = (
        _split_repair_required_comments_from_follow_up_comments(first_quality_review)
    )
    if (
        not truthfulness_findings
        and not repair_required_comments
        and not candidate_follow_up_comments
    ):
        return RoleRewriteDecision(
            selected_rewrite_proposal=rewrite_proposal,
            selected_quality_review=first_quality_review,
            finalized_section=_accepted_section(
                first_rewritten_role,
                "Rewrote bullets with grounded role evidence; recruiter-readable and truthful.",
            ),
        )

    ####################################################
    # STEP 2: SPEND THE SINGLE REPAIR ON EVERY FINDING AT ONCE#
    ####################################################
    findings = truthfulness_findings + _render_quality_findings_for_repair(
        repair_required_comments + candidate_follow_up_comments
    )
    logger.info(
        "experience_rewrite_repair_requested",
        run_id=run_id,
        company=original_experience.company_name,
        truthfulness_findings=len(truthfulness_findings),
        quality_findings=len(repair_required_comments) + len(candidate_follow_up_comments),
    )
    repair_context = _build_experience_repair_context(context, rewrite_proposal, findings)
    repaired_rewrite_proposal = _request_role_rewrite_proposal(repair_context, run_id=run_id)
    repaired_role = _rebuild_rewritten_role_from_proposal(
        repaired_rewrite_proposal,
        original_experience,
    )

    ####################################################
    # STEP 3: SHIP THE BEST TRUTHFUL VERSION, SURFACING THIN BULLETS#
    ####################################################
    repaired_truthfulness_findings = _collect_rewrite_truthfulness_findings(
        source_role_resume,
        repaired_rewrite_proposal,
        repaired_role,
        original_experience,
    )
    repaired_quality_review = (
        ReviewResult(comments=[], summary="", score=None)
        if repaired_truthfulness_findings
        else _collect_rewrite_quality_review(evidence_experience, repaired_rewrite_proposal)
    )
    repaired_repair_required_comments, repaired_candidate_follow_up_comments = (
        _split_repair_required_comments_from_follow_up_comments(repaired_quality_review)
    )
    if not repaired_truthfulness_findings and not repaired_repair_required_comments:
        return RoleRewriteDecision(
            selected_rewrite_proposal=repaired_rewrite_proposal,
            selected_quality_review=repaired_quality_review,
            finalized_section=_accepted_section(
                repaired_role,
                "Repaired once with grounded rewrite feedback; truthful."
                + _build_candidate_follow_up_note(repaired_candidate_follow_up_comments),
            ),
        )
    if not truthfulness_findings and not repair_required_comments:
        # The first rewrite was already safe to ship; the repair made it worse, so
        # keep the first rewrite and surface any follow-up gaps it still has.
        return RoleRewriteDecision(
            selected_rewrite_proposal=rewrite_proposal,
            selected_quality_review=first_quality_review,
            finalized_section=_accepted_section(
                first_rewritten_role,
                "Kept the first grounded rewrite (repair introduced new issues)."
                + _build_candidate_follow_up_note(candidate_follow_up_comments),
            ),
        )
    logger.warning(
        "experience_rewrite_source_fallback",
        run_id=run_id,
        company=original_experience.company_name,
        findings=repaired_truthfulness_findings
        or _render_quality_findings_for_repair(repaired_repair_required_comments),
    )
    return RoleRewriteDecision(
        selected_rewrite_proposal=repaired_rewrite_proposal,
        selected_quality_review=repaired_quality_review,
        finalized_section=_source_preserved_section(
            original_experience,
            repaired_truthfulness_findings
            or _render_quality_findings_for_repair(repaired_repair_required_comments),
        ),
    )


def _run_single_experience_optimization(
    resume: Resume,
    job_description: JobDescription,
    strategy: AlignmentStrategy,
    experience: Experience,
    role_answers: list[ExperienceBulletClarification],
    run_id: str = "unknown",
) -> tuple[OptimizedExperienceSection, list[ExperienceBulletClarification]]:
    """Rewrite one role's bullets with a role-scoped CrewAI call plus a code-owned gate.

    Expects job_description and strategy to be present in pipeline state.
    Returns the truth-verified rewritten role (or the untouched source role when
    the rewrite fails the truthfulness gate twice), plus any questions this role
    raises for the candidate.
    """
    if role_answers:
        logger.info(
            "experience_clarification_answers_applied",
            run_id=run_id,
            company=experience.company_name,
            answers=len(role_answers),
        )
    # Candidate answers travel on exactly one channel per consumer. The writer's
    # prompt carries them once, as structured candidate_clarification_evidence
    # (built by the formatter from role_answers). The deterministic truth floor
    # and the LLM reviews instead see them folded into the role description, so
    # answer-sourced facts count as source evidence and are never flagged as
    # invented.
    writer_role_resume = _build_resume_with_single_experience(resume, experience)
    evidence_experience = experience_with_candidate_answers(experience, role_answers)
    evidence_role_resume = _build_resume_with_single_experience(resume, evidence_experience)
    context = format_experience_optimizer_context(
        resume=writer_role_resume,
        job_description=job_description,
        strategy=strategy,
        clarification_answers=role_answers,
        format_type="toon",
    )
    rewrite_proposal = _request_role_rewrite_proposal(context, run_id=run_id)
    rewrite_decision = _decide_role_rewrite_outcome(
        rewrite_proposal,
        context,
        evidence_role_resume,
        experience,
        run_id=run_id,
    )
    # The fact-gap review receives the evidence experience so facts the candidate
    # already provided count as role evidence and are not asked for again.
    clarifications = build_bullet_clarifications(
        experience=evidence_experience,
        shipped_bullets=rewrite_decision.finalized_section.optimized_experiences[0].achievements,
        rewritten_bullets=rewrite_decision.selected_rewrite_proposal.rewritten_bullets,
        run_id=run_id,
    )
    return rewrite_decision.finalized_section, clarifications


def _merge_optimized_experience_sections(
    sections: list[OptimizedExperienceSection],
) -> OptimizedExperienceSection:
    """Merge role-scoped optimization results into one section.

    Expects each section to contain at least one optimized experience.
    Returns one OptimizedExperienceSection for downstream ATS assembly.
    """
    optimized_experiences = []
    optimization_notes = []
    keywords_integrated = []
    relevance_scores = []

    for section in sections:
        optimized_experiences.extend(section.optimized_experiences)
        if section.optimization_notes:
            optimization_notes.append(section.optimization_notes)
        keywords_integrated.extend(section.keywords_integrated)
        relevance_scores.extend(section.relevance_scores)

    return OptimizedExperienceSection(
        optimized_experiences=optimized_experiences,
        optimization_notes="\n".join(optimization_notes),
        keywords_integrated=list(dict.fromkeys(keywords_integrated)),
        relevance_scores=relevance_scores,
    )


def _run_experience_optimization_workers(
    resume: Resume,
    job_description: JobDescription,
    strategy: AlignmentStrategy,
    experiences: list[Experience],
    clarification_answers: list[ExperienceBulletClarification],
    run_id: str = "unknown",
) -> list[tuple[OptimizedExperienceSection, list[ExperienceBulletClarification]]]:
    """Run role-scoped experience optimization calls in parallel.

    Expects a non-empty experiences list from resume.work_experience.
    Returns one (section, candidate questions) pair per input experience; each
    role receives only the answered clarifications routed to it.
    """
    max_workers = min(len(experiences), 4)
    logger.debug(
        "experience_optimization_parallelism",
        experience_count=len(experiences),
        max_workers=max_workers,
    )
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(
            executor.map(
                lambda experience: _run_single_experience_optimization(
                    resume,
                    job_description,
                    strategy,
                    experience,
                    answers_for_role(experience, clarification_answers),
                    run_id,
                ),
                experiences,
            )
        )
