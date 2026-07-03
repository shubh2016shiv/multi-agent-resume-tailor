"""Stage 3 professional experience rewrite flow.

The LLM rewrites one role's bullets 1:1 into specific, recruiter-ready
accomplishments, mining the role's own description and skills for truthful
detail. Code holds two lines with deliberately different force:

  * Truthfulness is non-negotiable. A rewrite must keep the source bullet count
    and introduce no figure the source role lacks (a deterministic check). Only a
    truthful version may ship; if none survives one repair, the source bullets do.
  * Substance is best-effort. A semantic rewrite review flags unsupported
    specificity, ownership inflation, vague accomplishments, or brochure tone.
    One repair tries to fix those issues, and any remaining soft gaps are
    surfaced in notes rather than silently dressed up.

Role metadata is always rebuilt from the source object, so the LLM's writable
surface is the achievements text only.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

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

    Reads: resume, job_description, and alignment_strategy.
    Writes: optimized_experience.
    Returns: partial state with merged OptimizedExperienceSection output.
    """
    start_time = time.monotonic()
    resume = state["resume"]
    job_description = state["job_description"]
    strategy = state["alignment_strategy"]
    logger.info(
        "pipeline_stage_started",
        stage="optimize_experience",
        run_id=state["run_id"],
    )
    if resume is None or job_description is None or strategy is None:
        raise ValueError("resume, job_description, and alignment_strategy must be set.")

    optimized_experience = _optimize_experience_entries(resume, job_description, strategy, state["run_id"])
    duration_ms = round((time.monotonic() - start_time) * 1000)
    logger.info(
        "pipeline_stage_completed",
        stage="optimize_experience",
        run_id=state["run_id"],
        duration_ms=duration_ms,
    )
    return {"optimized_experience": optimized_experience}


def _optimize_experience_entries(
    resume: Resume,
    job_description: JobDescription,
    strategy: AlignmentStrategy,
    run_id: str = "unknown",
) -> OptimizedExperienceSection:
    """Optimize every resume experience and merge the role-scoped sections.

    Expects resume.work_experience to contain at least one entry.
    Returns one OptimizedExperienceSection for downstream ATS assembly.
    """
    experiences = resume.work_experience
    if not experiences:
        raise ValueError("resume.work_experience must contain at least one entry.")

    sections = _run_experience_optimization_workers(
        resume=resume,
        job_description=job_description,
        strategy=strategy,
        experiences=experiences,
        run_id=run_id,
    )
    return _merge_optimized_experience_sections(sections)


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
        "to that bullet. Keep the same bullet count and add no figure the source role "
        "does not state. Return only ExperienceRewriteProposal JSON."
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
        first_rewritten_role,
        original_experience,
    )
    first_quality_review = (
        ReviewResult(comments=[], summary="", score=None)
        if truthfulness_findings
        else _collect_rewrite_quality_review(original_experience, rewrite_proposal)
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
        repaired_role,
        original_experience,
    )
    repaired_quality_review = (
        ReviewResult(comments=[], summary="", score=None)
        if repaired_truthfulness_findings
        else _collect_rewrite_quality_review(original_experience, repaired_rewrite_proposal)
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


def _finalize_role_rewrite(
    rewrite_proposal: ExperienceRewriteProposal,
    context: str,
    source_role_resume: Resume,
    original_experience: Experience,
    run_id: str = "unknown",
) -> OptimizedExperienceSection:
    """Return the finalized section for one role after the rewrite decision is made."""
    rewrite_decision = _decide_role_rewrite_outcome(
        rewrite_proposal,
        context,
        source_role_resume,
        original_experience,
        run_id=run_id,
    )
    return rewrite_decision.finalized_section


def _run_single_experience_optimization(
    resume: Resume,
    job_description: JobDescription,
    strategy: AlignmentStrategy,
    experience: Experience,
    run_id: str = "unknown",
) -> OptimizedExperienceSection:
    """Rewrite one role's bullets with a role-scoped CrewAI call plus a code-owned gate.

    Expects job_description and strategy to be present in pipeline state.
    Returns a truth-verified rewritten role, or the untouched source role when the
    rewrite fails the truthfulness gate twice.
    """
    role_resume = _build_resume_with_single_experience(resume, experience)
    context = format_experience_optimizer_context(
        resume=role_resume,
        job_description=job_description,
        strategy=strategy,
        format_type="toon",
    )
    rewrite_proposal = _request_role_rewrite_proposal(context, run_id=run_id)
    return _finalize_role_rewrite(
        rewrite_proposal,
        context,
        role_resume,
        experience,
        run_id=run_id,
    )


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
    run_id: str = "unknown",
) -> list[OptimizedExperienceSection]:
    """Run role-scoped experience optimization calls in parallel.

    Expects a non-empty experiences list from resume.work_experience.
    Returns one OptimizedExperienceSection per input experience.
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
                    run_id,
                ),
                experiences,
            )
        )
