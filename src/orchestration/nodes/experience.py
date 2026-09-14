"""Stage 3 (parallel with summary and skills): rewrite the work-experience bullets.

Start here if you are new to this file. Functions below appear in the order they run.

WHAT THIS STAGE DOES
    One LLM call per role rewrites that role's bullets 1:1 into specific,
    recruiter-readable accomplishments, using only that role's own evidence. Code
    then decides whether the rewrite may ship, because an LLM asked to make bullets
    impressive will otherwise invent the impressive part.

THE TWO CHECKS, WHICH HAVE DIFFERENT FORCE
    Truthfulness is non-negotiable (_collect_rewrite_truthfulness_findings, code only).
    A rewrite must keep the source bullet count, keep bullet identity and order, and
    introduce no number the source role does not state. Only a rewrite that clears
    this may ship; if neither the first attempt nor the repair clears it, the original
    bullets ship unchanged.

    Substance is best-effort (audit_experience_rewrite_quality, an LLM review). It
    flags unsupported specificity, ownership inflation, vague accomplishments,
    brochure tone, and JD-keyword decoration. MAJOR findings must be fixed before
    shipping; MINOR ones may ship and are surfaced to the candidate instead.

HOW THE CODE IS LAID OUT
    optimize_experience                 the graph node: fan out over roles, merge back
      _optimize_experience_entries      one thread per role (see the note there on
                                        why this parallelism is mostly nominal)
        _run_single_experience_optimization      one role, end to end
          _request_role_rewrite_proposal         the LLM call
          _check_proposal                        score one proposal on both checks
          _decide_role_rewrite_outcome           pick which version ships, repair once
          build_bullet_clarifications            -> src/hitl/professional_experience/

    await_candidate_clarifications      a separate graph node: the pause boundary,
                                        reached after all of Stage 3 has finished

AT MOST ONE REPAIR
    A failing proposal earns exactly one more LLM attempt, never a loop. If the repair
    comes back worse than the first attempt, the first attempt ships. This is the same
    no-retry-loops rule the whole pipeline follows.

WHAT THE LLM CANNOT CHANGE
    Only bullet text. Role metadata (company, title, dates, description, skills_used)
    is always rebuilt from the source object, whatever the proposal contains.

THE HUMAN-IN-THE-LOOP PART
    Bullets that stayed truthful but thin produce a question for the candidate, which
    pauses the run until they answer. The review that decides this, and the
    pause/resume file handling, live in src/hitl/professional_experience/; this module
    only calls into it and owns the pause boundary before ATS assembly.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from langgraph.types import interrupt

from src.agents.professional_experience import create_professional_experience_agent
from src.agents.professional_experience.models import (
    ExperienceRewriteProposal,
    OptimizedExperienceSection,
)
from src.core.logger import get_logger
from src.core.settings import get_config
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
from src.orchestration.nodes._stage import pipeline_stage
from src.orchestration.state import ResumeEnhancementPipelineState, require
from src.tools.contracts import ReviewComment, ReviewResult, Severity
from src.tools.engines.resume_diagnostics import audit_experience_rewrite_quality
from src.tools.engines.truthfulness import detect_claim_inflation

logger = get_logger(__name__)


@dataclass(frozen=True)
class RoleRewriteDecision:
    """What one role ships: the finished section, and the proposal that produced it.

    The proposal is kept alongside the section because the fact-gap review needs the
    per-bullet rewrite records, not just the final bullet text.
    """

    selected_rewrite_proposal: ExperienceRewriteProposal
    finalized_section: OptimizedExperienceSection


@dataclass(frozen=True)
class _ProposalCheck:
    """What one rewrite proposal scored on both layers, and what that allows.

    Findings are kept in three separate lists because they carry different weight:
    truthfulness_findings blocks shipping outright, repair_required_comments
    (MAJOR/BLOCKER) must be fixed first, and follow_up_comments (MINOR/SUGGESTION)
    may ship as long as the candidate is told about them.
    """

    rewritten_role: Experience
    truthfulness_findings: list[str]
    repair_required_comments: list[ReviewComment]
    follow_up_comments: list[ReviewComment]

    @property
    def is_shippable(self) -> bool:
        """True when this proposal may ship, with or without follow-up notes."""
        return not self.truthfulness_findings and not self.repair_required_comments

    @property
    def is_flawless(self) -> bool:
        """True when this proposal may ship and there is nothing to tell the candidate."""
        return self.is_shippable and not self.follow_up_comments

    @property
    def repair_findings(self) -> list[str]:
        """Every finding the one repair attempt should fix, truth first then quality.

        Follow-up comments are included: the repair is the only chance to improve
        them, even though they would not block shipping on their own.
        """
        return self.truthfulness_findings + _render_quality_findings_for_repair(
            self.repair_required_comments + self.follow_up_comments
        )

    @property
    def blocking_findings(self) -> list[str]:
        """Why this proposal cannot ship, for the note on the preserved source role.

        Truth findings win when present; they are the more fundamental failure.
        """
        return self.truthfulness_findings or _render_quality_findings_for_repair(
            self.repair_required_comments
        )


@pipeline_stage("optimize_experience")
def optimize_experience(state: ResumeEnhancementPipelineState) -> dict:
    """Rewrite work-experience bullets with one role-scoped call per entry.

    Reads: resume, job_description, alignment_strategy, and (when the candidate
    answered a previous run's sheet) clarification_answers.
    Writes: optimized_experience and experience_clarifications.
    Returns: partial state with the merged section and the candidate questions.
    """
    resume = require(state["resume"], "resume")
    job_description = require(state["job_description"], "job_description")
    strategy = require(state["alignment_strategy"], "alignment_strategy")
    clarification_answers = state.get("clarification_answers") or []
    optimized_experience, clarifications = _optimize_experience_entries(
        resume, job_description, strategy, clarification_answers, state["run_id"]
    )
    source_resume_for_downstream_review = _resume_with_candidate_answers_as_source(
        resume,
        clarification_answers,
    )
    # Both HITL counts in one event: how many answers this pass consumed, and how
    # many new questions it raised. Kept out of the timing event so every stage's
    # start/complete pair has the same shape.
    logger.info(
        "experience_clarification_flow",
        run_id=state["run_id"],
        answered_clarifications=len(clarification_answers),
        clarifications_requested=len(clarifications),
    )
    return {
        "resume": source_resume_for_downstream_review,
        "optimized_experience": optimized_experience,
        "experience_clarifications": clarifications,
        "clarification_answers": [],
    }


# HITL COMPONENT 2 -- PAUSE. See
# src/hitl/professional_experience/README.md#5-component-2--pause-mechanism
#
# Resume-safety note (README section 11 "The double-execution trap"): LangGraph
# re-runs this whole node from the top on resume, not just from the interrupt()
# line. The `if clarification_answers:` check below is what makes that safe --
# on resume, Command(update=...) has already populated clarification_answers,
# so this branch returns BEFORE interrupt() is reached a second time. Do not
# reorder these checks or add code above interrupt() without re-verifying that.
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
            "questions": [
                clarification.model_dump(mode="json") for clarification in clarifications
            ],
        }
    )
    return {}


def _optimize_experience_entries(
    resume: Resume,
    job_description: JobDescription,
    strategy: AlignmentStrategy,
    clarification_answers: list[ExperienceBulletClarification],
    run_id: str,
) -> tuple[OptimizedExperienceSection, list[ExperienceBulletClarification]]:
    """Rewrite every role in parallel, then merge the role-scoped results.

    Expects resume.work_experience to contain at least one entry. Returns the merged
    OptimizedExperienceSection for downstream ATS assembly, plus every question the
    roles raised for the candidate. Each role sees only the answers routed to it.

    Note that the parallelism here buys less than it appears to: every rewrite call
    funnels through the process-wide kickoff lock (see orchestration_architecture.md
    section 9b), so only the non-Crew review calls truly overlap.
    """
    experiences = resume.work_experience
    if not experiences:
        raise ValueError("resume.work_experience must contain at least one entry.")

    max_workers = min(len(experiences), 4)
    logger.debug(
        "experience_optimization_parallelism",
        experience_count=len(experiences),
        max_workers=max_workers,
    )
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        role_outcomes = list(
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
    sections = [section for section, _ in role_outcomes]
    clarifications = [
        clarification
        for _, role_clarifications in role_outcomes
        for clarification in role_clarifications
    ]
    return (
        _merge_optimized_experience_sections(sections),
        _cap_clarifications(clarifications, run_id),
    )


def _run_single_experience_optimization(
    resume: Resume,
    job_description: JobDescription,
    strategy: AlignmentStrategy,
    experience: Experience,
    role_answers: list[ExperienceBulletClarification],
    run_id: str,
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
    writer_role_resume = resume.model_copy(update={"work_experience": [experience]})
    evidence_experience = experience_with_candidate_answers(experience, role_answers)
    evidence_role_resume = resume.model_copy(update={"work_experience": [evidence_experience]})
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


def _request_role_rewrite_proposal(
    context: str,
    run_id: str,
) -> ExperienceRewriteProposal:
    """Ask the professional experience agent to rewrite one role's bullets.

    Expects TOON context for a single role.
    Returns an ExperienceRewriteProposal validated by run_agent_task against
    the agent's raw output.
    """
    return run_agent_task(
        agent=create_professional_experience_agent(),
        task_name="optimize_experience_section_task",
        context=context,
        output_model=ExperienceRewriteProposal,
        run_id=run_id,
    )


def _check_proposal(
    rewrite_proposal: ExperienceRewriteProposal,
    source_role_resume: Resume,
    original_experience: Experience,
) -> _ProposalCheck:
    """Score one proposal: the deterministic truth floor, then the semantic review.

    Three steps:
      1. Rebuild the role from the proposal, taking only its bullet text.
      2. Run the truth floor (bullet count, bullet ids, invented numbers).
      3. Run the LLM review for substance -- but only if step 2 passed, since a
         proposal that fails the floor cannot ship however good its prose is.
    """
    rewritten_role = _rebuild_rewritten_role_from_proposal(
        rewrite_proposal,
        original_experience,
    )
    truthfulness_findings = _collect_rewrite_truthfulness_findings(
        source_role_resume,
        rewrite_proposal,
        rewritten_role,
        original_experience,
    )
    # work_experience[0] is the evidence role: the source role with the candidate's
    # clarification answers folded in. Reviewing against the bare original would
    # flag anything sourced from those answers as unsupported.
    quality_review = (
        ReviewResult(comments=[], summary="", score=None)
        if truthfulness_findings
        else audit_experience_rewrite_quality(
            source_role_resume.work_experience[0],
            rewrite_proposal.rewritten_bullets,
        )
    )
    repair_required_comments, follow_up_comments = _split_comments_by_severity(quality_review)
    return _ProposalCheck(
        rewritten_role=rewritten_role,
        truthfulness_findings=truthfulness_findings,
        repair_required_comments=repair_required_comments,
        follow_up_comments=follow_up_comments,
    )


def _decide_role_rewrite_outcome(
    rewrite_proposal: ExperienceRewriteProposal,
    context: str,
    source_role_resume: Resume,
    original_experience: Experience,
    run_id: str,
) -> RoleRewriteDecision:
    """Choose the best rewrite to ship, repairing once when the first draft falls short.

    The two lines have different force. Truthfulness is non-negotiable: only a
    version that clears the truth floor may ship, and if none survives one repair
    the source bullets do. Substance is best-effort: thin bullets earn a repair, and
    whatever is still thin afterward is surfaced to the candidate, never invented away.

    Four outcomes, in the order they are tried: the first draft ships as-is; the
    repair ships; the first draft ships anyway because the repair regressed; the
    source bullets ship because neither draft was truthful.
    """
    first = _check_proposal(rewrite_proposal, source_role_resume, original_experience)
    if first.is_flawless:
        return RoleRewriteDecision(
            selected_rewrite_proposal=rewrite_proposal,
            finalized_section=_accepted_section(
                first.rewritten_role,
                "Rewrote bullets with grounded role evidence; recruiter-readable and truthful.",
            ),
        )

    logger.info(
        "experience_rewrite_repair_requested",
        run_id=run_id,
        company=original_experience.company_name,
        truthfulness_findings=len(first.truthfulness_findings),
        quality_findings=len(first.repair_required_comments) + len(first.follow_up_comments),
    )
    repair_context = _build_experience_repair_context(
        context,
        rewrite_proposal,
        first.repair_findings,
    )
    repaired_proposal = _request_role_rewrite_proposal(repair_context, run_id=run_id)
    repaired = _check_proposal(repaired_proposal, source_role_resume, original_experience)

    if repaired.is_shippable:
        return RoleRewriteDecision(
            selected_rewrite_proposal=repaired_proposal,
            finalized_section=_accepted_section(
                repaired.rewritten_role,
                "Repaired once with grounded rewrite feedback; truthful."
                + _build_candidate_follow_up_note(repaired.follow_up_comments),
            ),
        )
    if first.is_shippable:
        # The first rewrite was already safe to ship; the repair made it worse, so
        # keep the first rewrite and surface any follow-up gaps it still has.
        return RoleRewriteDecision(
            selected_rewrite_proposal=rewrite_proposal,
            finalized_section=_accepted_section(
                first.rewritten_role,
                "Kept the first grounded rewrite (repair introduced new issues)."
                + _build_candidate_follow_up_note(first.follow_up_comments),
            ),
        )
    logger.warning(
        "experience_rewrite_source_fallback",
        run_id=run_id,
        company=original_experience.company_name,
        findings=repaired.blocking_findings,
    )
    return RoleRewriteDecision(
        selected_rewrite_proposal=repaired_proposal,
        finalized_section=_source_preserved_section(
            original_experience,
            repaired.blocking_findings,
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
        bullet_rewrite.bullet_id for bullet_rewrite in rewrite_proposal.rewritten_bullets
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
    return _render_quality_findings_for_repair(inflation.comments)


def _split_comments_by_severity(
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


def _rebuild_rewritten_role_from_proposal(
    rewrite_proposal: ExperienceRewriteProposal,
    original_experience: Experience,
) -> Experience:
    """Keep only the rewritten bullet text; every other field comes from the source.

    This is the containment boundary: metadata, description, and skills_used can
    never be changed by the LLM, whatever the rewrite proposal contains.
    """
    rewritten_achievements = [
        bullet_rewrite.rewritten_bullet for bullet_rewrite in rewrite_proposal.rewritten_bullets
    ]
    return original_experience.model_copy(update={"achievements": rewritten_achievements})


def _accepted_section(revised_role: Experience, note: str) -> OptimizedExperienceSection:
    """Wrap an accepted (truth-verified) role for the downstream merge."""
    return OptimizedExperienceSection(
        optimized_experiences=[revised_role],
        optimization_notes=note,
    )


def _source_preserved_section(
    experience: Experience, reasons: list[str]
) -> OptimizedExperienceSection:
    """Return the untouched source role after no truthful rewrite survived one repair."""
    return OptimizedExperienceSection(
        optimized_experiences=[experience],
        optimization_notes=(
            "Rewrite could not stay truthful after one repair; original bullets "
            "preserved. Findings: " + "; ".join(reasons)
        ),
    )


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


def _render_quality_findings_for_repair(
    rewrite_comments: list[ReviewComment],
) -> list[str]:
    """Convert structured rewrite comments into concise repair instructions."""
    return [
        f"{review_comment.message}. {review_comment.advice}" for review_comment in rewrite_comments
    ]


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


def _cap_clarifications(
    clarifications: list[ExperienceBulletClarification],
    run_id: str,
) -> list[ExperienceBulletClarification]:
    """Bound how many questions one pause may ask the candidate.

    A candidate handed forty questions answers none of them carefully, so the
    review's own recall becomes self-defeating past a point. Roles are processed
    in resume order, so truncating keeps whole earlier (most recent) roles rather
    than a scattering across all of them, and what was dropped is logged rather
    than silently discarded.
    """
    limit = get_config().workflow.max_clarifications_per_run
    if len(clarifications) <= limit:
        return clarifications
    logger.warning(
        "experience_clarifications_capped",
        run_id=run_id,
        requested=len(clarifications),
        kept=limit,
    )
    return clarifications[:limit]


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
