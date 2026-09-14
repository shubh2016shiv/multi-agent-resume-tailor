"""The two graph nodes for Stage 3, and the fan-out that rewrites each role.

optimize_experience rewrites every role and merges the results.
await_candidate_clarifications is a separate node that pauses the run when a role
raised a question for the candidate. See this package's __init__ for the whole flow.
"""

from concurrent.futures import ThreadPoolExecutor

from langgraph.types import interrupt

from src.agents.professional_experience.models import OptimizedExperienceSection
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
from src.hitl.professional_experience.models import ExperienceBulletClarification
from src.orchestration.nodes._stage import pipeline_stage
from src.orchestration.nodes.experience.rewrite import (
    decide_role_rewrite_outcome,
    request_role_rewrite_proposal,
)
from src.orchestration.state import ResumeEnhancementPipelineState, require

logger = get_logger(__name__)


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
    rewrite_proposal = request_role_rewrite_proposal(context, run_id=run_id)
    rewrite_decision = decide_role_rewrite_outcome(
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
