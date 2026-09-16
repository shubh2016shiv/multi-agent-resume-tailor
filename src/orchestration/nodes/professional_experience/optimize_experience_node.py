"""Graph node that rewrites each experience role once and keeps only safe results."""

from src.agents.professional_experience import create_professional_experience_agent
from src.agents.professional_experience.models import (
    ExperienceRewriteProposal,
    OptimizedExperienceSection,
)
from src.core.logger import get_logger
from src.core.settings import get_config
from src.data_models.resume import Experience, Resume
from src.formatters.experience_optimizer_formatter import format_experience_optimizer_context
from src.hitl.professional_experience.answers import (
    answers_for_role,
    experience_with_candidate_answers,
)
from src.hitl.professional_experience.clarifications import build_bullet_clarifications
from src.hitl.professional_experience.models import ExperienceBulletClarification
from src.orchestration.crew_task_execution import run_agent_task
from src.orchestration.nodes.node_contract import StateUpdate
from src.orchestration.nodes.node_logging import log_node_execution
from src.orchestration.nodes.professional_experience.experience_truthfulness_rules import (
    select_truthful_rewrite,
)
from src.orchestration.state import ResumeEnhancementPipelineState, require

logger = get_logger(__name__)


@log_node_execution("optimize_experience")
def optimize_experience(state: ResumeEnhancementPipelineState) -> StateUpdate:
    """Rewrite roles once, reject unsafe rewrites, and return candidate questions."""
    resume = require(state["resume"], "resume")
    answers = state.get("clarification_answers") or []
    optimized, clarifications = _optimize_roles(state, resume, answers)
    _log_clarification_flow(state["run_id"], answers, clarifications)
    return {
        "resume": _resume_with_candidate_answers(resume, answers),
        "optimized_experience": optimized,
        "experience_clarifications": clarifications,
        "clarification_answers": [],
    }


def _optimize_roles(
    state: ResumeEnhancementPipelineState,
    resume: Resume,
    answers: list[ExperienceBulletClarification],
) -> tuple[OptimizedExperienceSection, list[ExperienceBulletClarification]]:
    """Optimize roles sequentially and merge their results in resume order."""
    if not resume.work_experience:
        raise ValueError("resume.work_experience must contain at least one entry.")

    outcomes = [
        _optimize_role(state, resume, role, answers_for_role(role, answers))
        for role in resume.work_experience
    ]
    sections = [section for section, _ in outcomes]
    questions = [question for _, items in outcomes for question in items]
    return _merge_sections(sections), _cap_clarifications(questions, state["run_id"])


def _optimize_role(
    state: ResumeEnhancementPipelineState,
    resume: Resume,
    role: Experience,
    answers: list[ExperienceBulletClarification],
) -> tuple[OptimizedExperienceSection, list[ExperienceBulletClarification]]:
    """Rewrite one role once, then accept it or preserve its source evidence."""
    _log_applied_answers(role, answers, state["run_id"])
    evidence_role = experience_with_candidate_answers(role, answers)
    evidence_resume = resume.model_copy(update={"work_experience": [evidence_role]})
    context = _build_role_context(state, resume, role, answers)
    proposal = _request_role_rewrite(context, state["run_id"])
    selected_role, findings = select_truthful_rewrite(
        evidence_resume,
        proposal,
        evidence_role,
    )
    section = _build_role_section(selected_role, findings, state["run_id"])
    questions = build_bullet_clarifications(
        experience=evidence_role,
        shipped_bullets=selected_role.achievements,
        rewritten_bullets=proposal.rewritten_bullets,
        run_id=state["run_id"],
    )
    return section, questions


def _build_role_context(
    state: ResumeEnhancementPipelineState,
    resume: Resume,
    role: Experience,
    answers: list[ExperienceBulletClarification],
) -> str:
    """Build the agent context for one role and its candidate answers."""
    role_resume = resume.model_copy(update={"work_experience": [role]})
    return format_experience_optimizer_context(
        resume=role_resume,
        job_description=require(state["job_description"], "job_description"),
        strategy=require(state["alignment_strategy"], "alignment_strategy"),
        clarification_answers=answers,
        format_type="toon",
    )


def _request_role_rewrite(context: str, run_id: str) -> ExperienceRewriteProposal:
    """Request one structured rewrite from the professional-experience agent."""
    return run_agent_task(
        agent=create_professional_experience_agent(),
        task_name="optimize_experience_section_task",
        context=context,
        output_model=ExperienceRewriteProposal,
        run_id=run_id,
    )


def _build_role_section(
    role: Experience,
    findings: list[str],
    run_id: str,
) -> OptimizedExperienceSection:
    """Wrap the selected role and record a truthful-rewrite fallback."""
    if findings:
        logger.warning(
            "experience_rewrite_source_fallback",
            run_id=run_id,
            company=role.company_name,
            findings=findings,
        )
        note = "Rewrite failed truth checks; source role preserved. Findings: "
        note += "; ".join(findings)
    else:
        note = "Rewrote bullets once; deterministic truth checks passed."
    return OptimizedExperienceSection(
        optimized_experiences=[role],
        optimization_notes=note,
    )


def _merge_sections(
    sections: list[OptimizedExperienceSection],
) -> OptimizedExperienceSection:
    """Merge role-level results while retaining order and unique keywords."""
    experiences = [role for section in sections for role in section.optimized_experiences]
    notes = [section.optimization_notes for section in sections if section.optimization_notes]
    keywords = [keyword for section in sections for keyword in section.keywords_integrated]
    scores = [score for section in sections for score in section.relevance_scores]
    return OptimizedExperienceSection(
        optimized_experiences=experiences,
        optimization_notes="\n".join(notes),
        keywords_integrated=list(dict.fromkeys(keywords)),
        relevance_scores=scores,
    )


def _cap_clarifications(
    clarifications: list[ExperienceBulletClarification],
    run_id: str,
) -> list[ExperienceBulletClarification]:
    """Limit candidate questions while preserving resume order."""
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


def _resume_with_candidate_answers(
    resume: Resume,
    answers: list[ExperienceBulletClarification],
) -> Resume:
    """Apply candidate-owned facts to the downstream source resume."""
    if not answers:
        return resume
    roles = [
        experience_with_candidate_answers(role, answers_for_role(role, answers))
        for role in resume.work_experience
    ]
    return resume.model_copy(update={"work_experience": roles})


def _log_applied_answers(
    role: Experience,
    answers: list[ExperienceBulletClarification],
    run_id: str,
) -> None:
    """Record candidate answers used for one role."""
    if answers:
        logger.info(
            "experience_clarification_answers_applied",
            run_id=run_id,
            company=role.company_name,
            answers=len(answers),
        )


def _log_clarification_flow(
    run_id: str,
    answers: list[ExperienceBulletClarification],
    clarifications: list[ExperienceBulletClarification],
) -> None:
    """Record answers consumed and new questions raised by the stage."""
    logger.info(
        "experience_clarification_flow",
        run_id=run_id,
        answered_clarifications=len(answers),
        clarifications_requested=len(clarifications),
    )


__all__ = ["optimize_experience"]
