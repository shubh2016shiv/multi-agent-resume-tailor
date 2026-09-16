"""Graph node that ranks existing resume skills with bounded LLM judgment."""

from src.agents.skill_optimizer import create_skill_optimizer_agent
from src.core.logger import get_logger
from src.data_models.job import JobDescription
from src.data_models.resume import (
    OptimizedSkillsSection,
    Resume,
    Skill,
    SkillCategory,
    SkillRankingDecision,
    SkillsRankingResponse,
)
from src.formatters.skills_optimizer_formatter import format_skills_optimizer_context
from src.orchestration.crew_task_execution import run_agent_task
from src.orchestration.exceptions import AgentOutputError
from src.orchestration.nodes.node_contract import StateUpdate
from src.orchestration.nodes.node_logging import log_node_execution
from src.orchestration.state import ResumeEnhancementPipelineState, require

logger = get_logger(__name__)
SKILLS_OUTPUT_ACTION = (
    "Retry the run. If the same Skills output repeats, inspect its debug checkpoint."
)


@log_node_execution("optimize_skills")
def optimize_skills(state: ResumeEnhancementPipelineState) -> StateUpdate:
    """Rank parsed skills semantically, then assemble the final section in code."""
    resume = require(state["resume"], "resume")
    job = require(state["job_description"], "job_description")
    rankings = _request_rankings(resume, job, state["run_id"])
    return {"optimized_skills": _assemble_skills(resume, job, rankings)}


def _request_rankings(
    resume: Resume,
    job: JobDescription,
    run_id: str,
) -> SkillsRankingResponse:
    """Ask the Skills LLM only for ID-based ranking and grouping decisions."""
    context = format_skills_optimizer_context(resume, job)
    logger.info(
        "skills_context_built",
        resume_skill_count=len(resume.skills),
        job_requirement_count=len(job.requirements),
        context_characters=len(context),
    )
    return run_agent_task(
        agent=create_skill_optimizer_agent(),
        task_name="optimize_skills_section_task",
        context=context,
        output_model=SkillsRankingResponse,
        run_id=run_id,
    )


def _assemble_skills(
    resume: Resume,
    job: JobDescription,
    response: SkillsRankingResponse,
) -> OptimizedSkillsSection:
    """Validate LLM references and reconstruct skills from source objects."""
    skill_ids = {f"S{index:03d}" for index in range(1, len(resume.skills) + 1)}
    job_ids = {f"J{index:03d}" for index in range(1, _relevant_job_count(job) + 1)}
    decisions = _validate_decisions(response.ranked_skills, skill_ids, job_ids)
    ordered = _ordered_source_skills(resume.skills, decisions)
    return OptimizedSkillsSection(
        optimized_skills=ordered,
        skill_categories=_build_categories(ordered),
        optimization_notes="Existing resume skills ranked against parsed job requirements.",
        ats_match_score=_match_score(decisions, job_ids),
    )


def _validate_decisions(
    decisions: list[SkillRankingDecision],
    skill_ids: set[str],
    job_ids: set[str],
) -> list[SkillRankingDecision]:
    """Reject invented and duplicate references before using model decisions."""
    returned_ids = [decision.resume_skill_id for decision in decisions]
    if len(returned_ids) != len(set(returned_ids)):
        _raise_invalid_ranking("returned duplicate resume skill IDs")
    if unknown := set(returned_ids) - skill_ids:
        _raise_invalid_ranking(f"returned unknown resume skill IDs: {sorted(unknown)}")
    referenced_jobs = {
        item for decision in decisions for item in decision.matched_job_requirement_ids
    }
    if unknown := referenced_jobs - job_ids:
        _raise_invalid_ranking(f"returned unknown job requirement IDs: {sorted(unknown)}")
    return decisions


def _ordered_source_skills(
    skills: list[Skill],
    decisions: list[SkillRankingDecision],
) -> list[Skill]:
    """Apply ranks and categories while preserving omitted source skills at the end."""
    indexed = {f"S{index:03d}": (index, skill) for index, skill in enumerate(skills, start=1)}
    ordered = sorted(decisions, key=lambda item: (item.rank, indexed[item.resume_skill_id][0]))
    output = [
        indexed[item.resume_skill_id][1].model_copy(update={"category": item.category})
        for item in ordered
    ]
    returned_ids = {item.resume_skill_id for item in decisions}
    output.extend(skill for skill_id, (_, skill) in indexed.items() if skill_id not in returned_ids)
    return output


def _build_categories(skills: list[Skill]) -> list[SkillCategory]:
    """Group ordered skill names without allowing the LLM to regenerate facts."""
    grouped: dict[str, list[str]] = {}
    for skill in skills:
        grouped.setdefault(skill.category or "Other Skills", []).append(skill.skill_name)
    return [SkillCategory(name=name, skills=names) for name, names in grouped.items()]


def _relevant_job_count(job: JobDescription) -> int:
    """Count requirements included by the formatter's must/should filter."""
    return sum(item.importance.value in {"must_have", "should_have"} for item in job.requirements)


def _match_score(decisions: list[SkillRankingDecision], job_ids: set[str]) -> float:
    """Calculate transparent requirement coverage from validated semantic matches."""
    if not job_ids:
        return 0.0
    matched = {item for decision in decisions for item in decision.matched_job_requirement_ids}
    return round(len(matched) / len(job_ids) * 100, 1)


def _raise_invalid_ranking(reason: str) -> None:
    """Raise one caller-friendly error for an invalid semantic ranking response."""
    raise AgentOutputError(
        stage="Skills Section Prioritizer (optimize_skills_section_task)",
        reason=reason,
        user_action=SKILLS_OUTPUT_ACTION,
    )


__all__ = ["optimize_skills"]
