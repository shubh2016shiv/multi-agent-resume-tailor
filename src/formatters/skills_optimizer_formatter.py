"""Build the minimal semantic-ranking context for the Skills LLM."""

from typing import Any

from src.data_models.job import JobDescription, SkillImportance
from src.data_models.resume import Resume
from src.formatters.llm_context_rendering import OutputFormat, render_context_data

RELEVANT_REQUIREMENT_LEVELS = {
    SkillImportance.MUST_HAVE,
    SkillImportance.SHOULD_HAVE,
}


def build_skills_optimizer_payload(
    resume: Resume,
    job_description: JobDescription,
) -> dict[str, Any]:
    """Return only existing skills and high-priority parsed job requirements."""
    return {
        "resume_skills": _resume_skill_inputs(resume),
        "job_requirements": _job_requirement_inputs(job_description),
    }


def format_skills_optimizer_context(
    resume: Resume,
    job_description: JobDescription,
    format_type: OutputFormat = "toon",
) -> str:
    """Render the bounded Skills request without resume narrative or strategy prose."""
    return render_context_data(
        build_skills_optimizer_payload(resume, job_description),
        format_type=format_type,
        description="Skills Ranking Context",
    )


def _resume_skill_inputs(resume: Resume) -> list[dict[str, Any]]:
    """Assign stable request-local IDs to parsed resume skills."""
    return [
        {
            "id": f"S{index:03d}",
            "name": skill.skill_name,
            "canonical_name": skill.canonicalized_skill,
            "current_category": skill.category,
        }
        for index, skill in enumerate(resume.skills, start=1)
    ]


def _job_requirement_inputs(job: JobDescription) -> list[dict[str, Any]]:
    """Assign IDs to parsed must-have and should-have job requirements."""
    relevant = (item for item in job.requirements if item.importance in RELEVANT_REQUIREMENT_LEVELS)
    return [
        {
            "id": f"J{index:03d}",
            "name": item.requirement,
            "canonical_name": item.canonicalized_requirement,
            "importance": item.importance.value,
        }
        for index, item in enumerate(relevant, start=1)
    ]


__all__ = ["build_skills_optimizer_payload", "format_skills_optimizer_context"]
