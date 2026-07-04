"""Build context for the professional experience optimizer.

Caller:
- `src/orchestration/nodes/experience.py`

Consumer:
- the `optimize_experience_section_task`

This formatter keeps:
- the role entry whose bullets will be rewritten
- a compact, prioritized JD-fit signal for that role
- short experience-specific strategy guidance

This formatter drops:
- resume contact information
- education and unrelated static sections
- full job text and unrelated job metadata

Toy example:
    If the resume contains one role-scoped payload, this formatter returns a
    small context string centered on that role's source bullets, role
    description, skills_used, and top JD requirements.
"""

from typing import Any

from src.data_models.job import JobDescription
from src.data_models.resume import Resume
from src.data_models.strategy import AlignmentStrategy
from src.formatters.llm_context_rendering import OutputFormat, render_context_data
from src.hitl.professional_experience.models import (
    ExperienceBulletClarification,
    build_experience_bullet_id,
)

MAX_PRIORITY_REQUIREMENTS_FOR_EXPERIENCE_REWRITE = 6


def select_resume_context(
    resume: Resume,
    clarification_answers: list[ExperienceBulletClarification] | None = None,
) -> dict[str, Any]:
    """Keep only the role evidence the optimizer may rewrite."""
    clarification_answers = clarification_answers or []
    return {
        "work_experience": [
            {
                "job_title": experience.job_title,
                "company_name": experience.company_name,
                "description": experience.description,
                "source_bullets": [
                    {
                        "bullet_id": build_experience_bullet_id(experience, bullet_index),
                        "text": bullet_text,
                    }
                    for bullet_index, bullet_text in enumerate(experience.achievements)
                ],
                "skills_used": list(experience.skills_used),
                "candidate_clarification_evidence": _clarification_context_for_role(
                    experience,
                    clarification_answers,
                ),
            }
            for experience in resume.work_experience
        ]
    }


def select_job_context(job_description: JobDescription) -> dict[str, Any]:
    """Keep a compact, prioritized JD-fit signal for the rewrite."""
    prioritized_requirements = [
        requirement
        for _, requirement in sorted(
            enumerate(job_description.requirements),
            key=lambda requirement_entry: (
                _requirement_importance_rank(
                    requirement_entry[1].importance.value
                ),
                requirement_entry[0],
            ),
        )[:MAX_PRIORITY_REQUIREMENTS_FOR_EXPERIENCE_REWRITE]
    ]
    return {
        "job_title": job_description.job_title,
        "priority_requirements": [
            requirement.model_dump(mode="json") for requirement in prioritized_requirements
        ],
    }


def select_strategy_context(strategy: AlignmentStrategy) -> dict[str, Any]:
    """Keep only the short strategy guidance the experience optimizer should read."""
    return {"experience_guidance": strategy.experience_guidance.strip()}


def build_experience_optimizer_payload(
    resume: Resume,
    job_description: JobDescription,
    strategy: AlignmentStrategy,
    clarification_answers: list[ExperienceBulletClarification] | None = None,
) -> dict[str, Any]:
    """Build the filtered payload for the experience optimizer."""
    ####################################################
    # STEP 1: KEEP ONLY THE ROLE EVIDENCE THE AGENT IS ALLOWED TO REWRITE#
    ####################################################
    resume_context = select_resume_context(resume, clarification_answers)

    ####################################################
    # STEP 2: KEEP ONLY THE JOB SIGNALS USED TO PRIORITIZE BULLETS#
    ####################################################
    job_context = select_job_context(job_description)

    ####################################################
    # STEP 3: KEEP ONLY THE STRATEGY GUIDANCE MEANT FOR EXPERIENCE WORK#
    ####################################################
    strategy_context = select_strategy_context(strategy)

    return {
        "role_rewrite_context": resume_context,
        "target_role_signal": job_context,
        "rewrite_guidance": strategy_context,
    }


def format_experience_optimizer_context(
    resume: Resume,
    job_description: JobDescription,
    strategy: AlignmentStrategy,
    clarification_answers: list[ExperienceBulletClarification] | None = None,
    format_type: OutputFormat = "toon",
) -> str:
    """Return the experience optimizer's context string."""
    ####################################################
    # STEP 1: BUILD THE SMALL DATA PAYLOAD THE EXPERIENCE OPTIMIZER NEEDS#
    ####################################################
    payload = build_experience_optimizer_payload(
        resume,
        job_description,
        strategy,
        clarification_answers,
    )

    ####################################################
    # STEP 2: RENDER THAT PAYLOAD INTO THE REQUESTED OUTPUT FORMAT#
    ####################################################
    return render_context_data(
        payload,
        format_type=format_type,
        description="Experience Optimizer Context",
    )


def _requirement_importance_rank(requirement_importance: str) -> int:
    """Return a stable rank so must-have requirements appear before nice-to-have ones."""
    importance_rank = {
        "must_have": 0,
        "should_have": 1,
        "nice_to_have": 2,
    }
    return importance_rank.get(requirement_importance, 3)


def _clarification_context_for_role(
    experience,
    clarification_answers: list[ExperienceBulletClarification],
) -> list[dict[str, str]]:
    """Return answered clarification evidence already known for this role's bullets."""
    bullet_ids_for_role = {
        build_experience_bullet_id(experience, bullet_index)
        for bullet_index, _ in enumerate(experience.achievements)
    }
    return [
        {
            "bullet_id": clarification.bullet_id,
            "source_bullet": clarification.bullet,
            "candidate_answer": clarification.answer,
        }
        for clarification in clarification_answers
        if clarification.answer.strip() and clarification.bullet_id in bullet_ids_for_role
    ]
