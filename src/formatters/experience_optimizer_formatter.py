"""Build context for the professional experience optimizer.

Caller:
- `src/orchestration/nodes/experience.py`

Consumer:
- the `optimize_experience_section_task`

This formatter keeps:
- the role entries whose bullets will be reordered
- the job requirements and ATS keywords those bullets should answer
- the experience-specific strategy guidance from gap analysis

This formatter drops:
- resume contact information
- education and unrelated static sections
- full job text and unrelated job metadata

Toy example:
    If the resume contains one role-scoped payload, this formatter returns a
    small context string centered on that role's existing evidence.
"""

from typing import Any

from src.data_models.job import JobDescription
from src.data_models.resume import Resume
from src.data_models.strategy import AlignmentStrategy
from src.formatters.llm_context_rendering import OutputFormat, render_context_data


def select_resume_context(resume: Resume) -> dict[str, Any]:
    """Keep only the work-experience evidence the optimizer may reorder."""
    return {
        "work_experience": [
            {
                "experience_id": experience.experience_id,
                "job_title": experience.job_title,
                "company_name": experience.company_name,
                "start_date": str(experience.start_date),
                "end_date": str(experience.end_date) if experience.end_date else None,
                "is_current_position": experience.is_current_position,
                "location": experience.location,
                "description": experience.description,
                "achievements": list(experience.achievements),
                "skills_used": list(experience.skills_used),
            }
            for experience in resume.work_experience
        ]
    }


def select_job_context(job_description: JobDescription) -> dict[str, Any]:
    """Keep only the job fields the experience optimizer needs."""
    return {
        "job_title": job_description.job_title,
        "requirements": [requirement.model_dump(mode="json") for requirement in job_description.requirements],
        "ats_keywords": list(job_description.ats_keywords),
    }


def select_strategy_context(strategy: AlignmentStrategy) -> dict[str, Any]:
    """Keep only the strategy fields the experience optimizer should read."""
    return {
        "experience_guidance": strategy.experience_guidance,
        "keywords_to_integrate": list(strategy.keywords_to_integrate),
        "identified_matches": [
            {
                "resume_skill": match.resume_skill,
                "job_requirement": match.job_requirement,
                "match_score": match.match_score,
            }
            for match in strategy.identified_matches
        ],
        "identified_gaps": [
            {
                "missing_skill": gap.missing_skill,
                "importance": gap.importance,
                "suggestion": gap.suggestion,
            }
            for gap in strategy.identified_gaps
        ],
    }


def build_experience_optimizer_payload(
    resume: Resume,
    job_description: JobDescription,
    strategy: AlignmentStrategy,
) -> dict[str, Any]:
    """Build the filtered payload for the experience optimizer."""
    ####################################################
    # STEP 1: KEEP ONLY THE ROLE EVIDENCE THE AGENT IS ALLOWED TO REORDER#
    ####################################################
    resume_context = select_resume_context(resume)

    ####################################################
    # STEP 2: KEEP ONLY THE JOB SIGNALS USED TO PRIORITIZE BULLETS#
    ####################################################
    job_context = select_job_context(job_description)

    ####################################################
    # STEP 3: KEEP ONLY THE STRATEGY GUIDANCE MEANT FOR EXPERIENCE WORK#
    ####################################################
    strategy_context = select_strategy_context(strategy)

    return {
        "resume_work_experience": resume_context,
        "target_job": job_context,
        "experience_strategy": strategy_context,
    }


def format_experience_optimizer_context(
    resume: Resume,
    job_description: JobDescription,
    strategy: AlignmentStrategy,
    format_type: OutputFormat = "toon",
) -> str:
    """Return the experience optimizer's context string."""
    ####################################################
    # STEP 1: BUILD THE SMALL DATA PAYLOAD THE EXPERIENCE OPTIMIZER NEEDS#
    ####################################################
    payload = build_experience_optimizer_payload(resume, job_description, strategy)

    ####################################################
    # STEP 2: RENDER THAT PAYLOAD INTO THE REQUESTED OUTPUT FORMAT#
    ####################################################
    return render_context_data(
        payload,
        format_type=format_type,
        description="Experience Optimizer Context",
    )
