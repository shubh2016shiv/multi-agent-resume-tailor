"""Build context for the skills optimizer.

Caller:
- `src/orchestration/nodes/skills.py`

Consumer:
- the `optimize_skills_section_task`

This formatter keeps:
- the candidate's current skills
- the job requirements and ATS keywords those skills should cover
- the strategy guidance for ordering and gap handling

This formatter drops:
- work experience, education, and contact information
- full job text and unrelated metadata
- token-tracking and feature-flag formatting tricks

Toy example:
    If the resume has twenty skills, this formatter keeps that skill list plus
    the job targets and returns one compact context string.
"""

from typing import Any

from src.data_models.job import JobDescription, SkillImportance
from src.data_models.resume import Resume
from src.data_models.strategy import AlignmentStrategy
from src.formatters.llm_context_rendering import OutputFormat, render_context_data


def select_resume_context(resume: Resume) -> dict[str, Any]:
    """Keep only the current skills section the optimizer should reshape."""
    return {
        "skills": [
            {
                "skill_name": skill.skill_name,
                "category": skill.category,
                "proficiency_level": skill.proficiency_level,
                "years_of_experience": skill.years_of_experience,
            }
            for skill in resume.skills
        ]
    }


def select_job_context(job_description: JobDescription) -> dict[str, Any]:
    """Keep only the job fields the skills optimizer needs."""
    return {
        "job_title": job_description.job_title,
        "requirements": [
            {
                "requirement": requirement.requirement,
                "importance": requirement.importance.value,
                "years_required": requirement.years_required,
            }
            for requirement in job_description.requirements
            if requirement.importance in {SkillImportance.MUST_HAVE, SkillImportance.SHOULD_HAVE}
        ],
        "ats_keywords": list(job_description.ats_keywords),
    }


def select_strategy_context(strategy: AlignmentStrategy) -> dict[str, Any]:
    """Keep only the strategy fields the skills optimizer should read."""
    return {
        "skills_guidance": strategy.skills_guidance,
        "keywords_to_integrate": list(strategy.keywords_to_integrate),
        "identified_matches": [
            {
                "resume_skill": match.resume_skill,
                "job_requirement": match.job_requirement,
                "match_score": match.match_score,
                "justification": match.justification,
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


def build_skills_optimizer_payload(
    resume: Resume,
    job_description: JobDescription,
    strategy: AlignmentStrategy,
) -> dict[str, Any]:
    """Build the filtered payload for the skills optimizer."""
    ####################################################
    # STEP 1: KEEP ONLY THE CURRENT SKILLS THE AGENT IS ALLOWED TO REORDER#
    ####################################################
    resume_context = select_resume_context(resume)

    ####################################################
    # STEP 2: KEEP ONLY THE JOB SIGNALS THE SKILLS SECTION SHOULD ANSWER#
    ####################################################
    job_context = select_job_context(job_description)

    ####################################################
    # STEP 3: KEEP ONLY THE STRATEGY GUIDANCE MEANT FOR SKILL WORK#
    ####################################################
    strategy_context = select_strategy_context(strategy)

    return {
        "current_skills": resume_context,
        "target_job": job_context,
        "skills_strategy": strategy_context,
    }


def format_skills_optimizer_context(
    resume: Resume,
    job_description: JobDescription,
    strategy: AlignmentStrategy,
    format_type: OutputFormat = "toon",
) -> str:
    """Return the skills optimizer's context string."""
    ####################################################
    # STEP 1: BUILD THE SMALL DATA PAYLOAD THE SKILLS OPTIMIZER NEEDS#
    ####################################################
    payload = build_skills_optimizer_payload(resume, job_description, strategy)

    ####################################################
    # STEP 2: RENDER THAT PAYLOAD INTO THE REQUESTED OUTPUT FORMAT#
    ####################################################
    return render_context_data(
        payload,
        format_type=format_type,
        description="Skills Optimizer Context",
    )
