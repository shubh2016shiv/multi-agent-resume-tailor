"""Build context for the skills optimizer.

Caller:
- `src/orchestration/nodes/skills.py`

Consumer:
- the `optimize_skills_section_task`

This formatter keeps:
- the candidate's current skills
- compact per-role evidence the skills agent can legitimately reason from
- the job requirements and ATS keywords those skills should cover
- the strategy guidance for ordering only

This formatter drops:
- full work-experience bullets beyond the compact evidence slice
- education and contact information
- full job text and unrelated metadata
- token-tracking and feature-flag formatting tricks

Toy example:
    If the resume has twenty skills, this formatter keeps that skill list plus
    the job targets and returns one compact context string.
"""

from typing import Any

from src.data_models.job import JobDescription, SkillImportance
from src.data_models.resume import OptimizedSkillsSection, Resume
from src.data_models.strategy import AlignmentStrategy
from src.formatters.llm_context_rendering import OutputFormat, render_context_data


def select_resume_context(resume: Resume) -> dict[str, Any]:
    """Keep skills plus compact role evidence the optimizer can verify against."""
    return {
        "skills": [
            {
                "skill_name": skill.skill_name,
                "category": skill.category,
                "proficiency_level": skill.proficiency_level,
                "years_of_experience": skill.years_of_experience,
            }
            for skill in resume.skills
        ],
        "role_evidence": [
            {
                "job_title": experience.job_title,
                "skills_used": list(experience.skills_used),
                "top_achievements": experience.achievements[:2],
            }
            for experience in resume.work_experience
        ],
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
    """Keep only the ordering guidance the skills optimizer should read."""
    return {
        "skills_guidance": strategy.skills_guidance,
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


def select_rewrite_skills(section: OptimizedSkillsSection) -> list[dict[str, Any]]:
    """Keep only the name and category of each skill the rewrite must re-emit.

    proficiency, justification, evidence, and confidence are internal metadata the
    assembled resume never renders (it groups skill names under categories), so they
    are dropped here. A leaner context keeps the model on its one rewrite task --
    drop the flagged skills, keep the rest -- instead of re-deriving a large blob.
    """
    return [
        {"skill_name": skill.skill_name, "category": skill.category}
        for skill in section.optimized_skills
    ]


def build_skills_rewrite_payload(
    section: OptimizedSkillsSection,
    skills_to_remove: list[str],
) -> dict[str, Any]:
    """Build the minimal correction payload: the current skills and the names to drop."""
    return {
        "current_skills": select_rewrite_skills(section),
        "skills_to_remove": list(skills_to_remove),
    }


def format_skills_rewrite_context(
    section: OptimizedSkillsSection,
    skills_to_remove: list[str],
    format_type: OutputFormat = "toon",
) -> str:
    """Return the scoped context for one skills-correction (rewrite) pass.

    Deliberately omits the job requirements, ats_keywords, role evidence, and strategy
    the first pass needed: evidence judgement is already done by the audit, so the
    rewrite only needs the current skill list and the exact names to remove.
    """
    ####################################################
    # STEP 1: BUILD THE MINIMAL CORRECTION PAYLOAD#
    ####################################################
    payload = build_skills_rewrite_payload(section, skills_to_remove)

    ####################################################
    # STEP 2: RENDER THAT PAYLOAD INTO THE REQUESTED OUTPUT FORMAT#
    ####################################################
    return render_context_data(
        payload,
        format_type=format_type,
        description="Skills Rewrite Context",
    )
