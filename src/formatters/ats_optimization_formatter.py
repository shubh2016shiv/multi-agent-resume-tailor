"""Build context for the ATS resume assembly agent.

Caller:
- `src/orchestration/nodes/assembly.py`

Consumer:
- the `optimize_ats_resume_task`

This formatter keeps:
- the chosen summary text
- the optimized experience entries
- the optimized skills
- the original contact, education, certification, and language sections
- the job requirements and ATS keywords needed during assembly

This formatter drops:
- draft history
- optimization notes and scoring metadata
- full job text and unrelated job metadata

Toy example:
    The ATS assembler receives one payload with `assembled_sections`,
    `preserved_resume_sections`, and `ats_validation_context`.
"""

from typing import Any

from src.agents.professional_experience.models import OptimizedExperienceSection
from src.agents.professional_summary.models import ProfessionalSummary
from src.data_models.job import JobDescription
from src.data_models.resume import OptimizedSkillsSection, Resume
from src.formatters.llm_context_rendering import OutputFormat, render_context_data


def choose_summary_text(professional_summary: ProfessionalSummary) -> str:
    """Return the summary draft the assembler should place in the final resume."""
    for draft in professional_summary.drafts:
        if draft.version_name == professional_summary.recommended_version:
            return draft.content
    if professional_summary.drafts:
        return professional_summary.drafts[0].content
    raise ValueError("ProfessionalSummary must contain at least one draft.")


def select_experience_entries(optimized_experience: OptimizedExperienceSection) -> list[dict[str, Any]]:
    """Keep only the optimized experience entries the assembler should use."""
    return [
        experience.model_dump(mode="json")
        for experience in optimized_experience.optimized_experiences
    ]


def select_optimized_skills(optimized_skills: OptimizedSkillsSection) -> list[dict[str, Any]]:
    """Keep only the optimized skill entries the assembler should use."""
    return [
        skill.model_dump(mode="json")
        for skill in optimized_skills.optimized_skills
    ]


def select_resume_context(original_resume: Resume) -> dict[str, Any]:
    """Keep only the original resume sections that must carry forward unchanged."""
    return {
        "full_name": original_resume.full_name,
        "email": original_resume.email,
        "phone_number": original_resume.phone_number,
        "location": original_resume.location,
        "website_or_portfolio": original_resume.website_or_portfolio,
        "education": [education.model_dump(mode="json") for education in original_resume.education],
        "certifications": list(original_resume.certifications),
        "languages": list(original_resume.languages),
    }


def select_job_context(job_description: JobDescription) -> dict[str, Any]:
    """Keep only the job fields the assembler should validate against."""
    return {
        "job_title": job_description.job_title,
        "requirements": [requirement.model_dump(mode="json") for requirement in job_description.requirements],
        "ats_keywords": list(job_description.ats_keywords),
    }


def build_ats_optimization_payload(
    professional_summary: ProfessionalSummary,
    optimized_experience: OptimizedExperienceSection,
    optimized_skills: OptimizedSkillsSection,
    original_resume: Resume,
    job_description: JobDescription,
) -> dict[str, Any]:
    """Build the filtered payload for the ATS assembler."""
    ####################################################
    # STEP 1: KEEP ONLY THE OPTIMIZED SECTIONS THAT MUST BE ASSEMBLED#
    ####################################################
    assembled_sections = {
        "professional_summary_text": choose_summary_text(professional_summary),
        "optimized_experience_entries": select_experience_entries(optimized_experience),
        "optimized_skills": select_optimized_skills(optimized_skills),
    }

    ####################################################
    # STEP 2: KEEP ONLY THE ORIGINAL RESUME SECTIONS THAT MUST BE PRESERVED#
    ####################################################
    preserved_resume_sections = select_resume_context(original_resume)

    ####################################################
    # STEP 3: KEEP ONLY THE JOB SIGNALS USED FOR ATS VALIDATION#
    ####################################################
    job_context = select_job_context(job_description)

    return {
        "assembled_sections": assembled_sections,
        "preserved_resume_sections": preserved_resume_sections,
        "ats_validation_context": job_context,
    }


def format_ats_optimization_context(
    professional_summary: ProfessionalSummary,
    optimized_experience: OptimizedExperienceSection,
    optimized_skills: OptimizedSkillsSection,
    original_resume: Resume,
    job_description: JobDescription,
    format_type: OutputFormat = "toon",
) -> str:
    """Return the ATS assembler's context string."""
    ####################################################
    # STEP 1: BUILD THE SMALL DATA PAYLOAD THE ATS ASSEMBLER ACTUALLY NEEDS#
    ####################################################
    payload = build_ats_optimization_payload(
        professional_summary,
        optimized_experience,
        optimized_skills,
        original_resume,
        job_description,
    )

    ####################################################
    # STEP 2: RENDER THAT PAYLOAD INTO THE REQUESTED OUTPUT FORMAT#
    ####################################################
    return render_context_data(
        payload,
        format_type=format_type,
        description="ATS Optimization Context",
    )
