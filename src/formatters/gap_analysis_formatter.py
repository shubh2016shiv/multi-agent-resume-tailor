"""Build context for the gap analysis strategist.

Caller:
- `src/orchestration/nodes/strategy.py`

Consumer:
- the `create_alignment_strategy_task`

This formatter keeps:
- the candidate's skills, experience, education, certifications, and summary
- the job title, level, summary, requirements, and ATS keywords
- the code-computed match report the strategist should build on

This formatter drops:
- contact information
- full job text
- unrelated metadata that does not affect gap analysis

Toy example:
    The strategist receives one payload with `candidate_profile`,
    `target_job`, and `current_match_report`.
"""

from typing import Any

from src.data_models.job import JobDescription  # the target job to compare the candidate against
from src.data_models.resume import Resume  # the candidate's current resume

# Shared rendering: OutputFormat is the "toon"/"markdown" choice; render_context_data
# turns this formatter's filtered payload dict into the final LLM context string.
from src.formatters.llm_context_rendering import OutputFormat, render_context_data

# The code-computed match report (score + per-section findings) the strategist
# builds on -- a deterministic tool result, not an agent's own opinion.
from src.tools.contracts import ReviewResult


def select_resume_context(resume: Resume) -> dict[str, Any]:
    """Keep only the resume fields the strategist needs for comparison."""
    return {
        "professional_summary": resume.professional_summary,
        "skills": [
            {
                "skill_name": skill.skill_name,
                "category": skill.category,
                "proficiency_level": skill.proficiency_level,
                "years_of_experience": skill.years_of_experience,
            }
            for skill in resume.skills
        ],
        "work_experience": [
            {
                "job_title": experience.job_title,
                "company_name": experience.company_name,
                "date_range": f"{experience.start_date} - {experience.end_date or 'Present'}",
                "description": experience.description,
                "achievements": experience.achievements[:3],
                "skills_used": list(experience.skills_used),
            }
            for experience in resume.work_experience
        ],
        "education": [
            {
                "degree": education.degree,
                "field_of_study": education.field_of_study,
                "institution_name": education.institution_name,
                "graduation_year": education.graduation_year,
                "gpa": education.gpa,
                "honors": education.honors,
            }
            for education in resume.education
        ],
        "certifications": list(resume.certifications),
        "languages": list(resume.languages),
    }


def select_job_context(job_description: JobDescription) -> dict[str, Any]:
    """Keep only the job fields the strategist needs for comparison."""
    return {
        "job_title": job_description.job_title,
        "company_name": job_description.company_name,
        "job_level": job_description.job_level.value,
        "summary": job_description.summary,
        "requirements": [requirement.model_dump(mode="json") for requirement in job_description.requirements],
        "ats_keywords": list(job_description.ats_keywords),
    }


def select_match_report_context(match_report: ReviewResult) -> dict[str, Any]:
    """Keep only the code-owned match findings the strategist should build on."""
    return {
        "score": match_report.score,
        "summary": match_report.summary,
        "findings": [
            {
                "severity": comment.severity.value,
                "section": comment.location.section.value,
                "message": comment.message,
                "advice": comment.advice,
            }
            for comment in match_report.comments
        ],
    }


def build_gap_analysis_payload(
    resume: Resume,
    job_description: JobDescription,
    match_report: ReviewResult,
) -> dict[str, Any]:
    """Build the filtered payload for the gap analysis strategist."""
    ####################################################
    # STEP 1: KEEP ONLY THE CANDIDATE SIGNALS NEEDED FOR COMPARISON
    ####################################################
    resume_context = select_resume_context(resume)

    ####################################################
    # STEP 2: KEEP ONLY THE JOB SIGNALS THE STRATEGIST SHOULD ANSWER
    ####################################################
    job_context = select_job_context(job_description)

    ####################################################
    # STEP 3: KEEP THE CODE-COMPUTED MATCH REPORT SO THE AGENT BUILDS ON FACTS
    ####################################################
    match_report_context = select_match_report_context(match_report)

    return {
        "candidate_profile": resume_context,
        "target_job": job_context,
        "current_match_report": match_report_context,
    }


def format_gap_analysis_context(
    resume: Resume,
    job_description: JobDescription,
    match_report: ReviewResult,
    format_type: OutputFormat = "toon",
) -> str:
    """Return the gap analysis strategist's context string."""
    ####################################################
    # STEP 1: BUILD THE SMALL DATA PAYLOAD THE STRATEGIST ACTUALLY NEEDS
    ####################################################
    payload = build_gap_analysis_payload(resume, job_description, match_report)

    ####################################################
    # STEP 2: RENDER THAT PAYLOAD INTO THE REQUESTED OUTPUT FORMAT
    ####################################################
    return render_context_data(
        payload,
        format_type=format_type,
        description="Gap Analysis Context",
    )
