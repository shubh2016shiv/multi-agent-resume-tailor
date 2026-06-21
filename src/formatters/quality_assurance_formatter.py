"""Build context for the quality assurance reviewer.

Caller:
- `src/orchestration/nodes/quality.py`

Consumer:
- the `assess_quality_task`

This formatter keeps:
- the original resume for truthfulness comparison
- the tailored resume that must be audited
- the job requirements and keywords the tailored resume should answer

This formatter drops:
- ATS optimizer metadata outside the final assembled resume
- full job text
- fallback JSON dump behavior

Toy example:
    The reviewer receives three clearly named sections: original resume,
    tailored resume, and target job.
"""

from typing import Any

from src.agents.ats_optimizer.models import AtsOptimizedResume
from src.data_models.job import JobDescription
from src.data_models.resume import Resume
from src.formatters.llm_context_rendering import OutputFormat, render_context_data


def select_original_resume_context(original_resume: Resume) -> dict[str, Any]:
    """Keep the original resume exactly as the reviewer should compare it."""
    return original_resume.model_dump(mode="json")


def select_tailored_resume_context(optimized_resume: AtsOptimizedResume) -> dict[str, Any]:
    """Keep only the final assembled resume from the ATS optimizer output."""
    return optimized_resume.final_resume.model_dump(mode="json")


def select_job_context(job_description: JobDescription) -> dict[str, Any]:
    """Keep only the job fields the reviewer needs during scoring."""
    return {
        "job_title": job_description.job_title,
        "company_name": job_description.company_name,
        "summary": job_description.summary,
        "requirements": [requirement.model_dump(mode="json") for requirement in job_description.requirements],
        "ats_keywords": list(job_description.ats_keywords),
    }


def build_quality_assurance_payload(
    optimized_resume: AtsOptimizedResume,
    original_resume: Resume,
    job: JobDescription,
) -> dict[str, Any]:
    """Build the filtered payload for the quality assurance reviewer."""
    ####################################################
    # STEP 1: KEEP THE SOURCE RESUME THE REVIEWER MUST VERIFY AGAINST#
    ####################################################
    original_resume_context = select_original_resume_context(original_resume)

    ####################################################
    # STEP 2: KEEP THE FINAL TAILORED RESUME THAT MUST BE AUDITED#
    ####################################################
    tailored_resume_context = select_tailored_resume_context(optimized_resume)

    ####################################################
    # STEP 3: KEEP THE JOB SIGNALS THE REVIEW SHOULD SCORE AGAINST#
    ####################################################
    job_context = select_job_context(job)

    return {
        "original_resume": original_resume_context,
        "tailored_resume": tailored_resume_context,
        "target_job": job_context,
    }


def format_quality_assurance_context(
    optimized_resume: AtsOptimizedResume,
    original_resume: Resume,
    job: JobDescription,
    format_type: OutputFormat = "toon",
) -> str:
    """Return the quality assurance reviewer's context string."""
    ####################################################
    # STEP 1: BUILD THE SMALL DATA PAYLOAD THE REVIEWER ACTUALLY NEEDS#
    ####################################################
    payload = build_quality_assurance_payload(optimized_resume, original_resume, job)

    ####################################################
    # STEP 2: RENDER THAT PAYLOAD INTO THE REQUESTED OUTPUT FORMAT#
    ####################################################
    return render_context_data(
        payload,
        format_type=format_type,
        description="Quality Assurance Context",
    )
