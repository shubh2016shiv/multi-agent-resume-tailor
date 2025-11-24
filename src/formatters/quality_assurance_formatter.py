"""
Quality Assurance Agent Formatter
==================================

Formats and filters data specifically for the Quality Assurance Agent to reduce
token usage. This formatter:

1. Extracts only the `resume` field from OptimizedResume (excludes redundant
   markdown_content, json_content, and metadata)
2. Filters job description to essential fields
3. Converts both to TOON or Markdown format based on format_type parameter
4. Includes original resume for accuracy comparison

Expected token reduction: ~80% (from 115K to ~20-25K tokens) with TOON format
"""

from typing import Any

from src.core.logger import get_logger
from src.data_models.job import JobDescription
from src.data_models.resume import Resume
from src.formatters.base_formatter import FormatType, estimate_tokens, format_data

# Import OptimizedResume from the ATS agent
try:
    from src.agents.ats_optimization_agent import OptimizedResume
except ImportError:
    # Fallback for direct script execution
    import sys
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    from src.agents.ats_optimization_agent import OptimizedResume

logger = get_logger(__name__)


def _filter_resume_for_qa(resume_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Filter resume dictionary to include only fields needed for QA evaluation.

    QA agent needs:
    - For accuracy: work_experience (with descriptions, dates, companies), skills, education
    - For relevance: professional_summary, work_experience, skills
    - For ATS: all resume fields but in compact format

    Args:
        resume_dict: Resume dictionary from model_dump()

    Returns:
        Filtered resume dictionary with only QA-relevant fields
    """
    # Keep all resume fields - they're all needed for comprehensive QA evaluation
    # The main optimization is removing redundant OptimizedResume wrapper fields
    return resume_dict


def _filter_job_for_qa(job_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Filter job description dictionary to include only fields needed for QA evaluation.

    QA agent needs:
    - job_title, company_name (for context)
    - requirements (for relevance evaluation)
    - ats_keywords (for ATS evaluation)
    - summary (for context, but can be truncated)

    Args:
        job_dict: Job description dictionary from model_dump()

    Returns:
        Filtered job dictionary with only QA-relevant fields
    """
    filtered = {}

    # Essential fields for QA evaluation
    essential_fields = {
        "job_title",
        "company_name",
        "requirements",  # Needed for relevance evaluation
        "ats_keywords",  # Needed for ATS evaluation
        "summary",  # Brief context (can be truncated if too long)
    }

    for field in essential_fields:
        if field in job_dict:
            value = job_dict[field]

            filtered[field] = value

    return filtered


def format_quality_assurance_context(
    optimized_resume: OptimizedResume,
    original_resume: Resume,
    job: JobDescription,
    format_type: FormatType = "toon",
) -> str:
    """
    Format context data for Quality Assurance Agent using specified format (TOON or Markdown).

    This function:
    1. Extracts only the `resume` field from OptimizedResume (removes redundant
       markdown_content, json_content, and metadata)
    2. Filters job description to essential fields
    3. Converts both to TOON or Markdown format based on format_type
    4. Includes original resume for accuracy comparison

    Args:
        optimized_resume: OptimizedResume object containing the tailored resume
        original_resume: Original Resume object for accuracy comparison
        job: JobDescription object for relevance and ATS evaluation
        format_type: Format to use ("toon" for token reduction, "markdown" for readability)

    Returns:
        Formatted context string in the requested format ready for LLM input

    Example:
        >>> context = format_quality_assurance_context(
        ...     optimized_resume, original_resume, job, format_type="toon"
        ... )
        >>> # Returns TOON-formatted string with all three data structures
    """
    try:
        logger.info(f"Formatting QA context: filtering and converting to {format_type.upper()}")

        # Extract only the resume field from OptimizedResume (exclude redundant fields)
        optimized_resume_dict = optimized_resume.resume.model_dump()
        original_resume_dict = original_resume.model_dump()
        job_dict = job.model_dump()

        # Filter job description to essential fields only
        filtered_job = _filter_job_for_qa(job_dict)

        # Convert to requested format
        optimized_resume_formatted = format_data(
            optimized_resume_dict, format_type=format_type, description="Tailored Resume"
        )
        original_resume_formatted = format_data(
            original_resume_dict, format_type=format_type, description="Original Resume"
        )
        job_formatted = format_data(
            filtered_job, format_type=format_type, description="Job Description"
        )

        # Build formatted context with appropriate separators
        if format_type == "markdown":
            context = f"""# Quality Assurance Evaluation Context

{original_resume_formatted}

---

{optimized_resume_formatted}

---

{job_formatted}"""
        else:  # TOON format
            context = f"""ORIGINAL RESUME (for accuracy comparison):
{original_resume_formatted}

TAILORED RESUME (to be evaluated):
{optimized_resume_formatted}

JOB DESCRIPTION:
{job_formatted}"""

        # Log token reduction metrics
        original_size = len(
            f"FINAL RESUME:\n{optimized_resume.model_dump_json()}\n\n"
            f"JOB DESCRIPTION:\n{job.model_dump_json()}"
        )
        new_size = len(context)
        reduction_pct = (
            ((original_size - new_size) / original_size * 100) if original_size > 0 else 0
        )

        estimated_original_tokens = estimate_tokens(
            f"FINAL RESUME:\n{optimized_resume.model_dump_json()}\n\n"
            f"JOB DESCRIPTION:\n{job.model_dump_json()}"
        )
        estimated_new_tokens = estimate_tokens(context)
        token_reduction_pct = (
            ((estimated_original_tokens - estimated_new_tokens) / estimated_original_tokens * 100)
            if estimated_original_tokens > 0
            else 0
        )

        logger.info(
            f"QA context formatted ({format_type.upper()}): {reduction_pct:.1f}% size reduction, "
            f"~{token_reduction_pct:.1f}% token reduction "
            f"({estimated_original_tokens} -> {estimated_new_tokens} tokens)"
        )

        return context

    except Exception as e:
        logger.error(f"Error formatting QA context: {e}", exc_info=True)
        # Fallback to JSON format if conversion fails
        logger.warning("Falling back to JSON format for QA context")
        return (
            f"ORIGINAL RESUME:\n{original_resume.model_dump_json()}\n\n"
            f"TAILORED RESUME:\n{optimized_resume.resume.model_dump_json()}\n\n"
            f"JOB DESCRIPTION:\n{job.model_dump_json()}"
        )
