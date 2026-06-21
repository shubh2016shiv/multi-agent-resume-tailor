"""Agent-facing tools that the LLM is allowed to choose at runtime."""

from .ingestion_tools import (
    check_resume_markdown_quality,
    convert_resume_document_to_markdown,
    extract_structured_resume_from_markdown,
    redact_pii_from_resume_markdown,
)
from .resume_review_tools import (
    analyze_jd_keyword_coverage,
    audit_summary,
    audit_truthfulness,
    check_skills_evidence,
    validate_ats_compliance,
)

__all__ = [
    "analyze_jd_keyword_coverage",
    "audit_summary",
    "audit_truthfulness",
    "check_resume_markdown_quality",
    "check_skills_evidence",
    "convert_resume_document_to_markdown",
    "extract_structured_resume_from_markdown",
    "redact_pii_from_resume_markdown",
    "validate_ats_compliance",
]
