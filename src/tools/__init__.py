"""
Agent Tools Package — public surface re-exported for agent use.

Agents import from here, not from sub-packages directly.
"""

from .agent_facing_tools import (
    analyze_jd_keyword_coverage,
    audit_summary,
    audit_truthfulness,
    check_resume_markdown_quality,
    check_skills_evidence,
    convert_resume_document_to_markdown,
    extract_structured_resume_from_markdown,
    redact_pii_from_resume_markdown,
    validate_ats_compliance,
)
from .ats_compliance import (
    check_section_headers,
    get_incompatible_patterns,
    get_standard_headers,
    validate_ats_formatting,
)
from .document_ingestion import (
    convert_document_to_markdown,
    get_supported_formats,
    is_format_supported,
)
from .job_matching import (
    calculate_keyword_density,
    get_optimal_keyword_density_range,
)

__all__ = [
    # Document ingestion
    "convert_document_to_markdown",
    "get_supported_formats",
    "is_format_supported",
    # ATS compliance
    "validate_ats_formatting",
    "check_section_headers",
    "get_incompatible_patterns",
    "get_standard_headers",
    # Job matching (keyword coverage)
    "calculate_keyword_density",
    "get_optimal_keyword_density_range",
    # Agent-facing composite tools
    "audit_summary",
    "check_skills_evidence",
    "audit_truthfulness",
    "validate_ats_compliance",
    "analyze_jd_keyword_coverage",
    "check_resume_markdown_quality",
    "convert_resume_document_to_markdown",
    "redact_pii_from_resume_markdown",
    "extract_structured_resume_from_markdown",
]
