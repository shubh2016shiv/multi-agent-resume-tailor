"""
ATS section header validation: check that essential resume sections are present
under names an ATS recognizes. Matching is case-insensitive and markdown-aware.
"""

import re

from crewai.tools import tool

# Standard section names by category. These are domain-neutral resume
# conventions (TOOLING_PLAN section 36 permits freezing ergonomics like this),
# not curated professional knowledge.
STANDARD_SECTION_HEADERS: dict[str, list[str]] = {
    "summary": ["Professional Summary", "Summary", "Profile", "Executive Summary"],
    "experience": [
        "Work Experience",
        "Professional Experience",
        "Experience",
        "Employment History",
    ],
    "skills": ["Skills", "Technical Skills", "Core Competencies", "Key Skills"],
    "education": ["Education", "Academic Background", "Educational Qualifications"],
    "certifications": [
        "Certifications",
        "Professional Certifications",
        "Licenses & Certifications",
    ],
}

# A resume is incomplete without these. Summary and certifications are optional.
ESSENTIAL_SECTIONS = {"experience", "education", "skills"}

_HEADER_LINE_PATTERN = re.compile(r"^([A-Z][A-Za-z\s&]+):?\s*$")


@tool("Check Section Headers")
def check_section_headers(resume_text: str) -> str:
    """Validate that essential sections are present under ATS-recognized names.

    Args:
        resume_text: Complete resume content as text or Markdown.

    Returns:
        Validation report string grouping sections into present, missing
        essential (flagged), and missing optional (informational).
    """
    # TODO: Detect creative headers (e.g. "My Journey" instead of "Experience").
    #       Proposed: flag header lines that match no known section as warnings.
    #       Deferred: hard to separate real creative headers from job titles and
    #                 project names without false positives; needs real data.
    header_lines = _extract_header_lines(resume_text)
    present = []
    missing_essential = []
    missing_optional = []
    for section_type, aliases in STANDARD_SECTION_HEADERS.items():
        matched_header = _find_matching_header(aliases, header_lines)
        if matched_header is not None:
            present.append((section_type, matched_header))
        elif section_type in ESSENTIAL_SECTIONS:
            missing_essential.append((section_type, aliases[0]))
        else:
            missing_optional.append((section_type, aliases[0]))
    return _build_header_report(present, missing_essential, missing_optional)


def get_standard_headers() -> dict[str, list[str]]:
    """Return the ATS-recognized standard section header names by category."""
    return {section: list(headers) for section, headers in STANDARD_SECTION_HEADERS.items()}


def _extract_header_lines(resume_text: str) -> list[str]:
    """Pull header-like lines: strip leading markdown '#', keep short standalone lines."""
    headers = []
    for line in resume_text.splitlines():
        stripped = line.lstrip("#").strip()
        if _HEADER_LINE_PATTERN.match(stripped) and len(stripped.split()) <= 5:
            headers.append(stripped.rstrip(":"))
    return headers


def _find_matching_header(aliases: list[str], header_lines: list[str]) -> str | None:
    """Return the first resume header matching any alias (case-insensitive), or None."""
    for header in header_lines:
        if any(alias.lower() in header.lower() for alias in aliases):
            return header
    return None


def _build_header_report(
    present: list[tuple[str, str]],
    missing_essential: list[tuple[str, str]],
    missing_optional: list[tuple[str, str]],
) -> str:
    """Format section results. Missing essential sections set the report to NEEDS FIXES."""
    report_lines = ["Section Header Validation:", "=" * 50]
    for section_type, matched_header in present:
        report_lines.append(f"[OK] {section_type.title()}: {matched_header}")
    for section_type, recommended_name in missing_essential:
        report_lines.append(f"[!] {section_type.title()}: not found")
        report_lines.append(f"  -> Add an essential section with header: '{recommended_name}'")
    for section_type, recommended_name in missing_optional:
        report_lines.append(f"[i] {section_type.title()}: not found (optional)")
        report_lines.append(f"  -> Consider adding: '{recommended_name}'")
    report_lines.append("")
    report_lines.append(
        "Status: PASS" if not missing_essential else "Status: NEEDS FIXES"
    )
    report_lines.append("ATS recognizes section names case-insensitively.")
    return "\n".join(report_lines)
