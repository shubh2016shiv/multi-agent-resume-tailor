"""Check whether a resume uses section names an ATS can recognize."""

import re

from src.tools.contracts import (
    Confidence,
    Location,
    ReviewComment,
    ReviewResult,
    Section,
    Severity,
)

ENGINE_ID = "section_header_validator"

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

ESSENTIAL_SECTIONS = {"experience", "education", "skills"}
HEADER_LINE_PATTERN = re.compile(r"^([A-Z][A-Za-z\s&]+):?\s*$")


def audit_section_headers(resume_text: str) -> ReviewResult:
    """Check whether the resume uses section names an ATS can recognize."""
    ####################################################
    # STEP 1: CLASSIFY WHICH SECTIONS ARE PRESENT AND WHICH ARE MISSING#
    ####################################################
    # We first translate raw resume text into a simple answer:
    # which expected sections did we recognize, and which ones did we not?
    _, missing_essential_sections, missing_optional_sections = classify_sections(resume_text)

    ####################################################
    # STEP 2: TURN MISSING ESSENTIAL SECTIONS INTO MAJOR FINDINGS#
    ####################################################
    # Essential sections matter more because ATS reviewers expect them
    # on most resumes and their absence usually hurts clarity.
    comments = [
        build_header_finding(section_type, recommended_name, Severity.MAJOR)
        for section_type, recommended_name in missing_essential_sections
    ]

    ####################################################
    # STEP 3: TURN MISSING OPTIONAL SECTIONS INTO LIGHTER SUGGESTIONS#
    ####################################################
    # Optional sections are still helpful, but they should not be treated
    # with the same severity as core resume structure.
    comments += [
        build_header_finding(section_type, recommended_name, Severity.SUGGESTION)
        for section_type, recommended_name in missing_optional_sections
    ]

    ####################################################
    # STEP 4: RETURN A SUMMARY THAT EMPHASIZES ESSENTIAL COVERAGE#
    ####################################################
    summary = (
        "All essential sections present under ATS-recognized names"
        if not missing_essential_sections
        else f"{len(missing_essential_sections)} essential section(s) missing"
    )
    return ReviewResult(comments=comments, summary=summary)


def classify_sections(
    resume_text: str,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    """Sort sections into present, missing essential, and missing optional."""
    ####################################################
    # STEP 1: EXTRACT THE LINES THAT LOOK LIKE SECTION HEADERS#
    ####################################################
    # We do not try to understand the whole document body here.
    # We only isolate the lines that could realistically be headers.
    header_lines = extract_header_lines(resume_text)
    present_sections = []
    missing_essential_sections = []
    missing_optional_sections = []

    ####################################################
    # STEP 2: CHECK EACH EXPECTED SECTION AGAINST ITS ALLOWED HEADER NAMES#
    ####################################################
    # Each logical section can appear under several common names.
    # We mark it as present if any one of those names matches.
    for section_type, aliases in STANDARD_SECTION_HEADERS.items():
        matching_header = find_matching_header(aliases, header_lines)
        if matching_header is not None:
            present_sections.append((section_type, matching_header))
        elif section_type in ESSENTIAL_SECTIONS:
            missing_essential_sections.append((section_type, aliases[0]))
        else:
            missing_optional_sections.append((section_type, aliases[0]))
    return present_sections, missing_essential_sections, missing_optional_sections


def get_standard_headers() -> dict[str, list[str]]:
    """Return the ATS-recognized standard section headers."""
    return {
        section_name: list(headers) for section_name, headers in STANDARD_SECTION_HEADERS.items()
    }


def extract_header_lines(resume_text: str) -> list[str]:
    """Pull out lines that look like section headers."""
    ####################################################
    # STEP 1: WALK LINE BY LINE THROUGH THE RESUME TEXT#
    ####################################################
    # We inspect one line at a time because section headers are usually
    # isolated on their own line in resume text.
    header_lines = []
    for line in resume_text.splitlines():
        ####################################################
        # STEP 2: CLEAN COMMON MARKDOWN HEADER MARKERS AND EXTRA SPACES#
        ####################################################
        # This lets us treat "# Skills" and "Skills:" as the same basic header idea.
        stripped_line = line.lstrip("#").strip()

        ####################################################
        # STEP 3: KEEP ONLY LINES THAT LOOK LIKE SHORT SECTION HEADERS#
        ####################################################
        # The regex filters for title-like lines, and the word-count limit prevents
        # long descriptive sentences from being mistaken for headers.
        if HEADER_LINE_PATTERN.match(stripped_line) and len(stripped_line.split()) <= 5:
            header_lines.append(stripped_line.rstrip(":"))
    return header_lines


def find_matching_header(aliases: list[str], header_lines: list[str]) -> str | None:
    """Return the first resume header matching any alias."""
    ####################################################
    # STEP 1: COMPARE EACH CANDIDATE HEADER AGAINST THE KNOWN ALIASES#
    ####################################################
    # We use case-insensitive containment here because real resumes
    # often vary slightly in wording or capitalization.
    for header_line in header_lines:
        if any(alias.lower() in header_line.lower() for alias in aliases):
            return header_line
    return None


def build_header_finding(
    section_type: str,
    recommended_name: str,
    severity: Severity,
) -> ReviewComment:
    """Build one structured finding for a missing section header."""
    ####################################################
    # STEP 1: BUILD ONE STANDARD FINDING FOR THE MISSING SECTION#
    ####################################################
    # Using the shared review contract keeps these findings easy to
    # render, prioritize, and explain elsewhere in the pipeline.
    return ReviewComment(
        engine_id=ENGINE_ID,
        message=f"No ATS-recognized '{section_type}' header found",
        quoted_text="",
        location=Location(section=Section(section_type)),
        severity=severity,
        confidence=Confidence.HIGH,
        advice=f"Add a section titled '{recommended_name}'.",
    )
