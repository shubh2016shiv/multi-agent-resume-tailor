"""
ATS formatting validation: detect elements that break ATS parser ingestion.

Scope is strictly formatting and structure. Section-name validation lives in
section_header_validator.py; date and contact checks belong to diagnostics.
"""

import re
import unicodedata

from crewai.tools import tool

from src.tools.review_contract.review_models import (
    Confidence,
    Location,
    ReviewComment,
    ReviewResult,
    Section,
    Severity,
)

ENGINE_ID = "formatting_validator"

# Structural patterns that break ATS parsing. Box-drawing and other exotic
# symbols are handled by _find_problematic_characters via unicode category.
INCOMPATIBLE_PATTERNS = [
    r"\|.*\|",  # pipe-character tables
    r"\t",  # tab characters
    r"<table>",  # HTML tables
    r"<img>",  # HTML images
]

# Bullet and dash characters common in real resumes; modern ATS handle them.
SAFE_SYMBOL_CHARACTERS = {"•", "-", "*"}  # • is the bullet "."


@tool("Validate ATS Formatting")
def validate_ats_formatting(resume_text: str) -> str:
    """Check for formatting elements that break ATS parsing.

    Args:
        resume_text: Complete resume content as plain text or Markdown.

    Returns:
        Validation report string. Returns an error report for empty input.
    """
    if not resume_text.strip():
        return "ATS Formatting Validation:\n=========================\nError: empty input"
    issues = _find_formatting_issues(resume_text)
    return _build_validation_report(issues)


def audit_ats_formatting(resume_text: str) -> ReviewResult:
    """Engine surface: same checks as validate_ats_formatting, as a ReviewResult.

    Args:
        resume_text: Complete resume content as plain text or Markdown.

    Returns:
        A ReviewResult. An empty comment list means no formatting issues were
        found. Each issue becomes a document-level (Section.OTHER) comment, since
        these problems break parsing for the whole document, not one section.
    """
    if not resume_text.strip():
        return ReviewResult(comments=[], summary="Empty input: nothing to audit")
    comments = [_make_finding(issue) for issue in _find_formatting_issues(resume_text)]
    summary = (
        "No ATS formatting issues detected"
        if not comments
        else f"{len(comments)} ATS formatting issue(s)"
    )
    return ReviewResult(comments=comments, summary=summary)


def get_incompatible_patterns() -> list[str]:
    """Return the regex patterns for ATS-incompatible structural elements."""
    return list(INCOMPATIBLE_PATTERNS)


def _find_formatting_issues(resume_text: str) -> list[str]:
    """Run every formatting check and return all issue messages, flattened."""
    issues = []
    issues.extend(_find_incompatible_patterns(resume_text))
    issues.extend(_find_problematic_characters(resume_text))
    issues.extend(_find_pdf_extraction_artifacts(resume_text))
    issues.extend(_find_multi_column_layout(resume_text))
    issues.extend(_find_masked_hyperlinks(resume_text))
    return issues


def _find_incompatible_patterns(resume_text: str) -> list[str]:
    """Find pipe tables, tabs, HTML, and excessive blank lines."""
    issues = []
    for pattern in INCOMPATIBLE_PATTERNS:
        matches = re.findall(pattern, resume_text)
        if matches:
            issues.append(f"Incompatible pattern '{pattern}': {len(matches)} instance(s)")
    if re.search(r"\n\s*\n\s*\n", resume_text):
        issues.append("Multiple consecutive blank lines (should be single)")
    return issues


def _find_problematic_characters(resume_text: str) -> list[str]:
    """Find exotic symbols (emoji, arrows, stars, box-drawing) that confuse parsers.

    Uses unicode general category 'So' (symbol-other) and 'Cn' (unassigned)
    rather than a hardcoded list, so it generalizes across symbols without
    shipping a curated table. Common bullet/dash chars are excluded.
    """
    found = {
        char
        for char in resume_text
        if unicodedata.category(char) in ("So", "Cn") and char not in SAFE_SYMBOL_CHARACTERS
    }
    if not found:
        return []
    listed = ", ".join(f"'{char}'" for char in sorted(found))
    return [f"Non-standard symbol characters: {listed}"]


def _find_pdf_extraction_artifacts(resume_text: str) -> list[str]:
    """Find '(cid:N)' tokens left behind by failed PDF text extraction."""
    artifacts = re.findall(r"\(cid:\d+\)", resume_text)
    if not artifacts:
        return []
    return [
        f"PDF extraction artifacts found ({len(artifacts)}x '(cid:N)') "
        "- re-export the resume as a text-based PDF"
    ]


def _find_multi_column_layout(resume_text: str) -> list[str]:
    """Flag a likely multi-column layout from a high ratio of short lines.

    WHAT: counts non-empty lines under 30 chars and their share of all lines.
    WHY THIS APPROACH: multi-column PDFs export as many short fragmented lines;
        a simple ratio catches this without parsing geometry.
    COMPLEXITY: linear in the number of lines.
    LIMITATION: heuristic only - skills lists of short items can false-positive.
    """
    non_empty_lines = [line.strip() for line in resume_text.splitlines() if line.strip()]
    if not non_empty_lines:
        return []
    short_lines = [line for line in non_empty_lines if len(line) < 30]
    short_line_ratio = len(short_lines) / len(non_empty_lines)
    if len(short_lines) > 6 and short_line_ratio > 0.3:
        return ["Many short fragmented lines suggest a multi-column layout - use a single column"]
    return []


def _find_masked_hyperlinks(resume_text: str) -> list[str]:
    """Find Markdown links whose display text hides the real URL.

    ATS that read only display text lose the destination, so '[GitHub](http...)'
    drops the actual address. Returns one message per masked link.
    """
    # TODO: This may be noisy on resumes with many links.
    #       Proposed: only flag links to github/linkedin/portfolio domains.
    #       Deferred: need real resume data to confirm the noise level.
    issues = []
    for display_text, url in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", resume_text):
        if display_text.lower() not in url.lower():
            issues.append(f"Masked link: '{display_text}' hides URL '{url}' - show the full URL")
    return issues


def _make_finding(issue: str) -> ReviewComment:
    """Build a document-level formatting comment (mechanical, so HIGH confidence).

    Formatting issues break ATS parsing across the whole document, so they anchor
    to Section.OTHER with no excerpt; the detection message carries the specifics.
    """
    return ReviewComment(
        engine_id=ENGINE_ID,
        message=issue,
        quoted_text="",
        location=Location(section=Section.OTHER),
        severity=Severity.MAJOR,
        confidence=Confidence.HIGH,
        advice="Remove the flagged element so an ATS parser can read the resume.",
    )


def _build_validation_report(issues: list[str]) -> str:
    """Format issue messages into a readable report. Empty list means PASS."""
    if not issues:
        return (
            "ATS Formatting Validation:\n"
            "=========================\n"
            "[OK] No formatting issues detected\n"
            "[OK] Resume is ATS-compatible\n\n"
            "Status: PASS"
        )
    issue_lines = "\n".join(f"  {index + 1}. {issue}" for index, issue in enumerate(issues))
    return (
        "ATS Formatting Validation:\n"
        "=========================\n"
        f"[!] {len(issues)} formatting issue(s) detected:\n\n"
        f"{issue_lines}\n\n"
        "Status: NEEDS FIXES"
    )
