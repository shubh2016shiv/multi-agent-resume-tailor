"""ATS formatting validation: detect elements that break ATS parser ingestion."""

import re
import unicodedata

from src.tools.contracts import (
    Confidence,
    Location,
    ReviewComment,
    ReviewResult,
    Section,
    Severity,
)

ENGINE_ID = "formatting_validator"

INCOMPATIBLE_PATTERNS = [
    r"\|.*\|",
    r"\t",
    r"<table>",
    r"<img>",
]

SAFE_SYMBOL_CHARACTERS = {"•", "-", "*"}


def audit_ats_formatting(resume_text: str) -> ReviewResult:
    """Run the ATS formatting checks and return structured findings."""
    ####################################################
    # STEP 1: EXIT EARLY IF THERE IS NO TEXT TO AUDIT#
    ####################################################
    # An empty resume gives us nothing to inspect, so we return
    # a neutral result instead of manufacturing issues.
    if not resume_text.strip():
        return ReviewResult(comments=[], summary="Empty input: nothing to audit")

    ####################################################
    # STEP 2: COLLECT EVERY ATS FORMATTING ISSUE WE CAN SEE#
    ####################################################
    # Each lower-level check looks for one kind of problem:
    # incompatible structures, odd symbols, extraction artifacts,
    # layout fragmentation, or hidden links.
    issues = collect_formatting_issues(resume_text)

    ####################################################
    # STEP 3: TURN RAW ISSUE MESSAGES INTO STRUCTURED FINDINGS#
    ####################################################
    # The rest of the system expects ReviewComment objects, not plain strings.
    comments = [build_formatting_finding(issue) for issue in issues]

    ####################################################
    # STEP 4: RETURN A SHORT SUMMARY PLUS THE DETAILED COMMENTS#
    ####################################################
    summary = (
        "No ATS formatting issues detected"
        if not comments
        else f"{len(comments)} ATS formatting issue(s)"
    )
    return ReviewResult(comments=comments, summary=summary)


def get_incompatible_patterns() -> list[str]:
    """Return the regex patterns for ATS-incompatible structural elements."""
    return list(INCOMPATIBLE_PATTERNS)


def collect_formatting_issues(resume_text: str) -> list[str]:
    """Run every formatting check and return all issue messages."""
    ####################################################
    # STEP 1: RUN EACH CHECK AS AN INDEPENDENT SIGNAL#
    ####################################################
    # We keep the checks separate so each one stays easy to understand
    # and easy to tune without affecting unrelated logic.
    issues = []

    ####################################################
    # STEP 2: MERGE ALL DETECTED ISSUES INTO ONE FLAT LIST#
    ####################################################
    # A resume can fail multiple checks at once, so we accumulate
    # every message instead of stopping at the first issue.
    issues.extend(find_incompatible_patterns(resume_text))
    issues.extend(find_problematic_characters(resume_text))
    issues.extend(find_pdf_extraction_artifacts(resume_text))
    issues.extend(find_multi_column_layout(resume_text))
    issues.extend(find_masked_hyperlinks(resume_text))
    return issues


def find_incompatible_patterns(resume_text: str) -> list[str]:
    """Find pipe tables, tabs, HTML tags, and too many blank lines."""
    ####################################################
    # STEP 1: LOOK FOR STRUCTURES THAT ATS PARSERS OFTEN HANDLE BADLY#
    ####################################################
    # These are broad structural signals such as table-like pipes,
    # tab characters, or HTML fragments that often arrive from copy-paste
    # or document exports that are not ATS-friendly.
    issues = []
    for pattern in INCOMPATIBLE_PATTERNS:
        matches = re.findall(pattern, resume_text)
        if matches:
            issues.append(f"Incompatible pattern '{pattern}': {len(matches)} instance(s)")

    ####################################################
    # STEP 2: FLAG EXCESSIVE BLANK SPACING AS A LAYOUT SMELL#
    ####################################################
    # Multiple empty lines in a row can confuse parsing and usually
    # suggest visual formatting that should be simplified.
    if re.search(r"\n\s*\n\s*\n", resume_text):
        issues.append("Multiple consecutive blank lines (should be single)")
    return issues


def find_problematic_characters(resume_text: str) -> list[str]:
    """Find exotic symbols that ATS parsers often handle poorly."""
    ####################################################
    # STEP 1: COLLECT NON-STANDARD SYMBOLS THAT ARE NOT IN OUR SAFE LIST#
    ####################################################
    # We allow a few common bullet-like symbols, but many decorative
    # symbols cause parsing trouble or survive badly across conversions.
    found_characters = {
        character
        for character in resume_text
        if unicodedata.category(character) in ("So", "Cn")
        and character not in SAFE_SYMBOL_CHARACTERS
    }

    ####################################################
    # STEP 2: RETURN NOTHING IF WE DID NOT FIND ANY RISKY SYMBOLS#
    ####################################################
    if not found_characters:
        return []

    ####################################################
    # STEP 3: LIST THE SYMBOLS SO THE USER KNOWS WHAT TO REMOVE#
    ####################################################
    listed_characters = ", ".join(f"'{character}'" for character in sorted(found_characters))
    return [f"Non-standard symbol characters: {listed_characters}"]


def find_pdf_extraction_artifacts(resume_text: str) -> list[str]:
    """Find '(cid:N)' tokens left behind by bad PDF extraction."""
    ####################################################
    # STEP 1: LOOK FOR A KNOWN BROKEN PDF EXTRACTION MARKER#
    ####################################################
    # "(cid:N)" is a strong sign that the PDF text layer did not decode cleanly.
    artifacts = re.findall(r"\(cid:\d+\)", resume_text)

    ####################################################
    # STEP 2: RETURN NOTHING IF THE ARTIFACT DOES NOT APPEAR#
    ####################################################
    if not artifacts:
        return []

    ####################################################
    # STEP 3: REPORT THE ARTIFACT COUNT AND GIVE A PLAIN FIX#
    ####################################################
    return [
        f"PDF extraction artifacts found ({len(artifacts)}x '(cid:N)') "
        "- re-export the resume as a text-based PDF"
    ]


def find_multi_column_layout(resume_text: str) -> list[str]:
    """Guess whether the text came from a multi-column layout."""
    ####################################################
    # STEP 1: KEEP ONLY NON-EMPTY LINES FOR THE LAYOUT CHECK#
    ####################################################
    # Empty lines do not tell us anything about whether the resume
    # was split into narrow columns during extraction.
    non_empty_lines = [line.strip() for line in resume_text.splitlines() if line.strip()]
    if not non_empty_lines:
        return []

    ####################################################
    # STEP 2: COUNT HOW MANY LINES ARE UNUSUALLY SHORT#
    ####################################################
    # Multi-column resumes often extract into many short fragments because
    # each visual column breaks the text flow into narrow line chunks.
    short_lines = [line for line in non_empty_lines if len(line) < 30]
    short_line_ratio = len(short_lines) / len(non_empty_lines)

    ####################################################
    # STEP 3: ONLY FLAG THE LAYOUT IF THE PATTERN IS STRONG ENOUGH#
    ####################################################
    # A few short lines are normal. We only warn when there are enough of them
    # and they form a large share of the document.
    if len(short_lines) > 6 and short_line_ratio > 0.3:
        return ["Many short fragmented lines suggest a multi-column layout - use a single column"]
    return []


def find_masked_hyperlinks(resume_text: str) -> list[str]:
    """Find Markdown links whose visible label hides the real URL."""
    ####################################################
    # STEP 1: FIND LINKS WITH SEPARATE DISPLAY TEXT AND URL#
    ####################################################
    # ATS systems usually handle plain visible URLs better than links
    # where the real destination is hidden behind friendly text.
    issues = []
    for display_text, url in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", resume_text):
        ####################################################
        # STEP 2: FLAG LINKS WHERE THE LABEL DOES NOT SHOW THE REAL DESTINATION#
        ####################################################
        # Example: "[Portfolio](https://example.com)" hides the actual URL.
        if display_text.lower() not in url.lower():
            issues.append(f"Masked link: '{display_text}' hides URL '{url}' - show the full URL")
    return issues


def build_formatting_finding(issue: str) -> ReviewComment:
    """Build a structured comment for one ATS formatting issue."""
    ####################################################
    # STEP 1: WRAP THE ISSUE IN THE SHARED REVIEW SHAPE#
    ####################################################
    # This keeps every formatting issue consistent for downstream rendering,
    # scoring, and orchestration.
    return ReviewComment(
        engine_id=ENGINE_ID,
        message=issue,
        quoted_text="",
        location=Location(section=Section.OTHER),
        severity=Severity.MAJOR,
        confidence=Confidence.HIGH,
        advice="Remove the flagged element so an ATS parser can read the resume.",
    )
