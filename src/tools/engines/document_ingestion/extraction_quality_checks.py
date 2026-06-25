"""
Extraction quality auditing: did the document-to-Markdown conversion succeed?

Runs right after conversion, before any agent. A bad conversion (near-empty,
fragmented, or full of font artifacts) silently poisons everything downstream,
so this gates "proceed" vs "ask the user to re-upload". Pure counts and ratios,
no model and no brittle regex.
"""

from src.tools.contracts import (
    Confidence,
    Location,
    ReviewComment,
    ReviewResult,
    Section,
    Severity,
)

ENGINE_ID = "extraction_quality_auditor"

# TODO: Tune these thresholds against real converted resumes.
#       Proposed: collect a sample of good and failed conversions and calibrate.
#       Deferred: defaults below are reasonable starting points, not measured.
MIN_USABLE_CHARS = 200
MIN_TOKENS_FOR_FRAGMENTATION_CHECK = 50
MAX_SINGLE_CHAR_TOKEN_RATIO = 0.30
EXCERPT_TOKEN_COUNT = 20  # how many tokens to quote as a sample in a finding


def audit_extraction_quality(markdown: str) -> ReviewResult:
    """Report signs that conversion produced unusable text.

    Args:
        markdown: The converted document text to audit.

    Returns:
        A ReviewResult. A comment with severity BLOCKER means the orchestrator
        should stop and ask the user to re-upload; an empty result means the
        extraction looks usable.
    """
    ####################################################
    # STEP 1: RUN A SMALL SET OF SIMPLE QUALITY CHECKS#
    ####################################################
    # We do not try to "understand" the document here.
    # We only look for a few failure patterns that usually mean
    # the document conversion went wrong before any agent sees it.
    checks = [_check_text_volume, _check_fragmentation, _check_extraction_artifacts]
    comments = []

    ####################################################
    # STEP 2: COLLECT ONLY THE CHECKS THAT FOUND A REAL PROBLEM#
    ####################################################
    # Each check returns either:
    # - None when the text looks fine for that concern
    # - one ReviewComment when it sees a meaningful issue
    for run_check in checks:
        comment = run_check(markdown)
        if comment is not None:
            comments.append(comment)

    ####################################################
    # STEP 3: COUNT HOW SEVERE THE PROBLEMS ARE#
    ####################################################
    # Blockers mean the extraction is likely too broken to continue safely.
    blocker_count = sum(1 for comment in comments if comment.severity == Severity.BLOCKER)

    ####################################################
    # STEP 4: RETURN A SHORT SUMMARY PLUS THE DETAILED FINDINGS#
    ####################################################
    # The summary gives the orchestrator a quick read.
    # The comments carry the actual evidence and user-facing advice.
    summary = (
        "Extraction quality looks usable"
        if not comments
        else f"{len(comments)} extraction issue(s), {blocker_count} blocker(s)"
    )
    return ReviewResult(comments=comments, summary=summary)


def _check_text_volume(markdown: str) -> ReviewComment | None:
    """BLOCKER when almost no text was extracted (conversion effectively failed)."""
    ####################################################
    # STEP 1: REMOVE OUTER WHITESPACE BEFORE COUNTING#
    ####################################################
    # A file with lots of blank space but almost no real text
    # should still count as a failed extraction.
    stripped_text = markdown.strip()

    ####################################################
    # STEP 2: TREAT VERY SHORT OUTPUT AS A CONVERSION FAILURE#
    ####################################################
    # If the converter pulled only a tiny amount of text,
    # the downstream agents will have almost nothing useful to work with.
    if len(stripped_text) >= MIN_USABLE_CHARS:
        return None

    ####################################################
    # STEP 3: RETURN A BLOCKER WITH THE EXTRACTED SAMPLE#
    ####################################################
    # Showing the small extracted sample makes the failure easier to inspect.
    return _make_finding(
        message=f"Extraction produced only {len(stripped_text)} characters",
        quoted_text=stripped_text or "(empty)",
        severity=Severity.BLOCKER,
        advice="Conversion likely failed. Re-upload as a text-based PDF or DOCX.",
    )


def _check_fragmentation(markdown: str) -> ReviewComment | None:
    """MAJOR when a high share of tokens are single characters (garbled spacing)."""
    ####################################################
    # STEP 1: SPLIT THE TEXT INTO WORD-LIKE TOKENS#
    ####################################################
    # We use simple whitespace splitting because we only need
    # a rough signal, not perfect linguistic parsing.
    tokens = markdown.split()

    ####################################################
    # STEP 2: SKIP THIS CHECK WHEN THERE IS TOO LITTLE TEXT#
    ####################################################
    # Very short documents can make the ratio noisy and misleading.
    if len(tokens) < MIN_TOKENS_FOR_FRAGMENTATION_CHECK:
        return None

    ####################################################
    # STEP 3: COUNT HOW MANY TOKENS LOOK LIKE BROKEN LETTER FRAGMENTS#
    ####################################################
    # Example of bad extraction:
    # "S o f t w a r e E n g i n e e r"
    # A high share of one-letter alphabetic tokens often means the PDF
    # was extracted with broken spacing or layout reconstruction.
    single_char_tokens = [token for token in tokens if len(token) == 1 and token.isalpha()]
    single_char_ratio = len(single_char_tokens) / len(tokens)

    ####################################################
    # STEP 4: ONLY FLAG THE TEXT WHEN THE FRAGMENTATION IS HIGH ENOUGH#
    ####################################################
    # Some single-letter words are normal. We only flag this when the ratio
    # is high enough to suggest the whole extraction is degraded.
    if single_char_ratio <= MAX_SINGLE_CHAR_TOKEN_RATIO:
        return None

    ####################################################
    # STEP 5: RETURN A MAJOR FINDING WITH A SHORT TEXT EXCERPT#
    ####################################################
    # The excerpt helps the reader quickly verify whether the text is garbled.
    return _make_finding(
        message=f"{single_char_ratio:.0%} of words are single characters, suggesting fragmented text",
        quoted_text=" ".join(tokens[:EXCERPT_TOKEN_COUNT]),
        severity=Severity.MAJOR,
        advice="Text looks garbled (e.g. 'S o f t w a r e'). Try a different export or converter.",
    )


def _check_extraction_artifacts(markdown: str) -> ReviewComment | None:
    """MAJOR when '(cid:N)' font-extraction artifacts are present."""
    ####################################################
    # STEP 1: LOOK FOR A KNOWN BAD EXTRACTION PATTERN#
    ####################################################
    # "(cid:N)" usually appears when the PDF font encoding did not
    # decode cleanly during text extraction.
    artifact_count = markdown.count("(cid:")

    ####################################################
    # STEP 2: IGNORE THE CHECK IF THE ARTIFACT NEVER APPEARS#
    ####################################################
    if artifact_count == 0:
        return None

    ####################################################
    # STEP 3: REPORT THE ARTIFACT COUNT AS A MAJOR ISSUE#
    ####################################################
    # Even a few of these markers can make important text unreadable.
    return _make_finding(
        message=f"Found {artifact_count} '(cid:N)' font-extraction artifact(s)",
        quoted_text="(cid:N)",
        severity=Severity.MAJOR,
        advice="The PDF's fonts did not extract cleanly. Re-export as a text-based PDF.",
    )


def _make_finding(message: str, quoted_text: str, severity: Severity, advice: str) -> ReviewComment:
    """Build a document-level ReviewComment for this engine (mechanical, so HIGH confidence)."""
    ####################################################
    # STEP 1: BUILD A STANDARD REVIEW COMMENT SHAPE#
    ####################################################
    # Every check returns findings in the same structure so the rest
    # of the system can render and route them consistently.
    return ReviewComment(
        engine_id=ENGINE_ID,
        message=message,
        quoted_text=quoted_text,
        location=Location(section=Section.OTHER),
        severity=severity,
        confidence=Confidence.HIGH,
        advice=advice,
    )
