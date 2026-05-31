"""
Extraction quality auditing: did the document-to-Markdown conversion succeed?

Runs right after conversion, before any agent. A bad conversion (near-empty,
fragmented, or full of font artifacts) silently poisons everything downstream,
so this gates "proceed" vs "ask the user to re-upload". Pure counts and ratios,
no model and no brittle regex.
"""

from src.tools.review_contract.review_models import (
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
    checks = [_check_text_volume, _check_fragmentation, _check_extraction_artifacts]
    comments = []
    for run_check in checks:
        comment = run_check(markdown)
        if comment is not None:
            comments.append(comment)
    blocker_count = sum(1 for comment in comments if comment.severity == Severity.BLOCKER)
    summary = (
        "Extraction quality looks usable"
        if not comments
        else f"{len(comments)} extraction issue(s), {blocker_count} blocker(s)"
    )
    return ReviewResult(comments=comments, summary=summary)


def _check_text_volume(markdown: str) -> ReviewComment | None:
    """BLOCKER when almost no text was extracted (conversion effectively failed)."""
    stripped_text = markdown.strip()
    if len(stripped_text) >= MIN_USABLE_CHARS:
        return None
    return _make_finding(
        message=f"Extraction produced only {len(stripped_text)} characters",
        quoted_text=stripped_text or "(empty)",
        severity=Severity.BLOCKER,
        advice="Conversion likely failed. Re-upload as a text-based PDF or DOCX.",
    )


def _check_fragmentation(markdown: str) -> ReviewComment | None:
    """MAJOR when a high share of tokens are single characters (garbled spacing)."""
    tokens = markdown.split()
    if len(tokens) < MIN_TOKENS_FOR_FRAGMENTATION_CHECK:
        return None
    single_char_tokens = [token for token in tokens if len(token) == 1 and token.isalpha()]
    single_char_ratio = len(single_char_tokens) / len(tokens)
    if single_char_ratio <= MAX_SINGLE_CHAR_TOKEN_RATIO:
        return None
    return _make_finding(
        message=f"{single_char_ratio:.0%} of words are single characters, suggesting fragmented text",
        quoted_text=" ".join(tokens[:EXCERPT_TOKEN_COUNT]),
        severity=Severity.MAJOR,
        advice="Text looks garbled (e.g. 'S o f t w a r e'). Try a different export or converter.",
    )


def _check_extraction_artifacts(markdown: str) -> ReviewComment | None:
    """MAJOR when '(cid:N)' font-extraction artifacts are present."""
    artifact_count = markdown.count("(cid:")
    if artifact_count == 0:
        return None
    return _make_finding(
        message=f"Found {artifact_count} '(cid:N)' font-extraction artifact(s)",
        quoted_text="(cid:N)",
        severity=Severity.MAJOR,
        advice="The PDF's fonts did not extract cleanly. Re-export as a text-based PDF.",
    )


def _make_finding(
    message: str, quoted_text: str, severity: Severity, advice: str
) -> ReviewComment:
    """Build a document-level ReviewComment for this engine (mechanical, so HIGH confidence)."""
    return ReviewComment(
        engine_id=ENGINE_ID,
        message=message,
        quoted_text=quoted_text,
        location=Location(section=Section.OTHER),
        severity=severity,
        confidence=Confidence.HIGH,
        advice=advice,
    )
