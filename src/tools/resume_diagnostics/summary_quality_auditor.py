"""
Summary quality auditing: is the professional summary the right length, person, and substance?

Hybrid engine. The mechanical half (HIGH confidence, free) checks length and
first-person pronouns. The judgment half (one LLM call, MEDIUM confidence) checks
the two qualities no mechanical proxy can read: generic boilerplate
("results-oriented professional") and a missing value proposition.

Unlike quantification_auditor, the LLM call is unconditional for any non-empty
summary: there is no digit-style mechanical gate for boilerplate. That cost
(~one bounded call per resume) is justified because a generic summary is the most
common resume failure and the first thing a recruiter reads.
"""

import string

from src.data_models.resume import Resume
from src.tools.llm_gateway import load_tool_prompt, request_review
from src.tools.review_contract.review_models import (
    Confidence,
    Location,
    ReviewComment,
    ReviewResult,
    Section,
    Severity,
)

ENGINE_ID = "summary_quality_auditor"

# Aligned with summary_writer_agent (target 75-150 words) so the generator and
# this auditor never disagree and loop forever.
# TODO: Calibrate MIN/MAX_SUMMARY_WORDS on real summaries.
#       Proposed: measure against a labelled sample, not the generator's defaults.
#       Deferred: inherited from summary_writer_agent for loop-consistency; the risk
#       is shared error (both wrong in the same direction), not disagreement.
MIN_SUMMARY_WORDS = 50
MAX_SUMMARY_WORDS = 150

FIRST_PERSON_PRONOUNS = {"i", "i'm", "i've", "my", "me", "myself", "mine"}

# TODO: Produce ReviewResult.score (0-1) to gate the Summary Writer iteration loop.
#       Proposed: weight length, first-person, generic, and value into one score.
#       Deferred: no consumer reads score yet and the weighting is uncalibrated (YAGNI).

SUMMARY_RUBRIC = load_tool_prompt("resume_diagnostics/summary_quality.md")


def audit_summary_quality(resume: Resume) -> ReviewResult:
    """Audit the professional summary for length, person (mech), and generic/value (judgment).

    Args:
        resume: The resume to audit; only professional_summary is read.

    Returns:
        A ReviewResult merging mechanical (HIGH) and judgment (MEDIUM) comments.
        An empty summary yields a single MAJOR finding and makes no LLM call.
    """
    summary_text = resume.professional_summary
    if not summary_text.strip():
        missing = _make_finding(
            message="Resume has no professional summary",
            quoted_text="(no summary provided)",
            severity=Severity.MAJOR,
            advice="Add a 2-4 sentence professional summary at the top of the resume.",
        )
        return ReviewResult(comments=[missing], summary="Missing professional summary")
    comments: list[ReviewComment] = []
    comments.extend(_check_length(summary_text))
    comments.extend(_check_first_person(summary_text))
    comments.extend(request_review(ENGINE_ID, SUMMARY_RUBRIC, summary_text).comments)
    verdict = "Summary looks strong" if not comments else f"{len(comments)} summary issue(s)"
    return ReviewResult(comments=comments, summary=verdict)


def _check_length(summary_text: str) -> list[ReviewComment]:
    """Flag a summary below the word floor (MINOR) or above the limit (MAJOR)."""
    word_count = len(summary_text.split())
    if word_count < MIN_SUMMARY_WORDS:
        return [
            _make_finding(
                message=f"Summary is {word_count} words, under the {MIN_SUMMARY_WORDS}-word floor",
                quoted_text=summary_text,
                severity=Severity.MINOR,
                advice="Expand toward 75-150 words to give recruiters enough context.",
            )
        ]
    if word_count > MAX_SUMMARY_WORDS:
        return [
            _make_finding(
                message=f"Summary is {word_count} words, over the {MAX_SUMMARY_WORDS}-word limit",
                quoted_text=summary_text,
                severity=Severity.MAJOR,
                advice="Tighten to 75-150 words; long summaries lose recruiters.",
            )
        ]
    return []


def _check_first_person(summary_text: str) -> list[ReviewComment]:
    """Flag first-person pronouns via exact token match (so 'academy' never matches 'my')."""
    tokens = {word.strip(string.punctuation).lower() for word in summary_text.split()}
    found_pronouns = tokens & FIRST_PERSON_PRONOUNS
    if not found_pronouns:
        return []
    return [
        _make_finding(
            message=f"Summary uses first-person pronouns: {', '.join(sorted(found_pronouns))}",
            quoted_text=summary_text,
            severity=Severity.MAJOR,
            advice="Rewrite without first-person pronouns; drop 'I', 'my', 'me'.",
        )
    ]


def _make_finding(
    message: str, quoted_text: str, severity: Severity, advice: str
) -> ReviewComment:
    """Build a SUMMARY-section comment for this engine's mechanical checks (HIGH confidence)."""
    return ReviewComment(
        engine_id=ENGINE_ID,
        message=message,
        quoted_text=quoted_text,
        location=Location(section=Section.SUMMARY),
        severity=severity,
        confidence=Confidence.HIGH,
        advice=advice,
    )
