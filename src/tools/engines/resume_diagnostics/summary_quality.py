"""
Summary quality auditing: is the professional summary the right length, person, and substance?

Pipeline position: node STEP 5 (see src/orchestration/nodes/summary.py). The node's
quality gate calls audit_summary_text() on the draft that will ship and blocks the
run on any MAJOR+ finding. The same engine is also the agent-facing audit_summary tool.

Hybrid engine. The mechanical half (HIGH confidence, free) checks length and
first-person pronouns. The judgment half (one LLM call, MEDIUM confidence) checks
the qualities no mechanical proxy can read: generic boilerplate, weak thesis,
brochure tone, and a missing value proposition.

Unlike quantification_auditor, the LLM call is unconditional for any non-empty
summary: there is no digit-style mechanical gate for prose quality. That cost
(~one bounded call per summary) is justified because the professional summary is
often the first recruiter read and the easiest place for generic AI tone to leak in.
"""

import string

from src.core.prompt_catalog import load_tool_prompt
from src.data_models.resume import Resume
from src.tools.contracts import (
    Confidence,
    Location,
    ReviewComment,
    ReviewResult,
    Section,
    Severity,
)
from src.tools.llm_gateway import request_review

ENGINE_ID = "summary_quality_auditor"

# Aligned with write_professional_summary_task (target 80-110 words) so the
# generator and this auditor never disagree and loop forever.
MIN_SUMMARY_WORDS = 80
MAX_SUMMARY_WORDS = 110

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
    ####################################################
    # STEP 1: READ THE SUMMARY TEXT ONCE AT THE TOP#
    ####################################################
    summary_text = resume.professional_summary
    return audit_summary_text(summary_text)


def audit_summary_text(summary_text: str) -> ReviewResult:
    """Audit one summary string for length, person, and prompt-judged prose quality.

    Args:
        summary_text: The professional summary text to review.

    Returns:
        A ReviewResult merging mechanical (HIGH) and judgment (MEDIUM) comments.
        An empty summary yields a single MAJOR finding and makes no LLM call.
    """
    ####################################################
    # STEP 1: FAIL FAST WHEN NO SUMMARY TEXT WAS PROVIDED#
    ####################################################
    if not summary_text.strip():
        missing = _make_finding(
            message="Resume has no professional summary",
            quoted_text="(no summary provided)",
            severity=Severity.MAJOR,
            advice="Add a 3-4 sentence professional summary at the top of the resume.",
        )
        return ReviewResult(comments=[missing], summary="Missing professional summary")

    ####################################################
    # STEP 2: RUN THE MECHANICAL CHECKS FIRST#
    ####################################################
    # Length and first-person detection are deterministic, so we collect
    # those before asking the model to judge substance and boilerplate.
    comments: list[ReviewComment] = []
    comments.extend(_check_length(summary_text))
    comments.extend(_check_first_person(summary_text))

    ####################################################
    # STEP 3: ADD THE JUDGMENT-BASED SUMMARY FINDINGS#
    ####################################################
    comments.extend(request_review(ENGINE_ID, SUMMARY_RUBRIC, summary_text).comments)

    ####################################################
    # STEP 4: RETURN A SHORT VERDICT PLUS ALL FINDINGS#
    ####################################################
    verdict = "Summary looks strong" if not comments else f"{len(comments)} summary issue(s)"
    return ReviewResult(comments=comments, summary=verdict)


def _check_length(summary_text: str) -> list[ReviewComment]:
    """Flag a summary outside the 80-110 word range (MAJOR on either side).

    Both bounds are hard constraints of write_professional_summary_task, and the
    pipeline's quality gate blocks on MAJOR+ only -- a MINOR floor made "too short"
    advisory, and live runs shipped 67-word summaries that dropped the candidate's
    strongest evidence. Symmetric MAJOR makes the floor as real as the ceiling.
    """
    ####################################################
    # STEP 1: COUNT THE SUMMARY WORDS#
    ####################################################
    word_count = len(summary_text.split())

    ####################################################
    # STEP 2: FLAG SUMMARIES THAT ARE TOO SHORT#
    ####################################################
    if word_count < MIN_SUMMARY_WORDS:
        return [
            _make_finding(
                message=f"Summary is {word_count} words, under the {MIN_SUMMARY_WORDS}-word floor",
                quoted_text=summary_text,
                severity=Severity.MAJOR,
                advice="Expand toward 80-110 words so the summary can land a clear thesis and evidence.",
            )
        ]

    ####################################################
    # STEP 3: FLAG SUMMARIES THAT ARE TOO LONG#
    ####################################################
    if word_count > MAX_SUMMARY_WORDS:
        return [
            _make_finding(
                message=f"Summary is {word_count} words, over the {MAX_SUMMARY_WORDS}-word limit",
                quoted_text=summary_text,
                severity=Severity.MAJOR,
                advice="Tighten to 80-110 words; long summaries lose force and scanability.",
            )
        ]
    return []


def _check_first_person(summary_text: str) -> list[ReviewComment]:
    """Flag first-person pronouns via exact token match (so 'academy' never matches 'my')."""
    ####################################################
    # STEP 1: NORMALIZE THE SUMMARY INTO CLEAN LOWERCASE TOKENS#
    ####################################################
    tokens = {word.strip(string.punctuation).lower() for word in summary_text.split()}

    ####################################################
    # STEP 2: LOOK FOR FIRST-PERSON PRONOUNS IN THAT TOKEN SET#
    ####################################################
    found_pronouns = tokens & FIRST_PERSON_PRONOUNS
    if not found_pronouns:
        return []

    ####################################################
    # STEP 3: RETURN ONE FINDING LISTING THE PRONOUNS WE FOUND#
    ####################################################
    return [
        _make_finding(
            message=f"Summary uses first-person pronouns: {', '.join(sorted(found_pronouns))}",
            quoted_text=summary_text,
            severity=Severity.MAJOR,
            advice="Rewrite without first-person pronouns; drop 'I', 'my', 'me'.",
        )
    ]


def _make_finding(message: str, quoted_text: str, severity: Severity, advice: str) -> ReviewComment:
    """Build a SUMMARY-section comment for this engine's mechanical checks (HIGH confidence)."""
    ####################################################
    # STEP 1: WRAP THE RESULT IN THE SHARED REVIEW SHAPE#
    ####################################################
    return ReviewComment(
        engine_id=ENGINE_ID,
        message=message,
        quoted_text=quoted_text,
        location=Location(section=Section.SUMMARY),
        severity=severity,
        confidence=Confidence.HIGH,
        advice=advice,
    )
