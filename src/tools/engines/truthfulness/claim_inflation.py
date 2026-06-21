"""
Claim-inflation detection: did the rewrite introduce a NUMBER the original never had?

Fully deterministic -- no spaCy, no LLM. Same input always yields the same result,
which is exactly what a quality gate needs. It extracts numeric facts from the original
and the revised resume with a regex, normalises them (so "5K" and "5,000" compare equal),
and flags any value present in the revision but nowhere in the original. A figure that
materialised during a rewrite is the highest-risk, most verifiable inflation there is.

WHY NO NAMED-ENTITY CHECK ANYMORE: the previous version used spaCy NER to also flag
introduced organisations/products/people. spaCy NER is trained on prose and is unreliable
on resume text (skills lists and bullet fragments lack the sentence context it needs), so
it mislabelled common tech terms ("Python", "AWS", "PyTorch") as introduced ORGs even when
they were present in the original -- false positives that tanked the accuracy score. Entity
fabrication (an invented company, role, or degree) is a SEMANTIC judgment and is owned by
the complementary rewrite_drift_detector (LLM, confidence-gated). Introduced SKILLS are
owned by accuracy_rubric.grade_accuracy via keyword_present_in_text. This engine deliberately
does ONE thing -- numbers -- so nothing it reports overlaps those two, and every comment is
HIGH confidence because the presence of a value is a measurement, not a judgment.
"""

import re

from src.data_models.resume import Resume
from src.tools.contracts import (
    Confidence,
    Location,
    ReviewComment,
    ReviewResult,
    Section,
    Severity,
)
from src.tools.engines.document_rendering.resume_text_renderer import render_resume

ENGINE_ID = "claim_inflation_detector"

# Match number-like tokens: optional '$', digits with thousands commas, an optional decimal
# (a '.' only when followed by digits, so a sentence-ending period in 'team of 50.' is not
# swallowed into the surface), optional 'K'/'M'/'B' or '%' suffix (the ones _normalize_number
# understands).
# (?<!\w) stops a digit after a letter ('Python3', 'F1') from matching as a bare number.
# (?!\w)  stops a number glued to trailing letters ('5KB') from matching.
# Only '$' is matched because _normalize_number only strips '$' -- staying ASCII and in sync.
_NUMERIC_TOKEN = re.compile(r"(?<!\w)\$?\d[\d,]*(?:\.\d+)?\s*[KkMmBb%]?(?!\w)")

# Suffix multipliers so "5K" and "5,000" compare equal after normalisation.
_MULTIPLIERS = {"k": 1_000.0, "m": 1_000_000.0, "b": 1_000_000_000.0}


def detect_claim_inflation(original: Resume, revised: Resume) -> ReviewResult:
    """Flag numbers the revision introduced that the original lacked.

    Mechanical and deterministic: no LLM, no NER, so it cannot be fooled by a model
    marking its own homework and it returns the same result on every run. Confidence is
    always HIGH -- value presence is a measurement -- while severity conveys the risk.

    Args:
        original: The source-of-truth resume, before optimization.
        revised: The rewritten resume to check for introduced figures.

    Returns:
        A ReviewResult. An empty comment list means the revision introduced no numeric
        value that is absent from the original.
    """
    ####################################################
    # STEP 1: RENDER BOTH RESUMES INTO THE SAME PLAIN-TEXT SHAPE#
    ####################################################
    # We compare rendered text so both the original and the revision
    # are inspected through one consistent surface.
    comments = _new_number_comments(render_resume(original), render_resume(revised))

    ####################################################
    # STEP 2: RETURN A SHORT SUMMARY PLUS THE DETAILED FINDINGS#
    ####################################################
    summary = "No introduced facts found" if not comments else f"{len(comments)} introduced fact(s)"
    return ReviewResult(comments=comments, summary=summary)


def _new_number_comments(original_text: str, revised_text: str) -> list[ReviewComment]:
    """Flag normalised numeric values present in the revision but not anywhere in the original."""
    ####################################################
    # STEP 1: EXTRACT THE NORMALIZED NUMERIC VALUES FROM THE ORIGINAL#
    ####################################################
    # We keep only the values themselves here because the core question is:
    # did this number exist anywhere in the source-of-truth resume?
    original_values = set(_extract_numbers(original_text))

    ####################################################
    # STEP 2: CHECK EVERY NUMBER IN THE REVISION AGAINST THE ORIGINAL SET#
    ####################################################
    findings = []
    for value, surface in _extract_numbers(revised_text).items():
        ####################################################
        # STEP 3: FLAG NUMBERS THAT APPEAR ONLY IN THE REVISION#
        ####################################################
        if value not in original_values:
            findings.append(
                _make_finding(
                    message=f"Revision introduces the figure '{surface}', absent from the original",
                    quoted_text=surface,
                    advice="Confirm this figure is real and came from the candidate; if it cannot be sourced, remove it.",
                )
            )
    return findings


def _extract_numbers(text: str) -> dict[float, str]:
    """Map each normalised numeric value in the text to its first surface form.

    Regex-based and deterministic. Tokens that do not normalise to a float (e.g. a stray
    't' suffix) are skipped rather than mis-flagged -- a miss is safer than a false flag.
    """
    ####################################################
    # STEP 1: SCAN THE TEXT FOR NUMBER-LIKE SURFACES#
    ####################################################
    numbers: dict[float, str] = {}
    for match in _NUMERIC_TOKEN.finditer(text):
        surface = match.group().strip()

        ####################################################
        # STEP 2: NORMALIZE THE SURFACE SO EQUIVALENT VALUES COMPARE EQUAL#
        ####################################################
        # Example: "5K" and "5,000" should become the same canonical number.
        value = _normalize_number(surface)

        ####################################################
        # STEP 3: KEEP ONLY SURFACES THAT NORMALIZE CLEANLY#
        ####################################################
        if value is not None:
            numbers.setdefault(value, surface)
    return numbers


def _normalize_number(text: str) -> float | None:
    """Parse '5K', '$1.2M', '15%', '5,000' to a canonical float; None if not numeric.

    Word-number multipliers ('2 million') are not handled and return None, so they
    are skipped rather than mis-flagged. TODO: handle them - Proposed: parsed word
    multipliers - Deferred because: rare, and a miss is safer than a false flag.
    """
    ####################################################
    # STEP 1: STRIP PRESENTATION CHARACTERS THAT DO NOT CHANGE THE VALUE#
    ####################################################
    # We remove commas, dollar signs, and percent signs so the remaining
    # text is easier to parse numerically.
    cleaned = text.strip().lower().replace(",", "").replace("$", "").replace("%", "").strip()

    ####################################################
    # STEP 2: APPLY ANY RECOGNIZED K/M/B MULTIPLIER#
    ####################################################
    multiplier = 1.0
    if cleaned and cleaned[-1] in _MULTIPLIERS:
        multiplier = _MULTIPLIERS[cleaned[-1]]
        cleaned = cleaned[:-1]

    ####################################################
    # STEP 3: PARSE THE BASE NUMBER AND RETURN THE CANONICAL VALUE#
    ####################################################
    try:
        return float(cleaned) * multiplier
    except ValueError:
        return None


def _make_finding(message: str, quoted_text: str, advice: str) -> ReviewComment:
    """Build a comment for an introduced figure (mechanical, so HIGH confidence).

    The diff runs over the whole rendered resume, so comments anchor to Section.OTHER.
    Severity is MAJOR (review-worthy) rather than BLOCKER so a parser artefact cannot
    hard-block the pipeline on its own.
    """
    ####################################################
    # STEP 1: WRAP THE RESULT IN THE SHARED REVIEW SHAPE#
    ####################################################
    return ReviewComment(
        engine_id=ENGINE_ID,
        message=message,
        quoted_text=quoted_text,
        location=Location(section=Section.OTHER),
        severity=Severity.MAJOR,
        confidence=Confidence.HIGH,
        advice=advice,
    )
