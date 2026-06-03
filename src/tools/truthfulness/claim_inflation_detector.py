"""
Claim-inflation detection: did the rewrite introduce facts the original never had?

Mechanical engine, ZERO LLM -- and that is the whole point. issues.md flags the
original judgment design as structurally broken: a model that just rewrote a bullet
cannot reliably judge whether it overstated, because it already rationalised the
rewrite. This engine sidesteps that entirely by never asking a model. It
deterministically extracts the hard facts -- numbers and named entities -- from the
original and the revised resume, and flags any number or entity that appears in the
revised version but NOWHERE in the original. A figure or an organisation that
materialised during a rewrite is the highest-risk, most verifiable inflation there is.

It complements rewrite_drift_detector, which owns the SEMANTIC side (subjective
exaggeration, scope creep, dropped content) with a model and honest confidence
gating. This engine owns the FACTUAL side with certainty: the presence of a token
is a measurement, not a judgment, so every comment is HIGH confidence.

Scope is deliberately narrow. It does not align bullets, compare magnitudes, or
tolerance-tune -- those are calibration-heavy and false-positive prone (the proposal
that informed this engine called magnitude tuning its single biggest risk). "A new
fact not in the original" needs no calibration and is the high-precision signal;
magnitude and semantic overshoot are rewrite_drift_detector's job.
"""

import spacy
from spacy.language import Language

from src.data_models.resume import Resume
from src.tools.review_contract.review_models import (
    Confidence,
    Location,
    ReviewComment,
    ReviewResult,
    Section,
    Severity,
)

from src.tools.shared.resume_rendering import render_resume

ENGINE_ID = "claim_inflation_detector"
SPACY_MODEL = "en_core_web_lg"  # matches consistency_auditor; one spaCy model repo-wide

# spaCy labels for numeric facts (normalised, not string-matched) and for named
# entities whose appearance from nowhere signals likely fabrication.
NUMERIC_LABELS = {"CARDINAL", "MONEY", "PERCENT", "QUANTITY"}
FABRICATION_RISK_LABELS = {"ORG", "PERSON", "GPE", "PRODUCT", "FAC", "NORP"}

# Suffix multipliers so "5K" and "5,000" compare equal after normalisation.
_MULTIPLIERS = {"k": 1_000.0, "m": 1_000_000.0, "b": 1_000_000_000.0}

_nlp: Language | None = None


def detect_claim_inflation(original: Resume, revised: Resume) -> ReviewResult:
    """Flag numbers and named entities the revision introduced that the original lacked.

    Mechanical and deterministic: it makes no LLM call, so it cannot be fooled by a
    model marking its own homework. Confidence is always HIGH -- token presence is a
    measurement -- while severity conveys the fabrication risk.

    Args:
        original: The source-of-truth resume, before optimization.
        revised: The rewritten resume to check for introduced facts.

    Returns:
        A ReviewResult. An empty comment list means the revision introduced no number
        or named entity that is absent from the original.
    """
    original_doc = _get_nlp()(render_resume(original))
    revised_doc = _get_nlp()(render_resume(revised))
    comments = _new_number_comments(original_doc, revised_doc)
    comments += _new_entity_comments(original_doc, revised_doc)
    summary = (
        "No introduced facts found" if not comments else f"{len(comments)} introduced fact(s)"
    )
    return ReviewResult(comments=comments, summary=summary)


def _new_number_comments(original_doc, revised_doc) -> list[ReviewComment]:
    """Flag normalised numeric values present in the revision but not anywhere in the original."""
    original_values = set(_numbers(original_doc))
    findings = []
    for value, surface in _numbers(revised_doc).items():
        if value not in original_values:
            findings.append(
                _make_finding(
                    message=f"Revision introduces the figure '{surface}', absent from the original",
                    quoted_text=surface,
                    advice="Confirm this figure is real and came from the candidate; if it cannot be sourced, remove it.",
                )
            )
    return findings


def _new_entity_comments(original_doc, revised_doc) -> list[ReviewComment]:
    """Flag named entities (org, person, place, product) the revision introduced."""
    original_entities = {text.lower() for text, _ in _risk_entities(original_doc)}
    findings = []
    seen: set[str] = set()
    for text, label in _risk_entities(revised_doc):
        key = text.lower()
        if key not in original_entities and key not in seen:
            seen.add(key)
            findings.append(
                _make_finding(
                    message=f"Revision introduces {label} '{text}', absent from the original",
                    quoted_text=text,
                    advice="Confirm this is genuine; an employer, name, or place not in the original may be fabricated.",
                )
            )
    return findings


def _numbers(doc) -> dict[float, str]:
    """Map each normalised numeric value in the doc to its first surface form."""
    numbers: dict[float, str] = {}
    for ent in doc.ents:
        if ent.label_ in NUMERIC_LABELS:
            value = _normalize_number(ent.text)
            if value is not None:
                numbers.setdefault(value, ent.text)
    return numbers


def _risk_entities(doc) -> list[tuple[str, str]]:
    """Return (text, label) for entities whose appearance from nowhere risks fabrication."""
    return [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in FABRICATION_RISK_LABELS]


def _normalize_number(text: str) -> float | None:
    """Parse '5K', '$1.2M', '15%', '5,000' to a canonical float; None if not numeric.

    Word-number multipliers ('2 million') are not handled and return None, so they
    are skipped rather than mis-flagged. TODO: handle them — Proposed: spaCy-parsed
    word multipliers — Deferred because: rare, and a miss is safer than a false flag.
    """
    cleaned = text.strip().lower().replace(",", "").replace("$", "").replace("%", "").strip()
    multiplier = 1.0
    if cleaned and cleaned[-1] in _MULTIPLIERS:
        multiplier = _MULTIPLIERS[cleaned[-1]]
        cleaned = cleaned[:-1]
    try:
        return float(cleaned) * multiplier
    except ValueError:
        return None


def _make_finding(message: str, quoted_text: str, advice: str) -> ReviewComment:
    """Build a comment for an introduced fact (mechanical, so HIGH confidence).

    The diff runs over the whole rendered resume, so comments anchor to Section.OTHER.
    Severity is MAJOR (review-worthy) rather than BLOCKER so a parser artefact cannot
    hard-block the pipeline on its own.
    """
    return ReviewComment(
        engine_id=ENGINE_ID,
        message=message,
        quoted_text=quoted_text,
        location=Location(section=Section.OTHER),
        severity=Severity.MAJOR,
        confidence=Confidence.HIGH,
        advice=advice,
    )


def _get_nlp() -> Language:
    """Lazily load and cache the spaCy model (heavy load deferred off import)."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load(SPACY_MODEL)
    return _nlp
