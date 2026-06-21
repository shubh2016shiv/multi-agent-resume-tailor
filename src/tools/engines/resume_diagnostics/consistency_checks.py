"""
Consistency auditing: do experience bullets read consistently within a role?

Mechanical engine, no LLM. It flags two craft problems: opening verbs that mix
past and present tense inside one role, and the same verb opening many bullets.
Tense is detected with spaCy part-of-speech tags (a grammatical property regex
cannot read reliably across irregular verbs); repetition uses plain string match.
"""

from collections import Counter

import spacy
from spacy.language import Language

from src.data_models.resume import Experience, Resume
from src.tools.contracts import (
    Confidence,
    Location,
    ReviewComment,
    ReviewResult,
    Section,
    Severity,
)

ENGINE_ID = "consistency_auditor"
SPACY_MODEL = "en_core_web_lg"
REPEATED_VERB_THRESHOLD = 3  # a verb opening this many bullets in one role is monotonous

# Past-tense and past-participle POS tags; everything else verb-like reads as present.
PAST_TENSE_TAGS = ("VBD", "VBN")

_nlp: Language | None = None


def audit_consistency(resume: Resume) -> ReviewResult:
    """Flag tense mixing within a role and the same verb opening many bullets.

    Args:
        resume: The resume to audit; only work_experience is read.

    Returns:
        A ReviewResult. An empty comment list means bullets read consistently.
        All comments anchor to Section.EXPERIENCE.
    """
    ####################################################
    # STEP 1: DELEGATE TO THE EXPERIENCE-ONLY ENGINE SURFACE#
    ####################################################
    return audit_consistency_for_experiences(resume.work_experience)


def audit_consistency_for_experiences(experiences: list[Experience]) -> ReviewResult:
    """Flag tense mixing and repeated opening verbs across experience entries.

    Args:
        experiences: Experience entries to audit; no other resume fields are read.

    Returns:
        A ReviewResult. An empty comment list means bullets read consistently.
        All comments anchor to Section.EXPERIENCE.
    """
    ####################################################
    # STEP 1: EXIT EARLY IF THERE ARE NO ROLES TO CHECK#
    ####################################################
    if not experiences:
        return ReviewResult(comments=[], summary="No work experience to audit")

    ####################################################
    # STEP 2: CHECK TENSE CONSISTENCY AND VERB REPETITION SEPARATELY#
    ####################################################
    comments: list[ReviewComment] = []
    comments.extend(_check_tense_consistency(experiences))
    comments.extend(_check_repeated_opening_verbs(experiences))

    ####################################################
    # STEP 3: RETURN THE SUMMARY PLUS THE FINDINGS#
    ####################################################
    summary = (
        "Bullets read consistently" if not comments else f"{len(comments)} consistency issue(s)"
    )
    return ReviewResult(comments=comments, summary=summary)


def _check_tense_consistency(experiences: list[Experience]) -> list[ReviewComment]:
    """Flag a role whose opening verbs mix past and present tense."""
    ####################################################
    # STEP 1: DETECT THE OPENING-VERB TENSE USED IN EACH ROLE'S BULLETS#
    ####################################################
    findings = []
    for role in experiences:
        tenses = {_opening_verb_tense(bullet) for bullet in role.achievements}
        tenses.discard(None)

        ####################################################
        # STEP 2: FLAG ROLES THAT MIX PAST AND PRESENT TENSE#
        ####################################################
        if "past" in tenses and "present" in tenses:
            findings.append(
                _make_finding(
                    message="Bullets mix past and present tense within one role",
                    quoted_text=f"{role.job_title} at {role.company_name}",
                    severity=Severity.MINOR,
                    advice="Use one tense per role: past for a previous role, present for the current one.",
                )
            )
    return findings


def _check_repeated_opening_verbs(experiences: list[Experience]) -> list[ReviewComment]:
    """Flag a verb that opens REPEATED_VERB_THRESHOLD or more bullets in one role."""
    ####################################################
    # STEP 1: COLLECT THE FIRST WORD OF EACH BULLET PER ROLE#
    ####################################################
    findings = []
    for role in experiences:
        opening_words = []
        for bullet in role.achievements:
            words = bullet.split()
            if words:
                opening_words.append(words[0].lower())

        ####################################################
        # STEP 2: FLAG OPENING VERBS THAT REPEAT TOO OFTEN#
        ####################################################
        for verb, count in Counter(opening_words).items():
            if count >= REPEATED_VERB_THRESHOLD:
                findings.append(
                    _make_finding(
                        message=f"'{verb}' opens {count} bullets in one role",
                        quoted_text=f"{role.job_title} at {role.company_name}",
                        severity=Severity.SUGGESTION,
                        advice="Vary the opening verbs so bullets do not read repetitively.",
                    )
                )
    return findings


def _opening_verb_tense(bullet: str) -> str | None:
    """Return 'past' or 'present' for the bullet's first verb, or None if it has none.

    Uses spaCy POS tags so irregular verbs (Led, Built, Wrote) are classified
    correctly, which regex on '-ed' cannot do.
    """
    ####################################################
    # STEP 1: PARSE THE BULLET SO WE CAN FIND THE FIRST REAL VERB#
    ####################################################
    parsed_bullet = _get_nlp()(bullet)
    for token in parsed_bullet:
        ####################################################
        # STEP 2: CLASSIFY THE FIRST VERB AS PAST OR PRESENT#
        ####################################################
        if token.pos_ in ("VERB", "AUX"):
            return "past" if token.tag_ in PAST_TENSE_TAGS else "present"
    return None


def _get_nlp() -> Language:
    """Lazily load and cache the spaCy model (heavy load deferred off import)."""
    global _nlp
    ####################################################
    # STEP 1: LOAD THE SPACY MODEL ONLY ON FIRST USE#
    ####################################################
    if _nlp is None:
        _nlp = spacy.load(SPACY_MODEL)
    return _nlp


def _make_finding(message: str, quoted_text: str, severity: Severity, advice: str) -> ReviewComment:
    """Build a role-level EXPERIENCE comment for this engine (mechanical, HIGH confidence)."""
    ####################################################
    # STEP 1: WRAP THE RESULT IN THE SHARED REVIEW SHAPE#
    ####################################################
    return ReviewComment(
        engine_id=ENGINE_ID,
        message=message,
        quoted_text=quoted_text,
        location=Location(section=Section.EXPERIENCE),
        severity=severity,
        confidence=Confidence.HIGH,
        advice=advice,
    )
