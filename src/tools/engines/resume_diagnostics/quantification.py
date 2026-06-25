"""
Quantification auditing: which experience bullets lack a metric, and what could fix them?

Hybrid engine. The mechanical half is bulletproof and owns detection: a bullet
with no digit is a candidate. The judgment half (one bounded LLM call) owns only
the suggestion: a fitting metric category per bullet. The mechanical pass gates
what the model ever sees, so it cannot over-flag, and a fully quantified resume
costs no LLM call at all.
"""

from src.core.prompt_catalog import load_tool_prompt
from src.data_models.resume import Experience, Resume
from src.tools.contracts import ReviewResult
from src.tools.llm_gateway import request_review

ENGINE_ID = "quantification_auditor"

QUANTIFICATION_RUBRIC = load_tool_prompt("resume_diagnostics/quantification.md")


def audit_quantification(resume: Resume) -> ReviewResult:
    """Flag experience bullets with no number and suggest what metric each could add.

    Args:
        resume: The resume to audit; only work_experience is read.

    Returns:
        A ReviewResult of suggestion-level comments, one per unquantified bullet.
        An empty result (no LLM call made) means every bullet already has a number.
    """
    ####################################################
    # STEP 1: DELEGATE TO THE EXPERIENCE-ONLY ENGINE SURFACE#
    ####################################################
    return audit_quantification_for_experiences(resume.work_experience)


def audit_quantification_for_experiences(experiences: list[Experience]) -> ReviewResult:
    """Flag bullets with no number and suggest what metric each could add.

    Args:
        experiences: Experience entries to audit; only achievements are read.

    Returns:
        A ReviewResult of suggestion-level comments, one per unquantified bullet.
        An empty result (no LLM call made) means every bullet already has a number.
    """
    ####################################################
    # STEP 1: FIND THE BULLETS THAT DO NOT ALREADY CONTAIN A NUMBER#
    ####################################################
    unquantified_bullets = _find_unquantified_bullets(experiences)

    ####################################################
    # STEP 2: EXIT EARLY IF EVERY BULLET IS ALREADY QUANTIFIED#
    ####################################################
    if not unquantified_bullets:
        return ReviewResult(comments=[], summary="All experience bullets include a metric")

    ####################################################
    # STEP 3: NUMBER THE CANDIDATE BULLETS FOR THE LLM#
    ####################################################
    bullets_for_prompt = _format_bullets_for_prompt(unquantified_bullets)

    ####################################################
    # STEP 4: ASK THE REVIEW GATEWAY WHAT METRIC EACH BULLET COULD ADD#
    ####################################################
    return request_review(ENGINE_ID, QUANTIFICATION_RUBRIC, bullets_for_prompt)


def _find_unquantified_bullets(experiences: list[Experience]) -> list[str]:
    """Return achievement bullets that contain no digit (mechanical detection)."""
    ####################################################
    # STEP 1: WALK THROUGH EVERY ACHIEVEMENT BULLET#
    ####################################################
    unquantified_bullets = []
    for role in experiences:
        for bullet in role.achievements:
            ####################################################
            # STEP 2: KEEP ONLY BULLETS THAT SHOW NO NUMERIC EVIDENCE#
            ####################################################
            if not _has_number(bullet):
                unquantified_bullets.append(bullet)
    return unquantified_bullets


def _has_number(bullet: str) -> bool:
    """True if the bullet contains any digit.

    Note: counts only digits, so spelled-out numbers ('doubled', 'tripled') read
    as unquantified. See TODO below.
    """
    # TODO: Treat spelled-out magnitudes ('doubled', 'tripled', 'half') as quantified.
    #       Proposed: a small set of magnitude words checked alongside digits.
    #       Deferred: digit-only is the safe direction; add when false positives appear.
    ####################################################
    # STEP 1: TREAT ANY DIGIT AS A SIGN OF QUANTIFICATION#
    ####################################################
    return any(character.isdigit() for character in bullet)


def _format_bullets_for_prompt(bullets: list[str]) -> str:
    """Render the unquantified bullets as a numbered list for the LLM."""
    ####################################################
    # STEP 1: NUMBER THE BULLETS SO THE MODEL CAN REFERENCE THEM CLEARLY#
    ####################################################
    return "\n".join(f"{index}. {bullet}" for index, bullet in enumerate(bullets, start=1))
