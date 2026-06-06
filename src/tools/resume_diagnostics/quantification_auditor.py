"""
Quantification auditing: which experience bullets lack a metric, and what could fix them?

Hybrid engine. The mechanical half is bulletproof and owns detection: a bullet
with no digit is a candidate. The judgment half (one bounded LLM call) owns only
the suggestion: a fitting metric category per bullet. The mechanical pass gates
what the model ever sees, so it cannot over-flag, and a fully quantified resume
costs no LLM call at all.
"""

from src.data_models.resume import Experience, Resume
from src.tools.llm_gateway import request_review
from src.tools.review_contract.review_models import ReviewResult

ENGINE_ID = "quantification_auditor"

QUANTIFICATION_RUBRIC = """You review resume achievement bullets that contain no numbers.
Each bullet below would be stronger with a concrete, quantified result.

For EACH numbered bullet, return one review comment with:
- severity: "suggestion"
- confidence: "medium"
- message: a short note that the bullet lacks a quantified result
- quoted_text: the exact bullet text
- advice: suggest ONE fitting metric category to add, such as team size, time saved,
  scale (users, requests, data volume), cost, revenue, or adoption. Name the category
  with a brief example. Do NOT invent specific numbers.
- location: section "experience"

Only comment on the bullets provided. Do not invent achievements or numbers.
"""


def audit_quantification(resume: Resume) -> ReviewResult:
    """Flag experience bullets with no number and suggest what metric each could add.

    Args:
        resume: The resume to audit; only work_experience is read.

    Returns:
        A ReviewResult of suggestion-level comments, one per unquantified bullet.
        An empty result (no LLM call made) means every bullet already has a number.
    """
    return audit_quantification_for_experiences(resume.work_experience)


def audit_quantification_for_experiences(experiences: list[Experience]) -> ReviewResult:
    """Flag bullets with no number and suggest what metric each could add.

    Args:
        experiences: Experience entries to audit; only achievements are read.

    Returns:
        A ReviewResult of suggestion-level comments, one per unquantified bullet.
        An empty result (no LLM call made) means every bullet already has a number.
    """
    unquantified_bullets = _find_unquantified_bullets(experiences)
    if not unquantified_bullets:
        return ReviewResult(comments=[], summary="All experience bullets include a metric")
    bullets_for_prompt = _format_bullets_for_prompt(unquantified_bullets)
    return request_review(ENGINE_ID, QUANTIFICATION_RUBRIC, bullets_for_prompt)


def _find_unquantified_bullets(experiences: list[Experience]) -> list[str]:
    """Return achievement bullets that contain no digit (mechanical detection)."""
    unquantified_bullets = []
    for role in experiences:
        for bullet in role.achievements:
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
    return any(character.isdigit() for character in bullet)


def _format_bullets_for_prompt(bullets: list[str]) -> str:
    """Render the unquantified bullets as a numbered list for the LLM."""
    return "\n".join(f"{index}. {bullet}" for index, bullet in enumerate(bullets, start=1))
