"""
Language quality auditing: are experience bullets achievements, or duties and filler?

Pure judgment engine, no mechanical half. Duty language has no mechanical proxy:
"Led" is an achievement verb, "Responsible for" is duty language, except when it
is field-standard ("Responsible for FDA audit compliance") -- and no rule can read
"except when it is not". So the detection IS the judgment, and the rubric below
bears the entire design.

This replaces the curated duty-phrase lists in the agents (the frozen-list
anti-pattern TOOLING_PLAN section 9 rejects) with model judgment that is aware of
the candidate's field. Confidence gating is how it avoids false positives on niche
domains and avoids driving an endless rewrite loop: low-confidence comments are
advisory, not blocking.
"""

from src.data_models.resume import Experience, Resume
from src.tools.llm_gateway import request_review
from src.tools.review_contract.review_models import ReviewResult

ENGINE_ID = "language_quality_auditor"

LANGUAGE_RUBRIC = """You review resume achievement bullets for two language-quality problems,
judged relative to the candidate's apparent professional field.

1. Duty language: phrasing that states a responsibility instead of an achievement,
   such as "responsible for", "worked on", "tasked with", "duties included", or
   "helped with". These should be reframed as a concrete accomplishment with an outcome.

2. Hollow phrasing: vague filler that conveys no specific contribution, such as
   "various tasks", "team player", or empty intensifiers.

Domain rule (critical for avoiding false positives):
- First infer the candidate's field from the bullets.
- If a phrase is standard, meaningful terminology in that field, do NOT flag it.
  "Responsible for FDA audit compliance" is legitimate for a compliance officer.
- When you are unsure whether phrasing is field-appropriate, set confidence to "low"
  instead of flagging it confidently.

Return one review comment per real issue, with:
- severity: "minor"
- confidence: "high" only for clear, field-independent duty language or filler;
  "medium" for likely issues; "low" when the judgment depends on domain knowledge
  you are unsure about
- message: what is weak
- quoted_text: the exact bullet text
- advice: how to reframe it as a specific achievement
- location: section "experience"

Only comment on the bullets provided. Do not invent achievements, numbers, or outcomes.
If a bullet is already a strong, specific achievement, return no comment for it.
"""


def audit_language_quality(resume: Resume) -> ReviewResult:
    """Flag duty-language and hollow phrasing in experience bullets, judged for the field.

    Args:
        resume: The resume to audit; only work_experience achievements are read.

    Returns:
        A ReviewResult of judgment comments with honest confidence (low where the
        call depends on domain knowledge). An empty result (no LLM call) means
        there were no bullets to review.
    """
    return audit_language_quality_for_experiences(resume.work_experience)


def audit_language_quality_for_experiences(experiences: list[Experience]) -> ReviewResult:
    """Flag duty-language and hollow phrasing in experience bullets.

    Args:
        experiences: Experience entries to audit; only achievements are read.

    Returns:
        A ReviewResult of judgment comments with honest confidence. An empty
        result means there were no bullets to review.
    """
    # TODO: Also review role description text, not just achievements.
    #       Proposed: send descriptions in a separate pass with a duty-tolerant rubric.
    #       Deferred: descriptions are role-summary prose and would false-positive here.
    # TODO: Domain coverage is uneven -- unfamiliar fields yield mostly low-confidence
    #       comments. Proposed: note coverage gaps to the user. Deferred: honest for now.
    bullets = _collect_bullets(experiences)
    if not bullets:
        return ReviewResult(comments=[], summary="No experience bullets to review")
    numbered_bullets = "\n".join(
        f"{index}. {bullet}" for index, bullet in enumerate(bullets, start=1)
    )
    return request_review(ENGINE_ID, LANGUAGE_RUBRIC, numbered_bullets)


def _collect_bullets(experiences: list[Experience]) -> list[str]:
    """Gather every achievement bullet across all roles."""
    bullets = []
    for role in experiences:
        bullets.extend(role.achievements)
    return bullets
