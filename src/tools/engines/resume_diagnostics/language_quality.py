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

from src.core.prompt_catalog import load_tool_prompt
from src.data_models.resume import Experience, Resume
from src.tools.contracts import ReviewResult
from src.tools.llm_gateway import request_review

ENGINE_ID = "language_quality_auditor"

LANGUAGE_RUBRIC = load_tool_prompt("resume_diagnostics/language_quality.md")


def audit_language_quality(resume: Resume) -> ReviewResult:
    """Flag duty-language and hollow phrasing in experience bullets, judged for the field.

    Args:
        resume: The resume to audit; only work_experience achievements are read.

    Returns:
        A ReviewResult of judgment comments with honest confidence (low where the
        call depends on domain knowledge). An empty result (no LLM call) means
        there were no bullets to review.
    """
    ####################################################
    # STEP 1: DELEGATE TO THE EXPERIENCE-ONLY ENGINE SURFACE#
    ####################################################
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
    ####################################################
    # STEP 1: GATHER ALL EXPERIENCE BULLETS INTO ONE FLAT LIST#
    ####################################################
    bullets = _collect_bullets(experiences)

    ####################################################
    # STEP 2: EXIT EARLY IF THERE ARE NO BULLETS TO JUDGE#
    ####################################################
    if not bullets:
        return ReviewResult(comments=[], summary="No experience bullets to review")

    ####################################################
    # STEP 3: NUMBER THE BULLETS SO THE MODEL CAN REFER TO THEM CLEARLY#
    ####################################################
    numbered_bullets = "\n".join(
        f"{index}. {bullet}" for index, bullet in enumerate(bullets, start=1)
    )

    ####################################################
    # STEP 4: SEND THE NUMBERED BULLETS TO THE REVIEW GATEWAY#
    ####################################################
    return request_review(ENGINE_ID, LANGUAGE_RUBRIC, numbered_bullets)


def _collect_bullets(experiences: list[Experience]) -> list[str]:
    """Gather every achievement bullet across all roles."""
    ####################################################
    # STEP 1: FLATTEN ALL ROLE ACHIEVEMENTS INTO ONE LIST#
    ####################################################
    bullets = []
    for role in experiences:
        bullets.extend(role.achievements)
    return bullets
