"""Deterministic accuracy grading: is the optimized resume truthful to the original?

Grades the accuracy dimension from mechanical evidence only -- never from an LLM's
self-assessment, which is the source of the false positives this replaces (flagging
supported facts, company names, or "Python" as exaggerations). It composes two
mechanical signals: claim-inflation detection (numbers/entities the rewrite
introduced) and a skill-presence check (skills absent from the original resume text).
"""

from src.data_models.evaluation import AccuracyMetrics
from src.data_models.resume import Resume
from src.tools.engines.document_rendering.resume_text_renderer import render_resume
from src.tools.engines.job_matching import keyword_present_in_text
from src.tools.engines.truthfulness import detect_claim_inflation

# Each mechanically-detected fabrication (introduced fact or unsupported skill) costs
# this many accuracy points. accuracy_score is 100 minus the total penalty, floored at 0.
ACCURACY_PENALTY_PER_FINDING = 15.0


def grade_accuracy(original: Resume, revised: Resume) -> AccuracyMetrics:
    """Grade the accuracy dimension from mechanical evidence (no LLM).

    Expects the original (source-of-truth) and revised resumes.
    Returns AccuracyMetrics whose exaggerated_claims are the claim-inflation findings,
    unsupported_skills are revised skills absent from the original resume text, and
    accuracy_score is 100 minus ACCURACY_PENALTY_PER_FINDING per finding (floored at 0).
    """
    exaggerated_claims = [
        comment.message for comment in detect_claim_inflation(original, revised).comments
    ]
    original_text = render_resume(original)
    unsupported_skills = [
        skill.skill_name
        for skill in revised.skills
        if not keyword_present_in_text(skill.skill_name, original_text)
    ]
    finding_count = len(exaggerated_claims) + len(unsupported_skills)
    accuracy_score = max(0.0, 100.0 - ACCURACY_PENALTY_PER_FINDING * finding_count)
    return AccuracyMetrics(
        accuracy_score=accuracy_score,
        exaggerated_claims=exaggerated_claims,
        unsupported_skills=unsupported_skills,
        justification=(
            f"Mechanical evidence check: {len(exaggerated_claims)} introduced fact(s) "
            f"and {len(unsupported_skills)} unsupported skill(s) vs the original resume."
        ),
    )
