"""Evaluate whether the tailored resume stays truthful to the source resume.

Grades the accuracy dimension from mechanical evidence only -- never from an LLM's
self-assessment, which is the source of the false positives this replaces (flagging
supported facts, company names, or "Python" as exaggerations). It composes two
signals: claim-inflation detection (numbers the rewrite introduced) and a
skill-support check (each revised skill must be evidenced among the original resume's
skills, matched by embedding similarity rather than a literal/synonym table).
"""

from src.data_models.evaluation import TruthfulnessEvaluation
from src.data_models.resume import Resume, Skill
from src.resume_quality_evaluation.skill_similarity_match import (
    is_required_skill_evidenced,
    match_term_for_skill,
)
from src.tools.engines.truthfulness import detect_claim_inflation

# Each mechanically-detected fabrication (introduced fact or unsupported skill) costs
# this many accuracy points. accuracy_score is 100 minus the total penalty, floored at 0.
ACCURACY_PENALTY_PER_FINDING = 15.0


def evaluate_resume_truthfulness(original: Resume, revised: Resume) -> TruthfulnessEvaluation:
    """Grade the accuracy dimension from mechanical and similarity evidence.

    Expects the original (source-of-truth) and revised resumes.
    Returns TruthfulnessEvaluation whose exaggerated_claims are the findings,
    unsupported_skills are revised skills not evidenced among the original resume's
    skills, and accuracy_score is 100 minus ACCURACY_PENALTY_PER_FINDING per finding
    (floored at 0).
    """
    exaggerated_claims = [
        comment.message for comment in detect_claim_inflation(original, revised).comments
    ]
    unsupported_skills = [
        skill.skill_name
        for skill in revised.skills
        if not _skill_is_supported(skill, original.skills)
    ]
    finding_count = len(exaggerated_claims) + len(unsupported_skills)
    accuracy_score = max(0.0, 100.0 - ACCURACY_PENALTY_PER_FINDING * finding_count)
    return TruthfulnessEvaluation(
        accuracy_score=accuracy_score,
        exaggerated_claims=exaggerated_claims,
        unsupported_skills=unsupported_skills,
        justification=(
            f"Mechanical evidence check: {len(exaggerated_claims)} introduced fact(s) "
            f"and {len(unsupported_skills)} unsupported skill(s) vs the original resume."
        ),
    )


def _skill_is_supported(revised_skill: Skill, original_skills: list[Skill]) -> bool:
    """Return whether a revised skill is evidenced among the original resume's skills.

    Matches by embedding similarity (surface-variant tolerant), so a reworded skill
    that means the same thing still counts as supported, while a genuinely new skill
    does not.
    """
    ####################################################
    # STEP 1: BUILD THE CANDIDATE SET FROM THE ORIGINAL RESUME SKILLS#
    ####################################################
    # The original resume is the source of truth. We match against its skills'
    # canonicalized form, falling back to the raw name when extraction set none.
    candidate_skills = [match_term_for_skill(original_skill) for original_skill in original_skills]

    ####################################################
    # STEP 2: ACCEPT THE SKILL ONLY IF IT MATCHES THE ORIGINAL EVIDENCE#
    ####################################################
    # is_required_skill_evidenced owns the similarity threshold; a genuinely new skill
    # scores below it against every original skill and is reported as unsupported.
    return is_required_skill_evidenced(match_term_for_skill(revised_skill), candidate_skills)
