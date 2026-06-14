"""Weighted blend of the three quality dimensions into one overall score.

The weights mirror the contract documented in
src/config/tasks/quality_assessment.yaml (Accuracy 40 / Relevance 35 / ATS 25).
They live here as the single code-authoritative source so the overall score is a
deterministic blend of the dimension scores, not an LLM-narrated number.
"""

ACCURACY_WEIGHT = 0.40
RELEVANCE_WEIGHT = 0.35
ATS_WEIGHT = 0.25


def blend_overall(accuracy_score: float, relevance_score: float, ats_score: float) -> float:
    """Blend the three 0-100 dimension scores into one 0-100 overall score.

    Expects each score in [0, 100]. Returns the weighted sum (Accuracy 40%,
    Relevance 35%, ATS 25%) rounded to one decimal place.
    """
    overall = (
        accuracy_score * ACCURACY_WEIGHT
        + relevance_score * RELEVANCE_WEIGHT
        + ats_score * ATS_WEIGHT
    )
    return round(overall, 1)
