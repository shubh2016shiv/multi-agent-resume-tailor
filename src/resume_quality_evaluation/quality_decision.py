"""Calculate the product-policy weighted resume quality score.

The weights mirror the contract documented in
src/config/tasks/quality_feedback.yaml (Accuracy 40 / Relevance 35 / ATS 25).
They live here as the single code-authoritative source so the overall score is a
deterministic blend of the dimension scores, not an LLM-narrated number.
"""

from src.data_models.evaluation import ResumeQualityReport

ACCURACY_WEIGHT = 0.40
RELEVANCE_WEIGHT = 0.35
ATS_WEIGHT = 0.25
QUALITY_PASS_THRESHOLD = 80.0


def calculate_overall_quality_score(
    accuracy_score: float,
    relevance_score: float,
    ats_score: float,
) -> float:
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


def apply_resume_quality_gate(
    quality_report: ResumeQualityReport,
    threshold: float = QUALITY_PASS_THRESHOLD,
) -> ResumeQualityReport:
    """Set the release decision from the deterministic overall score.

    Expects a validated report with an overall score from 0 to 100.
    Returns a copy whose pass flag reflects whether the score meets the threshold.
    """
    passes_quality_gate = quality_report.overall_quality_score >= threshold
    return quality_report.model_copy(update={"passes_quality_gate": passes_quality_gate})


def should_render_resume(quality_report: ResumeQualityReport) -> bool:
    """Return whether a gated quality report permits final rendering.

    Expects apply_resume_quality_gate to have set the report's pass flag.
    Returns the existing pass flag without recomputing the decision.
    """
    return quality_report.passes_quality_gate
