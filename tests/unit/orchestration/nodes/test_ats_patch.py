"""Contracts for deterministic rendered-structure recovery."""

from src.data_models.evaluation import (
    AtsCheckStatus,
    ATSMetrics,
    JobAlignmentEvaluation,
    RenderedStructureEvaluation,
    ResumeQualityReport,
    TruthfulnessEvaluation,
)
from src.orchestration.nodes.ats_patch import _regrade_ats_dimension


def _quality_report() -> ResumeQualityReport:
    """Return a report whose existing dimensions should survive ATS regrading."""
    return ResumeQualityReport(
        overall_quality_score=60.0,
        passes_quality_gate=False,
        assessment_summary="Existing narrative.",
        accuracy=TruthfulnessEvaluation(
            accuracy_score=90.0,
            exaggerated_claims=[],
            unsupported_skills=[],
            justification="Source faithful.",
        ),
        relevance=JobAlignmentEvaluation(
            relevance_score=80.0,
            must_have_skills_coverage=100.0,
            ats_keyword_coverage=75.0,
            missed_requirements=[],
            justification="Job aligned.",
        ),
        ats_optimization=ATSMetrics(
            ats_score=0.0,
            keyword_coverage=75.0,
            formatting_issues=["Missing skills header."],
            justification="Failed.",
        ),
        feedback_for_improvement=None,
    )


def test_regrade_ats_dimension_rebuilds_score_and_gate() -> None:
    """A passing recovery preserves other dimensions and reapplies the gate."""
    structure = RenderedStructureEvaluation(
        status=AtsCheckStatus.PASS,
        violations=[],
        ats_score=100.0,
        detail="All essential headers are present.",
    )

    regraded = _regrade_ats_dimension(_quality_report(), structure)

    assert regraded.accuracy.accuracy_score == 90.0
    assert regraded.relevance.relevance_score == 80.0
    assert regraded.ats_optimization.keyword_coverage == 75.0
    assert regraded.ats_optimization.ats_score == 100.0
    assert regraded.overall_quality_score == 89.0
    assert regraded.passes_quality_gate is True
