"""Shared fixtures for resume quality evaluation contracts."""

import pytest

from src.data_models.evaluation import (
    ATSMetrics,
    JobAlignmentEvaluation,
    ResumeQualityReport,
    TruthfulnessEvaluation,
)


def _truthfulness_evaluation() -> TruthfulnessEvaluation:
    """Return a passing truthfulness dimension."""
    return TruthfulnessEvaluation(
        accuracy_score=90.0,
        exaggerated_claims=[],
        unsupported_skills=[],
        justification="All claims are supported.",
    )


def _job_alignment_evaluation() -> JobAlignmentEvaluation:
    """Return a conclusive passing job-alignment dimension."""
    return JobAlignmentEvaluation(
        relevance_score=85.0,
        must_have_skills_coverage=100.0,
        ats_keyword_coverage=90.0,
        missed_requirements=[],
        justification="The important requirements are addressed.",
    )


def _ats_metrics() -> ATSMetrics:
    """Return passing rendered-structure metrics."""
    return ATSMetrics(
        ats_score=88.0,
        keyword_coverage=90.0,
        formatting_issues=[],
        justification="Rendered structure passed.",
    )


@pytest.fixture
def passing_quality_report() -> ResumeQualityReport:
    """Return a report whose score should pass the default gate."""
    return ResumeQualityReport(
        overall_quality_score=85.0,
        passes_quality_gate=False,
        assessment_summary="High-quality tailored resume.",
        accuracy=_truthfulness_evaluation(),
        relevance=_job_alignment_evaluation(),
        ats_optimization=_ats_metrics(),
        feedback_for_improvement=None,
    )


@pytest.fixture
def failing_quality_report() -> ResumeQualityReport:
    """Return a report whose score should fail the default gate."""
    return ResumeQualityReport(
        overall_quality_score=72.0,
        passes_quality_gate=True,
        assessment_summary="Resume needs improvement.",
        accuracy=_truthfulness_evaluation(),
        relevance=_job_alignment_evaluation(),
        ats_optimization=_ats_metrics(),
        feedback_for_improvement="Address the missed requirements.",
    )
