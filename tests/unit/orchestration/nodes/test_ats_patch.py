"""Contracts for deterministic rendered-structure recovery."""

from datetime import date
from typing import cast
from unittest.mock import patch

from src.agents.ats_optimizer.models import AtsOptimizedResume
from src.agents.professional_experience.models import OptimizedExperienceSection
from src.data_models.evaluation import (
    AtsCheckStatus,
    ATSMetrics,
    JobAlignmentEvaluation,
    RenderedStructureEvaluation,
    ResumeQualityReport,
    TruthfulnessEvaluation,
)
from src.data_models.resume import Experience, OptimizedSkillsSection, Resume
from src.orchestration.nodes.ats_patch import _regrade_ats_dimension, patch_ats_assembly
from src.orchestration.state import ResumeEnhancementPipelineState


def _quality_report(*, relevance_is_conclusive: bool = True) -> ResumeQualityReport:
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
            is_conclusive=relevance_is_conclusive,
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


def _passing_structure() -> RenderedStructureEvaluation:
    return RenderedStructureEvaluation(
        status=AtsCheckStatus.PASS,
        violations=[],
        ats_score=100.0,
        detail="All essential headers are present.",
    )


def test_regrade_ats_dimension_rebuilds_score_and_gate() -> None:
    """A passing recovery preserves other dimensions and reapplies the gate."""
    regraded = _regrade_ats_dimension(_quality_report(), _passing_structure())

    assert regraded.accuracy.accuracy_score == 90.0
    assert regraded.relevance.relevance_score == 80.0
    assert regraded.ats_optimization.keyword_coverage == 75.0
    assert regraded.ats_optimization.ats_score == 100.0
    assert regraded.overall_quality_score == 89.0
    assert regraded.passes_quality_gate is True


def test_regrade_ats_dimension_keeps_gate_blocked_on_inconclusive_relevance() -> None:
    """A passing ATS re-grade must not clear a relevance-inconclusive hard block.

    Regression test: _regrade_ats_dimension only re-applied the score-threshold
    gate, so a report that was blocked on inconclusive relevance (never
    re-graded here -- see the deferred TODO on this function) came back with
    passes_quality_gate=True purely because the ATS dimension recovered.
    """
    report = _quality_report(relevance_is_conclusive=False)

    regraded = _regrade_ats_dimension(report, _passing_structure())

    assert regraded.passes_quality_gate is False


def _minimal_resume() -> Resume:
    # Every field is passed explicitly, including the ones with defaults --
    # pyright's dataclass_transform integration does not recognize
    # Field(None, ...)'s positional default the way it recognizes
    # default_factory=, and reports those params as missing otherwise.
    return Resume(
        full_name="Jane Doe",
        email="jane.doe@example.com",
        phone_number=None,
        location=None,
        website_or_portfolio=None,
        professional_summary="",
        work_experience=[],
        education=[],
        skills=[],
        certifications=[],
        languages=[],
    )


def _patch_state(*, human_review_required: bool) -> ResumeEnhancementPipelineState:
    """Return the minimum state patch_ats_assembly reads, with a FAIL to recover from."""
    return cast(
        ResumeEnhancementPipelineState,
        {
            "run_id": "test-run",
            "optimized_resume": AtsOptimizedResume(final_resume=_minimal_resume()),
            "optimized_experience": OptimizedExperienceSection(
                optimized_experiences=[
                    Experience(
                        experience_id=None,
                        job_title="Engineer",
                        company_name="Acme",
                        start_date=date(2020, 1, 1),
                        end_date=None,
                        is_current_position=False,
                        location=None,
                        description="Built backend services.",
                    )
                ],
                optimization_notes="",
            ),
            "optimized_skills": OptimizedSkillsSection(optimized_skills=[], removed_skills=[]),
            "resume": _minimal_resume(),
            "quality_report": _quality_report(relevance_is_conclusive=not human_review_required),
            "human_review_required": human_review_required,
        },
    )


def test_patch_keeps_earlier_human_review_flag_through_a_successful_recovery() -> None:
    """A PASS re-grade must not clear a human_review_required set upstream at QA.

    Regression test: patch_ats_assembly overwrote human_review_required with
    is_ats_unrecoverable(new_outcome) instead of ORing it with the flag QA had
    already set for an unrelated reason (inconclusive relevance), so a
    successful section restore silently un-escalated the run.
    """
    with patch(
        "src.orchestration.nodes.ats_patch.evaluate_rendered_structure",
        return_value=_passing_structure(),
    ):
        result = patch_ats_assembly(_patch_state(human_review_required=True))

    assert result["human_review_required"] is True


def test_patch_does_not_escalate_a_clean_successful_recovery() -> None:
    """Guard: a PASS re-grade with no prior escalation stays un-escalated."""
    with patch(
        "src.orchestration.nodes.ats_patch.evaluate_rendered_structure",
        return_value=_passing_structure(),
    ):
        result = patch_ats_assembly(_patch_state(human_review_required=False))

    assert result["human_review_required"] is False
