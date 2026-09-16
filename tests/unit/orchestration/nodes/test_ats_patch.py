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
from src.orchestration.nodes.resume_quality.patch_missing_ats_sections_node import (
    patch_ats_assembly,
)
from src.orchestration.state import ResumeEnhancementPipelineState


def _quality_report(
    *,
    relevance_is_conclusive: bool = True,
    accuracy_score: float = 90.0,
    relevance_score: float = 80.0,
) -> ResumeQualityReport:
    """Return a report whose existing dimensions should survive ATS regrading."""
    return ResumeQualityReport(
        overall_quality_score=60.0,
        passes_quality_gate=False,
        assessment_summary="Existing narrative.",
        accuracy=TruthfulnessEvaluation(
            accuracy_score=accuracy_score,
            exaggerated_claims=[],
            unsupported_skills=[],
            justification="Source faithful.",
        ),
        relevance=JobAlignmentEvaluation(
            relevance_score=relevance_score,
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


def _failed_structure() -> RenderedStructureEvaluation:
    return RenderedStructureEvaluation(
        status=AtsCheckStatus.FAIL,
        violations=[],
        ats_score=40.0,
        detail="An essential section is still missing.",
    )


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
            "job_description": object(),
            "quality_report": _quality_report(relevance_is_conclusive=not human_review_required),
            "human_review_required": human_review_required,
        },
    )


def test_patch_recomputes_every_quality_dimension_from_recovered_resume() -> None:
    """Recovery must run the complete evaluator against the patched resume."""
    state = _patch_state(human_review_required=False)
    new_report = _quality_report(accuracy_score=70.0, relevance_score=95.0)
    with patch(
        "src.orchestration.nodes.resume_quality.patch_missing_ats_sections_node."
        "ground_quality_scores",
        return_value=(new_report, _passing_structure()),
    ) as ground_scores:
        result = patch_ats_assembly(state)

    original, revised, job = cast(tuple[Resume, Resume, object], ground_scores.call_args.args)
    assert original is state["resume"]
    assert revised.work_experience
    assert job is state["job_description"]
    quality_report = cast(ResumeQualityReport, result["quality_report"])
    assert quality_report is new_report
    assert quality_report.accuracy.accuracy_score == 70.0
    assert quality_report.relevance.relevance_score == 95.0


def test_patch_keeps_earlier_human_review_flag_through_a_successful_recovery() -> None:
    """A PASS re-grade must not clear a human_review_required set upstream at QA.

    Regression test: patch_ats_assembly overwrote human_review_required using only
    the new ATS outcome instead of ORing it with the flag QA had
    already set for an unrelated reason (inconclusive relevance), so a
    successful section restore silently un-escalated the run.
    """
    with patch(
        "src.orchestration.nodes.resume_quality.patch_missing_ats_sections_node."
        "ground_quality_scores",
        return_value=(_quality_report(), _passing_structure()),
    ):
        result = patch_ats_assembly(_patch_state(human_review_required=True))

    assert result["human_review_required"] is True


def test_patch_does_not_escalate_a_clean_successful_recovery() -> None:
    """Guard: a PASS re-grade with no prior escalation stays un-escalated."""
    with patch(
        "src.orchestration.nodes.resume_quality.patch_missing_ats_sections_node."
        "ground_quality_scores",
        return_value=(_quality_report(), _passing_structure()),
    ):
        result = patch_ats_assembly(_patch_state(human_review_required=False))

    assert result["human_review_required"] is False


def test_patch_escalates_newly_inconclusive_relevance() -> None:
    """The post-recovery relevance result must control review escalation."""
    with patch(
        "src.orchestration.nodes.resume_quality.patch_missing_ats_sections_node."
        "ground_quality_scores",
        return_value=(
            _quality_report(relevance_is_conclusive=False),
            _passing_structure(),
        ),
    ):
        result = patch_ats_assembly(_patch_state(human_review_required=False))

    assert result["human_review_required"] is True


def test_patch_escalates_when_recovery_still_fails() -> None:
    """A non-PASS result after the one recovery attempt requires human review."""
    with patch(
        "src.orchestration.nodes.resume_quality.patch_missing_ats_sections_node."
        "ground_quality_scores",
        return_value=(_quality_report(), _failed_structure()),
    ):
        result = patch_ats_assembly(_patch_state(human_review_required=False))

    assert result["human_review_required"] is True
