"""Contract tests for deterministic resume quality evaluation."""

from unittest.mock import patch

from src.data_models.evaluation import (
    AtsCheckStatus,
    JobAlignmentEvaluation,
    QualityFeedback,
    RenderedStructureEvaluation,
    TruthfulnessEvaluation,
)
from src.data_models.job import JobDescription, JobRequirement, SkillImportance
from src.data_models.resume import Resume, Skill
from src.orchestration.nodes.resume_quality import _ground_quality_dimensions
from src.resume_quality_evaluation import (
    calculate_overall_quality_score,
    evaluate_job_alignment,
    evaluate_rendered_structure,
    evaluate_resume_truthfulness,
)


def _resume(summary: str = "Built Python APIs.", skills: tuple[str, ...] = ("Python",)) -> Resume:
    """Return a minimal valid resume for evaluation characterization."""
    return Resume(
        full_name="Jane Doe",
        email="jane@example.com",
        professional_summary=summary,
        skills=[Skill(skill_name=skill) for skill in skills],
    )


def _job(keywords: tuple[str, ...]) -> JobDescription:
    """Return a minimal target job carrying the requested ATS keywords."""
    return JobDescription(
        job_title="Backend Engineer",
        company_name="Example Co",
        summary="Build backend services.",
        full_text="Backend role.",
        ats_keywords=list(keywords),
    )


def _quality_feedback() -> QualityFeedback:
    """Return advisory prose without numeric scores or a release decision."""
    return QualityFeedback(
        assessment_summary="Agent-authored narrative survives.",
        feedback_for_improvement="Agent-authored feedback survives.",
    )


def test_identical_resume_has_perfect_accuracy() -> None:
    """Current accuracy grading gives identical resumes a perfect score."""
    resume = _resume()

    result = evaluate_resume_truthfulness(resume, resume)

    assert result.accuracy_score == 100.0
    assert result.exaggerated_claims == []
    assert result.unsupported_skills == []


def test_introduced_number_reduces_accuracy_by_fifteen_points() -> None:
    """Current accuracy grading applies one fixed penalty per introduced number."""
    original = _resume(summary="Built Python APIs.")
    revised = _resume(summary="Built 12 Python APIs.")

    result = evaluate_resume_truthfulness(original, revised)

    assert result.accuracy_score == 85.0
    assert len(result.exaggerated_claims) == 1


def test_new_skill_reduces_accuracy_by_fifteen_points() -> None:
    """Current accuracy grading treats a new unmatched skill as unsupported."""
    original = _resume(skills=("Python",))
    revised = _resume(skills=("Python", "Terraform"))

    result = evaluate_resume_truthfulness(original, revised)

    assert result.accuracy_score == 85.0
    assert result.unsupported_skills == ["Terraform"]


def test_relevance_reports_complete_and_partial_keyword_coverage() -> None:
    """Current relevance is the percentage of ATS keywords found in resume text."""
    resume = _resume(summary="Built Python APIs on AWS.")

    complete = evaluate_job_alignment(resume, _job(("Python", "AWS")))
    partial = evaluate_job_alignment(resume, _job(("Python", "Terraform")))

    assert complete.relevance_score == 100.0
    assert complete.missed_requirements == []
    assert partial.relevance_score == 50.0
    assert partial.missed_requirements == ["Terraform"]


def test_job_without_requirements_or_keywords_is_inconclusive() -> None:
    """A target job without evaluation signals fails safely instead of scoring 100."""
    result = evaluate_job_alignment(_resume(), _job(()))

    assert result.relevance_score == 0.0
    assert result.is_conclusive is False


def test_structured_requirements_use_importance_weighting() -> None:
    """Must-have requirements contribute more than optional requirements."""
    job = _job(("Python", "AWS"))
    job.requirements = [
        JobRequirement(requirement="Python", importance=SkillImportance.MUST_HAVE),
        JobRequirement(requirement="AWS", importance=SkillImportance.SHOULD_HAVE),
        JobRequirement(requirement="Terraform", importance=SkillImportance.NICE_TO_HAVE),
    ]

    result = evaluate_job_alignment(_resume(summary="Built Python services on AWS."), job)

    assert result.relevance_score == 83.3
    assert result.must_have_skills_coverage == 100.0
    assert result.ats_keyword_coverage == 100.0
    assert result.missed_requirements == ["Terraform"]


def test_rendered_header_check_passes_and_fails_mechanically() -> None:
    """Current rendered ATS grading depends on essential LaTeX section headers."""
    complete_tex = "\\section*{EXPERIENCE}\n\\section*{EDUCATION}\n\\section*{SKILLS}"
    incomplete_tex = "\\section*{EXPERIENCE}"

    with patch(
        "src.resume_quality_evaluation.rendered_structure.build_resume_tex",
        return_value=complete_tex,
    ):
        passing = evaluate_rendered_structure(_resume())
    with patch(
        "src.resume_quality_evaluation.rendered_structure.build_resume_tex",
        return_value=incomplete_tex,
    ):
        failing = evaluate_rendered_structure(_resume())

    assert passing.status is AtsCheckStatus.PASS
    assert passing.ats_score == 100.0
    assert failing.status is AtsCheckStatus.FAIL
    assert failing.ats_score == 0.0
    assert len(failing.violations) == 2


def test_render_failure_is_inconclusive_and_scores_zero() -> None:
    """Current rendered ATS grading never guesses PASS after a render failure."""
    with patch(
        "src.resume_quality_evaluation.rendered_structure.build_resume_tex",
        side_effect=RuntimeError("template failed"),
    ):
        result = evaluate_rendered_structure(_resume())

    assert result.status is AtsCheckStatus.INCONCLUSIVE
    assert result.ats_score == 0.0


def test_weighted_score_uses_current_product_policy() -> None:
    """Current overall score uses the documented 40/35/25 weighting."""
    assert calculate_overall_quality_score(100.0, 50.0, 0.0) == 57.5


def test_grounding_builds_scores_independently_and_keeps_narrative() -> None:
    """Grounding builds every numeric dimension independently of LLM prose."""
    accuracy = TruthfulnessEvaluation(
        accuracy_score=90.0,
        exaggerated_claims=[],
        unsupported_skills=[],
        justification="Grounded.",
    )
    relevance = JobAlignmentEvaluation(
        relevance_score=80.0,
        must_have_skills_coverage=75.0,
        missed_requirements=[],
        justification="Grounded.",
    )
    ats = RenderedStructureEvaluation(
        status=AtsCheckStatus.PASS,
        violations=[],
        ats_score=100.0,
        detail="Grounded.",
    )
    with (
        patch(
            "src.orchestration.nodes.resume_quality.evaluate_resume_truthfulness",
            return_value=accuracy,
        ),
        patch(
            "src.orchestration.nodes.resume_quality.evaluate_job_alignment",
            return_value=relevance,
        ),
        patch(
            "src.orchestration.nodes.resume_quality.evaluate_rendered_structure",
            return_value=ats,
        ),
    ):
        report, _ = _ground_quality_dimensions(
            _quality_feedback(), _resume(), _resume(), _job(())
        )

    assert report.overall_quality_score == 89.0
    assert report.accuracy == accuracy
    assert report.relevance == relevance
    assert report.assessment_summary == "Agent-authored narrative survives."
    assert report.feedback_for_improvement == "Agent-authored feedback survives."


def test_non_pass_rendered_status_hard_blocks_quality_gate() -> None:
    """Current rendered ATS status blocks release even when the blend would pass."""
    ats = RenderedStructureEvaluation(
        status=AtsCheckStatus.INCONCLUSIVE,
        violations=[],
        ats_score=100.0,
        detail="Could not verify.",
    )
    perfect_accuracy = evaluate_resume_truthfulness(_resume(), _resume())
    perfect_relevance = evaluate_job_alignment(_resume(), _job(("Python",)))
    with (
        patch(
            "src.orchestration.nodes.resume_quality.evaluate_resume_truthfulness",
            return_value=perfect_accuracy,
        ),
        patch(
            "src.orchestration.nodes.resume_quality.evaluate_job_alignment",
            return_value=perfect_relevance,
        ),
        patch(
            "src.orchestration.nodes.resume_quality.evaluate_rendered_structure",
            return_value=ats,
        ),
    ):
        report, _ = _ground_quality_dimensions(
            _quality_feedback(), _resume(), _resume(), _job(())
        )

    assert report.overall_quality_score == 100.0
    assert report.passes_quality_gate is False


def test_inconclusive_job_alignment_hard_blocks_quality_gate() -> None:
    """Missing job signals block release even when other dimensions pass."""
    ats = RenderedStructureEvaluation(
        status=AtsCheckStatus.PASS,
        violations=[],
        ats_score=100.0,
        detail="Rendered structure passed.",
    )
    with patch(
        "src.orchestration.nodes.resume_quality.evaluate_rendered_structure",
        return_value=ats,
    ):
        report, _ = _ground_quality_dimensions(
            _quality_feedback(),
            _resume(),
            _resume(),
            _job(()),
        )

    assert report.relevance.is_conclusive is False
    assert report.passes_quality_gate is False
