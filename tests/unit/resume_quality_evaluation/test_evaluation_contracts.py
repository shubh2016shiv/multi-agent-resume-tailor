"""Contract tests for deterministic resume quality evaluation.

Skill/requirement matching now runs on embedding similarity (skill_similarity_match).
To keep these tests deterministic and offline, the matcher is patched at the boundary
where each evaluator imported it -- the tests verify scoring/weighting logic, not the
embedding model. The ATS-keyword and rendered-structure paths stay literal/mechanical
and need no patch.
"""

from collections.abc import Callable
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

_TRUTHFULNESS_MATCHER = "src.resume_quality_evaluation.truthfulness.is_required_skill_evidenced"
_JOB_ALIGNMENT_MATCHER = "src.resume_quality_evaluation.job_alignment.is_required_skill_evidenced"


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


def _matcher_evidencing(*evidenced_terms: str) -> Callable[[str, list[str]], bool]:
    """Return a stub matcher that evidences only the given (case-insensitive) terms.

    Replaces the embedding call so tests are deterministic; the term passed in is the
    skill/requirement's match term, which these tests control via skill_name/requirement.
    """
    evidenced = {term.casefold() for term in evidenced_terms}

    def _stub(required_skill: str, _candidate_skills: list[str]) -> bool:
        return required_skill.casefold() in evidenced

    return _stub


# --- truthfulness (skill support via similarity, matcher patched) -----------------


def test_identical_resume_has_perfect_accuracy() -> None:
    """An identical resume has every skill evidenced and scores perfectly."""
    resume = _resume()

    with patch(_TRUTHFULNESS_MATCHER, _matcher_evidencing("Python")):
        result = evaluate_resume_truthfulness(resume, resume)

    assert result.accuracy_score == 100.0
    assert result.exaggerated_claims == []
    assert result.unsupported_skills == []


def test_introduced_number_reduces_accuracy_by_fifteen_points() -> None:
    """One introduced number costs one fixed accuracy penalty (skill check passes)."""
    original = _resume(summary="Built Python APIs.")
    revised = _resume(summary="Built 12 Python APIs.")

    with patch(_TRUTHFULNESS_MATCHER, _matcher_evidencing("Python")):
        result = evaluate_resume_truthfulness(original, revised)

    assert result.accuracy_score == 85.0
    assert len(result.exaggerated_claims) == 1


def test_new_skill_not_evidenced_is_unsupported() -> None:
    """A revised skill with no similar original skill is reported unsupported."""
    original = _resume(skills=("Python",))
    revised = _resume(skills=("Python", "Terraform"))

    with patch(_TRUTHFULNESS_MATCHER, _matcher_evidencing("Python")):
        result = evaluate_resume_truthfulness(original, revised)

    assert result.accuracy_score == 85.0
    assert result.unsupported_skills == ["Terraform"]


def test_reworded_supported_skill_is_not_unsupported() -> None:
    """A reworded skill the matcher treats as similar is not flagged as fabricated."""
    original = _resume(summary="Built RAG pipelines.", skills=("RAG",))
    revised = _resume(summary="Built RAG pipelines.", skills=("RAG systems",))

    with patch(_TRUTHFULNESS_MATCHER, _matcher_evidencing("RAG systems")):
        result = evaluate_resume_truthfulness(original, revised)

    assert result.accuracy_score == 100.0
    assert result.unsupported_skills == []


# --- job alignment: ATS-keyword fallback path (literal, no matcher) ----------------


def test_relevance_reports_complete_and_partial_keyword_coverage() -> None:
    """With no structured requirements, relevance is literal ATS keyword coverage."""
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


# --- job alignment: structured requirements via similarity (matcher patched) -------


def test_structured_requirements_use_importance_weighting() -> None:
    """Must-have requirements contribute more weight than optional requirements."""
    job = _job(("Python", "AWS"))
    job.requirements = [
        JobRequirement(requirement="Python", importance=SkillImportance.MUST_HAVE),
        JobRequirement(requirement="AWS", importance=SkillImportance.SHOULD_HAVE),
        JobRequirement(requirement="Terraform", importance=SkillImportance.NICE_TO_HAVE),
    ]

    with patch(_JOB_ALIGNMENT_MATCHER, _matcher_evidencing("Python", "AWS")):
        result = evaluate_job_alignment(_resume(summary="Built Python services on AWS."), job)

    # matched weight (3 must + 2 should) / total (3 + 2 + 1) = 5/6 = 83.3
    assert result.relevance_score == 83.3
    assert result.must_have_skills_coverage == 100.0
    assert result.ats_keyword_coverage == 100.0
    assert result.missed_requirements == ["Terraform"]


def test_unmatched_requirement_is_a_gap_and_stays_conclusive() -> None:
    """An unmatched requirement is reported missed; the result stays conclusive (no abstain)."""
    job = _job(())
    job.requirements = [
        JobRequirement(requirement="Python", importance=SkillImportance.MUST_HAVE),
        JobRequirement(requirement="Leadership", importance=SkillImportance.NICE_TO_HAVE),
    ]

    with patch(_JOB_ALIGNMENT_MATCHER, _matcher_evidencing("Python")):
        result = evaluate_job_alignment(_resume(skills=("Python",)), job)

    # matched weight 3 / total (3 + 1) = 75.0
    assert result.relevance_score == 75.0
    assert result.must_have_skills_coverage == 100.0
    assert result.missed_requirements == ["Leadership"]
    assert result.is_conclusive is True


# --- rendered structure (mechanical, unchanged) ------------------------------------


def test_rendered_header_check_passes_and_fails_mechanically() -> None:
    """Rendered ATS grading depends on essential LaTeX section headers."""
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
    """Rendered ATS grading never guesses PASS after a render failure."""
    with patch(
        "src.resume_quality_evaluation.rendered_structure.build_resume_tex",
        side_effect=RuntimeError("template failed"),
    ):
        result = evaluate_rendered_structure(_resume())

    assert result.status is AtsCheckStatus.INCONCLUSIVE
    assert result.ats_score == 0.0


# --- overall score + grounding (evaluators fully mocked, no live calls) ------------


def test_weighted_score_uses_current_product_policy() -> None:
    """Overall score uses the documented 40/35/25 weighting."""
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
        report, _ = _ground_quality_dimensions(_quality_feedback(), _resume(), _resume(), _job(()))

    assert report.overall_quality_score == 89.0
    assert report.accuracy == accuracy
    assert report.relevance == relevance
    assert report.assessment_summary == "Agent-authored narrative survives."
    assert report.feedback_for_improvement == "Agent-authored feedback survives."


def test_non_pass_rendered_status_hard_blocks_quality_gate() -> None:
    """A non-PASS rendered status blocks release even when the blend would pass."""
    perfect_accuracy = TruthfulnessEvaluation(
        accuracy_score=100.0, exaggerated_claims=[], unsupported_skills=[], justification="ok"
    )
    perfect_relevance = JobAlignmentEvaluation(
        relevance_score=100.0, must_have_skills_coverage=100.0, missed_requirements=[], justification="ok"
    )
    inconclusive_ats = RenderedStructureEvaluation(
        status=AtsCheckStatus.INCONCLUSIVE, violations=[], ats_score=100.0, detail="Could not verify."
    )
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
            return_value=inconclusive_ats,
        ),
    ):
        report, _ = _ground_quality_dimensions(_quality_feedback(), _resume(), _resume(), _job(()))

    assert report.overall_quality_score == 100.0
    assert report.passes_quality_gate is False


def test_inconclusive_job_alignment_hard_blocks_quality_gate() -> None:
    """Missing job signals block release even when other dimensions pass."""
    accuracy = TruthfulnessEvaluation(
        accuracy_score=100.0, exaggerated_claims=[], unsupported_skills=[], justification="ok"
    )
    passing_ats = RenderedStructureEvaluation(
        status=AtsCheckStatus.PASS, violations=[], ats_score=100.0, detail="Rendered structure passed."
    )
    with (
        patch(
            "src.orchestration.nodes.resume_quality.evaluate_resume_truthfulness",
            return_value=accuracy,
        ),
        patch(
            "src.orchestration.nodes.resume_quality.evaluate_rendered_structure",
            return_value=passing_ats,
        ),
    ):
        # evaluate_job_alignment runs for real on a job with no requirements/keywords ->
        # the empty (inconclusive) path, which makes no embedding call.
        report, _ = _ground_quality_dimensions(_quality_feedback(), _resume(), _resume(), _job(()))

    assert report.relevance.is_conclusive is False
    assert report.passes_quality_gate is False
