"""Shared fixtures for tests/unit/agents/.

Fixtures here are scoped to the agents test tree.
Never put test functions in conftest.
"""

import pytest

from src.agents.professional_summary.models import ProfessionalSummary, SummaryDraft
from src.data_models.evaluation import (
    ATSMetrics,
    JobAlignmentEvaluation,
    ResumeQualityReport,
    TruthfulnessEvaluation,
)
from src.data_models.resume import Resume
from src.data_models.strategy import AlignmentStrategy, SkillGap, SkillMatch
from src.tools.contracts import ReviewComment, ReviewResult, Section, Severity

# ── shared resume fixture ──────────────────────────────────────────────────────


@pytest.fixture
def minimal_resume() -> Resume:
    """A minimal valid Resume used wherever a Resume object is required."""
    return Resume(
        full_name="Jane Doe",
        email="jane@example.com",
        phone_number=None,
        location=None,
        website_or_portfolio=None,
        professional_summary="Experienced software engineer.",
        work_experience=[],
        skills=[],
        education=[],
        certifications=[],
    )


# ── alignment strategy fixtures ────────────────────────────────────────────────


@pytest.fixture
def complete_alignment_strategy() -> AlignmentStrategy:
    """A fully-populated AlignmentStrategy with no quality issues."""
    return AlignmentStrategy(
        overall_fit_score=75.0,
        summary_of_strategy="Emphasise cloud and leadership experience.",
        identified_matches=[
            SkillMatch(
                resume_skill="Python",
                job_requirement="Python development",
                match_score=95.0,
                justification="Direct skill match.",
            )
        ],
        identified_gaps=[
            SkillGap(
                missing_skill="Terraform",
                importance="should_have",
                suggestion="Reframe IaC experience to mention Terraform-compatible tooling.",
            )
        ],
        keywords_to_integrate=["Python", "AWS", "CI/CD"],
        professional_summary_guidance="Open with cloud leadership experience and quantified outcomes.",
        experience_guidance="Lead with impact metrics on the most recent role.",
        skills_guidance="Prioritise AWS services and infrastructure tooling.",
    )


# ── quality report fixtures ────────────────────────────────────────────────────


def _make_accuracy() -> TruthfulnessEvaluation:
    return TruthfulnessEvaluation(
        accuracy_score=90.0,
        exaggerated_claims=[],
        unsupported_skills=[],
        justification="All claims are well-supported.",
    )


def _make_relevance() -> JobAlignmentEvaluation:
    return JobAlignmentEvaluation(
        relevance_score=85.0,
        must_have_skills_coverage=100.0,
        missed_requirements=[],
        justification="All must-have requirements are addressed.",
    )


def _make_ats_metrics() -> ATSMetrics:
    return ATSMetrics(
        ats_score=88.0,
        keyword_coverage=90.0,
        formatting_issues=[],
        justification="Good keyword density and clean formatting.",
    )


@pytest.fixture
def passing_quality_report() -> ResumeQualityReport:
    """A ResumeQualityReport with a score above the pass threshold."""
    return ResumeQualityReport(
        overall_quality_score=85.0,
        passes_quality_gate=False,
        assessment_summary="High-quality tailored resume.",
        accuracy=_make_accuracy(),
        relevance=_make_relevance(),
        ats_optimization=_make_ats_metrics(),
        feedback_for_improvement=None,
    )


@pytest.fixture
def failing_quality_report() -> ResumeQualityReport:
    """A ResumeQualityReport with a score below the pass threshold."""
    return ResumeQualityReport(
        overall_quality_score=72.0,
        passes_quality_gate=True,
        assessment_summary="Resume needs improvement.",
        accuracy=_make_accuracy(),
        relevance=_make_relevance(),
        ats_optimization=_make_ats_metrics(),
        feedback_for_improvement="Add more quantified achievements.",
    )


# ── review result helpers ──────────────────────────────────────────────────────


def make_review_comment(severity: Severity, message: str) -> ReviewComment:
    """Build a ReviewComment with the given severity and message."""
    from src.tools.contracts import Confidence, Location

    return ReviewComment(
        engine_id="test-engine",
        message=message,
        quoted_text="",
        location=Location(section=Section.OTHER),
        severity=severity,
        confidence=Confidence.HIGH,
        advice="",
    )


@pytest.fixture
def clean_review_result() -> ReviewResult:
    """A ReviewResult with no findings — represents a fully passing check."""
    return ReviewResult(comments=[], summary="No issues found.", score=1.0)


@pytest.fixture
def blocker_review_result() -> ReviewResult:
    """A ReviewResult with one BLOCKER finding."""
    return ReviewResult(
        comments=[make_review_comment(Severity.BLOCKER, "Missing required section header.")],
        summary="Blocker found.",
        score=0.0,
    )


@pytest.fixture
def major_review_result() -> ReviewResult:
    """A ReviewResult with one MAJOR finding."""
    return ReviewResult(
        comments=[make_review_comment(Severity.MAJOR, "Keyword density too low.")],
        summary="Major issue found.",
        score=0.4,
    )


@pytest.fixture
def minor_only_review_result() -> ReviewResult:
    """A ReviewResult with only MINOR/SUGGESTION findings (not serious)."""
    return ReviewResult(
        comments=[
            make_review_comment(Severity.MINOR, "Consider adding more whitespace."),
            make_review_comment(Severity.SUGGESTION, "Use stronger action verbs."),
        ],
        summary="Minor suggestions only.",
        score=0.9,
    )


# ── professional summary fixtures ─────────────────────────────────────────────


@pytest.fixture
def good_summary_draft() -> SummaryDraft:
    """A well-formed SummaryDraft: correct length, no clichés."""
    return SummaryDraft(
        version_name="Hook-Value-Future",
        strategy_used="Classic three-part structure.",
        evidence_used="Resume: 5 yrs ML + Python/AWS pipelines -> Machine Learning, Python, AWS.",
        content=(
            "Senior Machine Learning Engineer with 5 years of experience building "
            "production Python pipelines on AWS. Led a team of 4 engineers to deliver "
            "a real-time recommendation system that increased revenue by 18%. Seeking "
            "a principal-level role to architect large-scale ML infrastructure."
        ),
        score=85,
    )


@pytest.fixture
def professional_summary_with_one_good_draft(good_summary_draft) -> ProfessionalSummary:
    """A ProfessionalSummary containing a single well-formed draft."""
    return ProfessionalSummary(
        drafts=[good_summary_draft],
        recommended_version="Hook-Value-Future",
    )
