"""Unit tests for src/agents/ats_optimizer/engines.py

Contract under test: check_ats_quality(optimized, job) orchestrates three mechanical
ATS engines and returns a result dict with overall_status ('pass' / 'needs_review'),
keyword_coverage, per-engine message lists, and a list of serious (BLOCKER/MAJOR)
findings. Status is 'needs_review' iff serious findings exist.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.agents.ats_optimizer.engines import check_ats_quality
from src.agents.ats_optimizer.models import AtsOptimizedResume
from src.tools.contracts import ReviewResult, Severity
from tests.unit.agents.conftest import make_review_comment

# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def optimized_resume(minimal_resume) -> AtsOptimizedResume:
    """A minimal AtsOptimizedResume used as input to check_ats_quality."""
    return AtsOptimizedResume(final_resume=minimal_resume)


@pytest.fixture
def sample_job():
    """A minimal JobDescription stub with ats_keywords set."""
    job = MagicMock()
    job.ats_keywords = ["Python", "AWS"]
    return job


# ── helper that patches all three engine calls ────────────────────────────────


def _run_check_ats_quality(
    optimized_resume,
    sample_job,
    formatting_result,
    headers_result,
    coverage_result,
):
    """
    Run check_ats_quality with all three underlying engines replaced by controlled results.

    Mocking render_resume because it depends on the document rendering subsystem.
    Mocking audit_ats_formatting, audit_section_headers, and analyze_keyword_coverage
    because they are independent engine units with their own test suites — we are testing
    check_ats_quality's orchestration logic, not the engines themselves.
    """
    with (
        patch("src.agents.ats_optimizer.engines.render_resume", return_value="resume text"),
        patch(
            "src.agents.ats_optimizer.engines.audit_ats_formatting",
            return_value=formatting_result,
        ),
        patch(
            "src.agents.ats_optimizer.engines.audit_section_headers",
            return_value=headers_result,
        ),
        patch(
            "src.agents.ats_optimizer.engines.analyze_keyword_coverage",
            return_value=coverage_result,
        ),
    ):
        return check_ats_quality(optimized_resume, sample_job)


# ── tests ─────────────────────────────────────────────────────────────────────


class TestCheckAtsQuality:
    """Tests for check_ats_quality — orchestrates three ATS engines into one result dict."""

    def test_check_ats_quality_with_all_clean_results_returns_pass_status(
        self, optimized_resume, sample_job, clean_review_result
    ):
        """
        Contract: overall_status is 'pass' when no BLOCKER or MAJOR findings exist.
        Expected value derived from docstring: "pass iff no serious findings exist."
        """
        result = _run_check_ats_quality(
            optimized_resume,
            sample_job,
            formatting_result=clean_review_result,
            headers_result=clean_review_result,
            coverage_result=clean_review_result,
        )

        assert result["overall_status"] == "pass"
        assert result["serious_findings"] == []

    def test_check_ats_quality_with_blocker_formatting_issue_returns_needs_review(
        self, optimized_resume, sample_job, blocker_review_result, clean_review_result
    ):
        """
        Contract: overall_status is 'needs_review' when any BLOCKER finding is present.
        Expected value derived from docstring: "needs_review iff serious findings exist."
        """
        result = _run_check_ats_quality(
            optimized_resume,
            sample_job,
            formatting_result=blocker_review_result,
            headers_result=clean_review_result,
            coverage_result=clean_review_result,
        )

        assert result["overall_status"] == "needs_review"
        assert len(result["serious_findings"]) == 1
        assert "blocker" in result["serious_findings"][0].lower()

    def test_check_ats_quality_with_major_header_issue_returns_needs_review(
        self, optimized_resume, sample_job, major_review_result, clean_review_result
    ):
        """
        Contract: MAJOR findings count as serious and trigger needs_review.
        Expected value derived from docstring: "serious (BLOCKER/MAJOR) findings."
        """
        result = _run_check_ats_quality(
            optimized_resume,
            sample_job,
            formatting_result=clean_review_result,
            headers_result=major_review_result,
            coverage_result=clean_review_result,
        )

        assert result["overall_status"] == "needs_review"
        assert len(result["serious_findings"]) == 1
        assert "major" in result["serious_findings"][0].lower()

    def test_check_ats_quality_with_minor_and_suggestion_only_returns_pass(
        self, optimized_resume, sample_job, minor_only_review_result, clean_review_result
    ):
        """
        Contract: MINOR and SUGGESTION findings are NOT serious — status remains 'pass'.
        Expected value derived from docstring: serious means "BLOCKER/MAJOR" only.
        """
        result = _run_check_ats_quality(
            optimized_resume,
            sample_job,
            formatting_result=minor_only_review_result,
            headers_result=clean_review_result,
            coverage_result=clean_review_result,
        )

        assert result["overall_status"] == "pass"
        assert result["serious_findings"] == []

    def test_check_ats_quality_routes_findings_to_correct_dict_keys(
        self, optimized_resume, sample_job, clean_review_result
    ):
        """
        Contract: formatting_issues, header_issues, keyword_findings each reflect
        their respective engine's output, not a merged blob.
        """
        formatting_result = ReviewResult(
            comments=[make_review_comment(Severity.MINOR, "Spacing issue.")],
            summary="",
        )
        coverage_result = ReviewResult(
            comments=[make_review_comment(Severity.SUGGESTION, "Add more keywords.")],
            summary="",
            score=0.8,
        )

        result = _run_check_ats_quality(
            optimized_resume,
            sample_job,
            formatting_result=formatting_result,
            headers_result=clean_review_result,
            coverage_result=coverage_result,
        )

        assert len(result["formatting_issues"]) == 1
        assert "Spacing issue" in result["formatting_issues"][0]
        assert result["header_issues"] == []
        assert len(result["keyword_findings"]) == 1
        assert "Add more keywords" in result["keyword_findings"][0]

    def test_check_ats_quality_formats_findings_as_severity_colon_message(
        self, optimized_resume, sample_job, blocker_review_result, clean_review_result
    ):
        """
        Contract: each finding in serious_findings is formatted as 'SEVERITY: message'.
        Expected format derived from docstring: 'SEVERITY: message'.
        """
        result = _run_check_ats_quality(
            optimized_resume,
            sample_job,
            formatting_result=blocker_review_result,
            headers_result=clean_review_result,
            coverage_result=clean_review_result,
        )

        assert result["serious_findings"][0] == "blocker: Missing required section header."

    def test_check_ats_quality_returns_keyword_coverage_score_from_coverage_engine(
        self, optimized_resume, sample_job, clean_review_result
    ):
        """
        Contract: keyword_coverage in result comes from the coverage engine's score field.
        """
        coverage_with_score = ReviewResult(comments=[], summary="", score=0.75)

        result = _run_check_ats_quality(
            optimized_resume,
            sample_job,
            formatting_result=clean_review_result,
            headers_result=clean_review_result,
            coverage_result=coverage_with_score,
        )

        assert result["keyword_coverage"] == 0.75
