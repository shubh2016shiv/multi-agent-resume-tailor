"""Unit tests for src/formatters/gap_analysis_formatter.py.

Contract under test: the gap analysis formatter returns the candidate profile,
the target job, and the code-computed match report the strategist should build on.
"""

from src.formatters.gap_analysis_formatter import format_gap_analysis_context


class TestFormatGapAnalysisContext:
    """Tests for the gap analysis context builder."""

    def test_format_gap_analysis_context_includes_match_report_and_drops_full_job_text(
        self,
        sample_resume,
        sample_job_description,
        sample_match_report,
    ):
        """Contract: gap-analysis context includes match findings while omitting raw full-text job content."""
        result = format_gap_analysis_context(
            sample_resume,
            sample_job_description,
            sample_match_report,
        )

        assert result.startswith("candidate_profile:")
        assert "current_match_report:" in result
        assert "GraphQL is not evidenced in the resume." in result
        assert "Do not invent GraphQL experience." in result
        assert "full_text:" not in result
        assert "email:" not in result
