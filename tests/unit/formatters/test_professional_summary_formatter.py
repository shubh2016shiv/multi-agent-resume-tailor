"""Unit tests for src/formatters/professional_summary_formatter.py.

Contract under test: the summary formatter returns only the fields the summary
writer needs and omits unrelated resume/job data.
"""

from src.formatters.professional_summary_formatter import format_professional_summary_context


class TestFormatProfessionalSummaryContext:
    """Tests for the professional summary context builder."""

    def test_format_professional_summary_context_returns_summary_specific_toon_context(
        self,
        sample_resume,
        sample_job_description,
        sample_alignment_strategy,
    ):
        """Contract: summary context keeps narrative inputs and drops unrelated raw fields like email/full_text."""
        result = format_professional_summary_context(
            sample_resume,
            sample_job_description,
            sample_alignment_strategy,
        )

        assert result.startswith("candidate_background:")
        assert "current_professional_summary:" in result
        assert "professional_summary_guidance:" in result
        assert "Python" in result
        assert "full_text:" not in result
        assert "email:" not in result
        assert "nice_to_have" not in result
