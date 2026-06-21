"""Unit tests for src/formatters/quality_assurance_formatter.py.

Contract under test: the QA formatter passes the original resume, the final
tailored resume, and the target job without leaking ATS wrapper metadata.
"""

from src.formatters.quality_assurance_formatter import format_quality_assurance_context


class TestFormatQualityAssuranceContext:
    """Tests for the quality assurance context builder."""

    def test_format_quality_assurance_context_uses_final_resume_and_drops_wrapper_metadata(
        self,
        sample_optimized_resume,
        sample_resume,
        sample_job_description,
    ):
        """Contract: QA context contains the final resume, not ATS wrapper fields like section_order."""
        result = format_quality_assurance_context(
            sample_optimized_resume,
            sample_resume,
            sample_job_description,
        )

        assert result.startswith("original_resume:")
        assert "tailored_resume:" in result
        assert "target_job:" in result
        assert "section_order:" not in result
        assert "optimization_summary:" not in result
        assert "keyword_integration_notes:" not in result
