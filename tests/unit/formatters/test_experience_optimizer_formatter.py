"""Unit tests for src/formatters/experience_optimizer_formatter.py.

Contract under test: the experience formatter returns only role evidence, job
requirements, and strategy guidance needed for bullet prioritization.
"""

from src.formatters.experience_optimizer_formatter import format_experience_optimizer_context


class TestFormatExperienceOptimizerContext:
    """Tests for the experience optimizer context builder."""

    def test_format_experience_optimizer_context_keeps_role_evidence_and_drops_static_sections(
        self,
        sample_resume,
        sample_job_description,
        sample_alignment_strategy,
    ):
        """Contract: experience context includes work history and excludes unrelated static resume sections."""
        result = format_experience_optimizer_context(
            sample_resume,
            sample_job_description,
            sample_alignment_strategy,
        )

        assert result.startswith("resume_work_experience:")
        assert "achievements:" in result
        assert "experience_guidance:" in result
        assert "education:" not in result
        assert "certifications:" not in result
        assert "website_or_portfolio:" not in result
