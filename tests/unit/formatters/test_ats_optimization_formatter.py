"""Unit tests for src/formatters/ats_optimization_formatter.py.

Contract under test: the ATS assembly formatter keeps only the chosen summary,
optimized sections, preserved static resume fields, and ATS validation inputs.
"""

from src.agents.professional_summary.models import ProfessionalSummary, SummaryDraft
from src.formatters.ats_optimization_formatter import format_ats_optimization_context


class TestFormatAtsOptimizationContext:
    """Tests for the ATS assembly context builder."""

    def test_format_ats_optimization_context_keeps_data_only_and_omits_metadata(
        self,
        sample_professional_summary,
        sample_optimized_experience,
        sample_optimized_skills,
        sample_resume,
        sample_job_description,
    ):
        """Contract: ATS context includes assembled data only, not draft or optimization metadata."""
        result = format_ats_optimization_context(
            sample_professional_summary,
            sample_optimized_experience,
            sample_optimized_skills,
            sample_resume,
            sample_job_description,
        )

        assert result.startswith("assembled_sections:")
        assert "professional_summary_text:" in result
        assert "preserved_resume_sections:" in result
        assert "ats_validation_context:" in result
        assert "optimization_notes:" not in result
        assert "drafts:" not in result
        assert "full_text:" not in result

    def test_format_ats_optimization_context_uses_first_draft_when_recommended_version_is_missing(
        self,
        sample_optimized_experience,
        sample_optimized_skills,
        sample_resume,
        sample_job_description,
    ):
        """Contract: when the recommended draft is missing, the formatter falls back to the first available draft."""
        summary_with_missing_recommendation = ProfessionalSummary(
            drafts=[
                SummaryDraft(
                    version_name="fallback",
                    strategy_used="direct",
                    evidence_used="Fallback evidence.",
                    content=(
                        "Fallback summary that should still be used when the recommended "
                        "version name does not exist."
                    ),
                    score=75,
                )
            ],
            recommended_version="missing-version",
        )

        result = format_ats_optimization_context(
            summary_with_missing_recommendation,
            sample_optimized_experience,
            sample_optimized_skills,
            sample_resume,
            sample_job_description,
        )

        assert "Fallback summary that should still be used" in result
