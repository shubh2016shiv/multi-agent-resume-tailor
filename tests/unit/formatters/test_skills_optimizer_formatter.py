"""Unit tests for src/formatters/skills_optimizer_formatter.py.

Contract under test: the skills formatter returns current skills, high-priority
job requirements, and skills-specific strategy guidance.
"""

from src.formatters.skills_optimizer_formatter import format_skills_optimizer_context


class TestFormatSkillsOptimizerContext:
    """Tests for the skills optimizer context builder."""

    def test_format_skills_optimizer_context_drops_nice_to_have_requirements_and_non_skill_sections(
        self,
        sample_resume,
        sample_job_description,
        sample_alignment_strategy,
    ):
        """Contract: only must-have/should-have requirements remain, and unrelated resume sections are excluded."""
        result = format_skills_optimizer_context(
            sample_resume,
            sample_job_description,
            sample_alignment_strategy,
        )

        assert result.startswith("current_skills:")
        assert "skills_guidance:" in result
        assert "Python" in result
        assert "nice_to_have" not in result
        assert "work_experience:" not in result
        assert "professional_summary:" not in result
