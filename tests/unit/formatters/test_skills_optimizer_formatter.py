"""Unit tests for src/formatters/skills_optimizer_formatter.py.

Contract under test: the Skills LLM sees only parsed skills and high-priority
parsed job requirements.
"""

from src.formatters.skills_optimizer_formatter import (
    build_skills_optimizer_payload,
    format_skills_optimizer_context,
)


class TestFormatSkillsOptimizerContext:
    """Tests for the skills optimizer context builder."""

    def test_context_contains_only_skills_and_relevant_job_requirements(
        self,
        sample_resume,
        sample_job_description,
    ):
        """Narrative evidence and strategy prose must never enter this task."""
        payload = build_skills_optimizer_payload(sample_resume, sample_job_description)
        result = format_skills_optimizer_context(sample_resume, sample_job_description)

        assert set(payload) == {"resume_skills", "job_requirements"}
        assert payload["resume_skills"][0]["id"] == "S001"
        assert payload["job_requirements"][0]["id"] == "J001"
        assert result.startswith("resume_skills:")
        assert "Python" in result
        assert "nice_to_have" not in result
        for forbidden in ("role_evidence", "top_achievements", "skills_guidance", "ats_keywords"):
            assert forbidden not in result
