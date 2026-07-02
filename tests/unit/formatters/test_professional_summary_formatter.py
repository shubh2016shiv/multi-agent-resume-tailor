"""Unit tests for src/formatters/professional_summary_formatter.py.

Contracts under test:
  build_professional_summary_payload(...) — returns only summary-relevant data,
                                           ranked for narrative usefulness.
  format_professional_summary_context(...) — renders that payload without reintroducing
                                            dropped resume/job fields.
"""

from datetime import date

from src.data_models.resume import Experience, Resume
from src.formatters.professional_summary_formatter import (
    build_professional_summary_payload,
    format_professional_summary_context,
)


def _resume_with_rankable_roles(sample_resume: Resume) -> Resume:
    """Return a resume with four roles so ranking/capping behavior is observable."""
    return sample_resume.model_copy(
        update={
            "work_experience": [
                Experience(
                    experience_id="exp-1",
                    job_title="Data Scientist",
                    company_name="Insight Labs",
                    start_date=date(2023, 1, 1),
                    end_date=date(2024, 1, 1),
                    is_current_position=False,
                    location="Remote",
                    description="Built Python services and AWS ML workflows for production systems.",
                    achievements=[
                        "Improved model serving reliability for production APIs.",
                        "Shipped AWS-based ML workflows for customer-facing analytics.",
                        "Partnered with product on release readiness.",
                    ],
                    skills_used=["Python", "AWS", "REST APIs"],
                ),
                Experience(
                    experience_id="exp-2",
                    job_title="Backend Engineer",
                    company_name="Cloud Forge",
                    start_date=date(2021, 1, 1),
                    end_date=date(2022, 12, 31),
                    is_current_position=False,
                    location="Remote",
                    description="Maintained payment services and API integrations.",
                    achievements=[
                        "Stabilized service incidents for a payment workflow.",
                        "Refined API integrations across vendors.",
                    ],
                    skills_used=["Java", "Payments"],
                ),
                Experience(
                    experience_id="exp-3",
                    job_title="ML Platform Engineer",
                    company_name="Ops Grid",
                    start_date=date(2019, 1, 1),
                    end_date=date(2020, 12, 31),
                    is_current_position=False,
                    location="Remote",
                    description="Ran AWS platform automation for Python-based data products.",
                    achievements=[
                        "Scaled CI/CD for ML releases on AWS.",
                        "Reduced deployment toil for Python services.",
                    ],
                    skills_used=["Python", "AWS", "CI/CD"],
                ),
                Experience(
                    experience_id="exp-4",
                    job_title="Support Analyst",
                    company_name="Legacy Systems",
                    start_date=date(2017, 1, 1),
                    end_date=date(2018, 12, 31),
                    is_current_position=False,
                    location="Onsite",
                    description="Handled internal reporting and ticket queues.",
                    achievements=["Answered support tickets for internal users."],
                    skills_used=["Excel", "Support"],
                ),
            ]
        }
    )


class TestBuildProfessionalSummaryPayload:
    """Tests for the structured payload used by the summary writer."""

    def test_payload_drops_summary_stage_gaps_and_uses_supported_role_vocabulary(
        self,
        sample_resume,
        sample_job_description,
        sample_alignment_strategy,
    ):
        """Contract: summary payload keeps supported role vocabulary and omits gap payload."""
        payload = build_professional_summary_payload(
            sample_resume,
            sample_job_description,
            sample_alignment_strategy,
        )

        strategy_context = payload["summary_strategy"]

        assert "supported_role_vocabulary" in strategy_context
        assert strategy_context["supported_role_vocabulary"] == ["Python", "AWS"]
        assert "keywords_to_integrate" not in strategy_context
        assert "must_have_gaps" not in strategy_context

    def test_payload_caps_experience_highlights_and_ranks_more_relevant_roles_first(
        self,
        sample_resume,
        sample_job_description,
        sample_alignment_strategy,
    ):
        """Contract: summary payload keeps the three strongest supported roles in ranked order."""
        ranked_resume = _resume_with_rankable_roles(sample_resume)

        payload = build_professional_summary_payload(
            ranked_resume,
            sample_job_description,
            sample_alignment_strategy,
        )

        experience_highlights = payload["candidate_background"]["experience_highlights"]

        assert len(experience_highlights) == 3
        assert [role["company_name"] for role in experience_highlights] == [
            "Insight Labs",
            "Ops Grid",
            "Cloud Forge",
        ]

    def test_payload_replaces_noisy_guidance_with_supported_summary_guidance(
        self,
        sample_resume,
        sample_job_description,
        sample_alignment_strategy,
    ):
        """Contract: formatter rewrites summary guidance around supported priorities."""
        payload = build_professional_summary_payload(
            sample_resume,
            sample_job_description,
            sample_alignment_strategy,
        )

        guidance = payload["summary_strategy"]["professional_summary_guidance"]

        assert guidance.startswith("The summary's one job is to make a recruiter")
        assert "supported role vocabulary" in guidance
        assert "measurable outcome" in guidance
        assert "Open with backend and cloud outcomes." not in guidance


class TestFormatProfessionalSummaryContext:
    """Tests for the rendered summary context string."""

    def test_format_professional_summary_context_omits_raw_fields_and_old_summary(
        self,
        sample_resume,
        sample_job_description,
        sample_alignment_strategy,
    ):
        """Contract: rendered context includes summary inputs only, not dropped raw fields."""
        result = format_professional_summary_context(
            sample_resume,
            sample_job_description,
            sample_alignment_strategy,
        )

        assert result.startswith("candidate_background:")
        assert "professional_summary:" not in result
        assert "current_professional_summary:" not in result
        assert "supported_role_vocabulary:" in result
        assert "full_text:" not in result
        assert "email:" not in result
        assert "must_have_gaps:" not in result
        assert "nice_to_have" not in result
