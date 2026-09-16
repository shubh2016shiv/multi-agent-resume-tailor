"""Contracts for the ATS resume assembly node."""

from datetime import date
from unittest.mock import patch

from src.agents.ats_optimizer.models import AtsOptimizedResume
from src.agents.professional_experience.models import OptimizedExperienceSection
from src.data_models.resume import Experience, Resume
from src.orchestration.nodes.resume_assembly import assemble_ats_resume


def _experience(description: str) -> Experience:
    """Build the smallest valid experience used by this test."""
    return Experience(
        experience_id="exp-1",
        job_title="Engineer",
        company_name="Acme",
        start_date=date(2020, 1, 1),
        end_date=None,
        location=None,
        description=description,
    )


def _resume(experience: Experience) -> Resume:
    """Build the smallest valid resume used by this test."""
    return Resume(
        full_name="Candidate",
        email="candidate@example.com",
        phone_number=None,
        location=None,
        website_or_portfolio=None,
        professional_summary="Engineer.",
        work_experience=[experience],
    )


def test_assembly_preserves_truth_checked_experience() -> None:
    """The assembly agent must not replace verified upstream experience content."""
    verified = _experience("Truth-checked experience.")
    agent_result = AtsOptimizedResume(final_resume=_resume(_experience("LLM rewrite.")))
    state = {
        "run_id": "run-1",
        "professional_summary": object(),
        "optimized_experience": OptimizedExperienceSection(optimized_experiences=[verified]),
        "optimized_skills": object(),
        "resume": _resume(verified),
        "job_description": object(),
    }

    with (
        patch(
            "src.orchestration.nodes.resume_assembly.assemble_ats_resume_node."
            "format_ats_optimization_context",
            return_value="context",
        ),
        patch(
            "src.orchestration.nodes.resume_assembly.assemble_ats_resume_node._request_ats_resume",
            return_value=agent_result,
        ),
    ):
        result = assemble_ats_resume(state)  # type: ignore[arg-type]

    assembled = result["optimized_resume"]
    assert isinstance(assembled, AtsOptimizedResume)
    assert assembled.final_resume.work_experience == [verified]
