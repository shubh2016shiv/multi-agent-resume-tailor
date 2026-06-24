"""Contracts for the resume quality orchestration node."""

from unittest.mock import patch

from src.data_models.evaluation import QualityFeedback
from src.orchestration.nodes.resume_quality import _request_quality_feedback


def _state_with_required_feedback_inputs() -> dict:
    """Return the minimum mapping consumed by the feedback request helper."""
    return {
        "optimized_resume": object(),
        "resume": object(),
        "job_description": object(),
    }


def test_feedback_request_returns_agent_feedback() -> None:
    """Successful advisory execution returns the agent's narrow feedback model."""
    expected = QualityFeedback(
        assessment_summary="Strong source-faithful tailoring.",
        feedback_for_improvement=None,
    )
    with (
        patch(
            "src.orchestration.nodes.resume_quality.format_quality_feedback_context",
            return_value="context",
        ),
        patch("src.orchestration.nodes.resume_quality.create_quality_feedback_agent"),
        patch(
            "src.orchestration.nodes.resume_quality.run_agent_task",
            return_value=expected,
        ),
    ):
        feedback = _request_quality_feedback(_state_with_required_feedback_inputs())

    assert feedback == expected


def test_feedback_failure_returns_neutral_advisory_fallback() -> None:
    """An advisory-agent failure returns fallback prose instead of blocking evaluation."""
    with (
        patch(
            "src.orchestration.nodes.resume_quality.format_quality_feedback_context",
            return_value="context",
        ),
        patch(
            "src.orchestration.nodes.resume_quality.create_quality_feedback_agent",
            side_effect=RuntimeError("agent unavailable"),
        ),
    ):
        feedback = _request_quality_feedback(_state_with_required_feedback_inputs())

    assert feedback.assessment_summary == "Automated narrative feedback was unavailable."
    assert feedback.feedback_for_improvement is None
