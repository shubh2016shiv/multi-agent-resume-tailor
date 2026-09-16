"""Contracts for the resume quality orchestration node."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest

from src.data_models.evaluation import (
    AtsCheckStatus,
    RenderedStructureEvaluation,
)
from src.orchestration.nodes.resume_quality.evaluate_resume_quality_node import (
    evaluate_resume_quality,
)
from src.orchestration.state import ResumeEnhancementPipelineState


@pytest.mark.parametrize(
    ("ats_status", "relevance_is_conclusive", "expected"),
    [
        (AtsCheckStatus.INCONCLUSIVE, True, True),
        (AtsCheckStatus.PASS, False, True),
        (AtsCheckStatus.PASS, True, False),
    ],
)
def test_quality_escalates_only_unverifiable_results(
    ats_status: AtsCheckStatus,
    relevance_is_conclusive: bool,
    expected: bool,
) -> None:
    """ATS or relevance uncertainty should require human review."""
    report = SimpleNamespace(relevance=SimpleNamespace(is_conclusive=relevance_is_conclusive))
    structure = RenderedStructureEvaluation(
        status=ats_status,
        violations=[],
        ats_score=100.0,
        detail="Test structure result.",
    )
    state = cast(
        ResumeEnhancementPipelineState,
        {
            "run_id": "test-run",
            "resume": object(),
            "optimized_resume": SimpleNamespace(final_resume=object()),
            "job_description": object(),
        },
    )
    with patch(
        "src.orchestration.nodes.resume_quality.evaluate_resume_quality_node.ground_quality_scores",
        return_value=(report, structure),
    ):
        result = evaluate_resume_quality(state)

    assert result["human_review_required"] is expected
