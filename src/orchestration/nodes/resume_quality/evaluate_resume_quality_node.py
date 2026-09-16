"""Graph node that evaluates the assembled resume with grounded quality scores."""

from src.data_models.evaluation import AtsCheckStatus
from src.orchestration.nodes.node_contract import StateUpdate
from src.orchestration.nodes.node_logging import log_node_execution
from src.orchestration.nodes.resume_quality.ground_quality_scores import (
    ground_quality_scores,
)
from src.orchestration.state import ResumeEnhancementPipelineState, require


@log_node_execution("evaluate_resume_quality")
def evaluate_resume_quality(state: ResumeEnhancementPipelineState) -> StateUpdate:
    """Return deterministic quality results and review escalation state."""
    report, structure = ground_quality_scores(
        require(state["resume"], "resume"),
        require(state["optimized_resume"], "optimized_resume").final_resume,
        require(state["job_description"], "job_description"),
    )
    needs_review = (
        structure.status is AtsCheckStatus.INCONCLUSIVE or not report.relevance.is_conclusive
    )
    return {
        "quality_report": report,
        "rendered_structure_evaluation": structure,
        "human_review_required": needs_review,
    }


__all__ = ["evaluate_resume_quality"]
