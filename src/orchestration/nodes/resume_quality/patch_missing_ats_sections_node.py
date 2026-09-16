"""Graph node that restores missing ATS sections and re-grades quality."""

from src.core.logger import get_logger
from src.data_models.evaluation import AtsCheckStatus
from src.orchestration.nodes.node_contract import StateUpdate
from src.orchestration.nodes.node_logging import log_node_execution
from src.orchestration.nodes.resume_quality.ground_quality_scores import ground_quality_scores
from src.orchestration.nodes.resume_quality.restore_missing_resume_sections import (
    restore_missing_resume_sections,
)
from src.orchestration.state import ResumeEnhancementPipelineState, require

logger = get_logger(__name__)


@log_node_execution("patch_ats_assembly")
def patch_ats_assembly(state: ResumeEnhancementPipelineState) -> StateUpdate:
    """Restore empty essential sections once, then recompute the quality report."""
    original_resume = require(state["resume"], "resume")
    optimized_resume = require(state["optimized_resume"], "optimized_resume")
    patched_resume = restore_missing_resume_sections(
        optimized_resume.final_resume,
        require(state["optimized_experience"], "optimized_experience"),
        require(state["optimized_skills"], "optimized_skills"),
        original_resume,
    )
    report, structure = ground_quality_scores(
        original_resume,
        patched_resume,
        require(state["job_description"], "job_description"),
    )
    logger.info(
        "ats_patch_regraded",
        ats_score=structure.ats_score,
        accuracy_score=report.accuracy.accuracy_score,
        relevance_score=report.relevance.relevance_score,
        recovered=patched_resume != optimized_resume.final_resume,
    )
    return {
        "optimized_resume": optimized_resume.model_copy(update={"final_resume": patched_resume}),
        "quality_report": report,
        "rendered_structure_evaluation": structure,
        "human_review_required": state["human_review_required"]
        or structure.status is not AtsCheckStatus.PASS
        or not report.relevance.is_conclusive,
    }


__all__ = ["patch_ats_assembly"]
