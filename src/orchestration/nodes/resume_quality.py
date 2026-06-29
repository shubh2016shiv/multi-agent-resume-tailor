"""Stage 5 resume quality evaluation node."""

import time

from src.agents.quality_feedback import create_quality_feedback_agent
from src.core.logger import get_logger
from src.data_models.evaluation import (
    AtsCheckStatus,
    ATSMetrics,
    QualityFeedback,
    RenderedStructureEvaluation,
    ResumeQualityReport,
)
from src.data_models.job import JobDescription
from src.data_models.resume import Resume
from src.formatters.quality_feedback_formatter import format_quality_feedback_context
from src.orchestration.crew_task_execution import run_agent_task
from src.orchestration.human_review_policy import is_ats_unverifiable
from src.orchestration.state import ResumeEnhancementPipelineState
from src.resume_quality_evaluation import (
    apply_resume_quality_gate,
    calculate_overall_quality_score,
    evaluate_job_alignment,
    evaluate_rendered_structure,
    evaluate_resume_truthfulness,
)

logger = get_logger(__name__)


def evaluate_resume_quality(state: ResumeEnhancementPipelineState) -> dict:
    """Validate the optimized resume for quality and consistency.

    Reads: optimized_resume, original resume, and job_description.
    Writes the quality report and rendered-structure evaluation.
    Returns: partial state with both typed artifacts.
    """
    start_time = time.monotonic()
    logger.info(
        "pipeline_stage_started",
        stage="evaluate_resume_quality",
        run_id=state["run_id"],
    )
    assert state["optimized_resume"] is not None, (
        "optimized_resume must be set before quality assurance"
    )
    assert state["resume"] is not None, "resume must be set before quality assurance"
    assert state["job_description"] is not None, (
        "job_description must be set before quality assurance"
    )
    quality_feedback = _request_quality_feedback(state)
    quality_report, structure_evaluation = _ground_quality_dimensions(
        quality_feedback=quality_feedback,
        original_resume=state["resume"],
        revised_resume=state["optimized_resume"].final_resume,
        job=state["job_description"],
    )
    # An unverifiable ATS outcome (INCONCLUSIVE: no .tex to inspect) escalates to human
    # review here -- there is nothing to patch. A FAIL is left False: the patch node tries
    # a deterministic restore first. The full escalation policy lives in human_review_policy.
    human_review_required = (
        is_ats_unverifiable(structure_evaluation) or not quality_report.relevance.is_conclusive
    )
    duration_ms = round((time.monotonic() - start_time) * 1000)
    logger.info(
        "pipeline_stage_completed",
        stage="evaluate_resume_quality",
        run_id=state["run_id"],
        duration_ms=duration_ms,
    )
    return {
        "quality_report": quality_report,
        "rendered_structure_evaluation": structure_evaluation,
        "human_review_required": human_review_required,
    }


def _request_quality_feedback(
    state: ResumeEnhancementPipelineState,
) -> QualityFeedback:
    """Request optional LLM feedback without allowing failure to affect evaluation."""
    assert state["optimized_resume"] is not None, "optimized_resume is required for feedback"
    assert state["resume"] is not None, "resume is required for feedback"
    assert state["job_description"] is not None, "job_description is required for feedback"
    context = format_quality_feedback_context(
        optimized_resume=state["optimized_resume"],
        original_resume=state["resume"],
        job=state["job_description"],
        format_type="toon",
    )
    try:
        return run_agent_task(
            agent=create_quality_feedback_agent(),
            task_name="write_quality_feedback_task",
            context=context,
            output_model=QualityFeedback,
            run_id=state["run_id"],
        )
    except Exception as error:  # noqa: BLE001 -- advisory failure must not block evaluation
        logger.warning("Quality feedback unavailable", error=str(error))
        return QualityFeedback(
            assessment_summary="Automated narrative feedback was unavailable.",
            feedback_for_improvement=None,
        )


def _ground_quality_dimensions(
    quality_feedback: QualityFeedback,
    original_resume: Resume,
    revised_resume: Resume,
    job: JobDescription,
) -> tuple[ResumeQualityReport, RenderedStructureEvaluation]:
    """Replace the agent's three self-assessed dimensions with code-owned, grounded scores.

    The LLM's narration is where the false positives live (supported facts flagged as
    exaggerations, single-column resumes called multi-column). So code overrides all three
    dimensions from mechanical engines, re-blends the overall from the documented weights,
    applies the deterministic pass/fail gate, and then HARD-BLOCKS: a non-PASS rendered-ATS
    check overrides any passing score, because the rendered artifact outranks self-cert.

    Returns the grounded report and the rendered-ATS outcome (persisted for auditability).
    """
    grounded_accuracy = evaluate_resume_truthfulness(original_resume, revised_resume)
    grounded_relevance = evaluate_job_alignment(revised_resume, job)
    ats_outcome = evaluate_rendered_structure(revised_resume)
    grounded_ats = ATSMetrics(
        ats_score=ats_outcome.ats_score,
        keyword_coverage=grounded_relevance.ats_keyword_coverage,
        formatting_issues=ats_outcome.violations,
        justification=ats_outcome.detail,
    )
    grounded_overall = calculate_overall_quality_score(
        grounded_accuracy.accuracy_score,
        grounded_relevance.relevance_score,
        ats_outcome.ats_score,
    )
    quality_report = ResumeQualityReport(
        accuracy=grounded_accuracy,
        relevance=grounded_relevance,
        ats_optimization=grounded_ats,
        overall_quality_score=grounded_overall,
        passes_quality_gate=False,
        assessment_summary=quality_feedback.assessment_summary,
        feedback_for_improvement=quality_feedback.feedback_for_improvement,
    )
    quality_report = apply_resume_quality_gate(quality_report)
    logger.info(
        "quality_scores_computed",
        accuracy_score=grounded_accuracy.accuracy_score,
        relevance_score=grounded_relevance.relevance_score,
        ats_score=ats_outcome.ats_score,
        overall_score=grounded_overall,
        passes_gate=quality_report.passes_quality_gate,
    )
    # Hard block: the rendered ATS verdict is authoritative. FAIL or INCONCLUSIVE blocks
    # release even if the blended score cleared the threshold; self-cert never overrides.
    if ats_outcome.status is not AtsCheckStatus.PASS:
        quality_report = quality_report.model_copy(update={"passes_quality_gate": False})
        logger.info(
            "quality_hard_block_applied",
            reason="ats_not_pass",
            ats_status=ats_outcome.status.value,
        )
    if not grounded_relevance.is_conclusive:
        quality_report = quality_report.model_copy(update={"passes_quality_gate": False})
        logger.info(
            "quality_hard_block_applied",
            reason="relevance_inconclusive",
        )
    return quality_report, ats_outcome
