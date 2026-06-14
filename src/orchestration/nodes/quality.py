"""Stage 5 quality assurance node."""

from src.agents.quality_assessment import create_quality_assessment_agent
from src.agents.quality_assessment.engines import apply_quality_gate
from src.data_models.evaluation import (
    AtsCheckStatus,
    ATSMetrics,
    AtsRenderedOutcome,
    QualityReport,
)
from src.data_models.job import JobDescription
from src.data_models.resume import Resume
from src.evaluation_rubrics import blend_overall, grade_accuracy, grade_ats, grade_relevance
from src.formatters.quality_assurance_formatter import format_quality_assurance_context
from src.orchestration.crew_task_execution import run_agent_task
from src.orchestration.state import ResumeEnhancementPipelineState


def run_quality_assurance(state: ResumeEnhancementPipelineState) -> dict:
    """Validate the optimized resume for quality and consistency.

    Reads: optimized_resume, original resume, and job_description.
    Writes: qa_report (gate applied) and ats_rendered_outcome (the rendered-ATS verdict).
    Returns: partial state with both typed artifacts.
    """
    assert state["optimized_resume"] is not None, "optimized_resume must be set before quality assurance"
    assert state["resume"] is not None, "resume must be set before quality assurance"
    assert state["job_description"] is not None, "job_description must be set before quality assurance"
    context = format_quality_assurance_context(
        optimized_resume=state["optimized_resume"],
        original_resume=state["resume"],
        job=state["job_description"],
        format_type="toon",
    )
    agent = create_quality_assessment_agent()
    qa_report = run_agent_task(
        agent=agent,
        task_name="assess_quality_task",
        context=context,
        output_model=QualityReport,
    )
    qa_report, ats_outcome = _ground_quality_dimensions(
        qa_report=qa_report,
        original_resume=state["resume"],
        revised_resume=state["optimized_resume"].final_resume,
        job=state["job_description"],
    )
    return {"qa_report": qa_report, "ats_rendered_outcome": ats_outcome}


def _ground_quality_dimensions(
    qa_report: QualityReport,
    original_resume: Resume,
    revised_resume: Resume,
    job: JobDescription,
) -> tuple[QualityReport, AtsRenderedOutcome]:
    """Replace the agent's three self-assessed dimensions with code-owned, grounded scores.

    The LLM's narration is where the false positives live (supported facts flagged as
    exaggerations, single-column resumes called multi-column). So code overrides all three
    dimensions from mechanical engines, re-blends the overall from the documented weights,
    applies the deterministic pass/fail gate, and then HARD-BLOCKS: a non-PASS rendered-ATS
    check overrides any passing score, because the rendered artifact outranks self-cert.

    Returns the grounded report and the rendered-ATS outcome (persisted for auditability).
    """
    grounded_accuracy = grade_accuracy(original_resume, revised_resume)
    grounded_relevance = grade_relevance(revised_resume, job)
    ats_outcome = grade_ats(revised_resume)
    grounded_ats = ATSMetrics(
        ats_score=ats_outcome.ats_score,
        keyword_coverage=grounded_relevance.must_have_skills_coverage,
        formatting_issues=ats_outcome.violations,
        justification=ats_outcome.detail,
    )
    grounded_overall = blend_overall(
        grounded_accuracy.accuracy_score,
        grounded_relevance.relevance_score,
        ats_outcome.ats_score,
    )
    qa_report = qa_report.model_copy(
        update={
            "accuracy": grounded_accuracy,
            "relevance": grounded_relevance,
            "ats_optimization": grounded_ats,
            "overall_quality_score": grounded_overall,
        }
    )
    qa_report = apply_quality_gate(qa_report)
    # Hard block: the rendered ATS verdict is authoritative. FAIL or INCONCLUSIVE blocks
    # release even if the blended score cleared the threshold; self-cert never overrides.
    if ats_outcome.status is not AtsCheckStatus.PASS:
        qa_report = qa_report.model_copy(update={"passed_quality_threshold": False})
    return qa_report, ats_outcome
