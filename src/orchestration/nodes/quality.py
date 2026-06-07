"""Stage 5 quality assurance node."""

from src.agents.quality_assessment import create_quality_assessment_agent
from src.agents.quality_assessment.engines import apply_quality_gate
from src.data_models.evaluation import QualityReport
from src.formatters.quality_assurance_formatter import format_quality_assurance_context
from src.orchestration.crew_task_execution import run_agent_task
from src.orchestration.state import ResumeEnhancementPipelineState


def run_quality_assurance(state: ResumeEnhancementPipelineState) -> dict:
    """Validate the optimized resume for quality and consistency.

    Reads: optimized_resume, original resume, and job_description.
    Writes: qa_report (with the code-owned pass/fail gate already applied).
    Returns: partial state with the typed QualityReport.
    """
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
    # The agent's passed_quality_threshold is advisory; code sets it authoritatively.
    qa_report = apply_quality_gate(qa_report)
    return {"qa_report": qa_report}
