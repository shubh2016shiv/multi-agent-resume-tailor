"""Stage 3 professional summary node."""

from src.agents.professional_summary import create_professional_summary_agent
from src.agents.professional_summary.models import ProfessionalSummary
from src.formatters.professional_summary_formatter import format_professional_summary_context
from src.orchestration.crew_task_execution import run_agent_task
from src.orchestration.state import ResumeEnhancementPipelineState


def write_professional_summary(state: ResumeEnhancementPipelineState) -> dict:
    """Generate a professional summary tailored to the job description.

    Reads: resume, job_description, and alignment_strategy from prior stages.
    Writes: professional_summary.
    Returns: partial state with the typed ProfessionalSummary.
    """
    assert state["resume"] is not None, "resume must be set before summary writing"
    assert state["job_description"] is not None, "job_description must be set before summary writing"
    assert state["alignment_strategy"] is not None, "alignment_strategy must be set before summary writing"
    context = format_professional_summary_context(
        resume=state["resume"],
        job_description=state["job_description"],
        strategy=state["alignment_strategy"],
        format_type="toon",
    )
    agent = create_professional_summary_agent()
    professional_summary = run_agent_task(
        agent=agent,
        task_name="write_professional_summary_task",
        context=context,
        output_model=ProfessionalSummary,
    )
    return {"professional_summary": professional_summary}
