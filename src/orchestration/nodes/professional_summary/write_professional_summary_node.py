"""Graph node that writes the tailored professional summary."""

from src.agents.professional_summary import create_professional_summary_agent
from src.agents.professional_summary.models import ProfessionalSummary
from src.formatters.professional_summary_formatter import format_professional_summary_context
from src.orchestration.crew_task_execution import run_agent_task
from src.orchestration.nodes.node_contract import StateUpdate
from src.orchestration.nodes.node_logging import log_node_execution
from src.orchestration.nodes.professional_summary.enforce_summary_quality import (
    enforce_summary_quality_gate,
)
from src.orchestration.state import ResumeEnhancementPipelineState, require


@log_node_execution("write_professional_summary")
def write_professional_summary(state: ResumeEnhancementPipelineState) -> StateUpdate:
    """Build, validate, and return a summary from the required tailoring state."""
    resume = require(state["resume"], "resume")
    job = require(state["job_description"], "job_description")
    strategy = require(state["alignment_strategy"], "alignment_strategy")
    context = format_professional_summary_context(
        resume=resume,
        job_description=job,
        strategy=strategy,
        format_type="toon",
    )
    summary = _generate_professional_summary(context, state["run_id"])
    enforce_summary_quality_gate(summary)
    return {"professional_summary": summary}


def _generate_professional_summary(context: str, run_id: str) -> ProfessionalSummary:
    """Run the configured summary agent and return its structured response."""
    return run_agent_task(
        agent=create_professional_summary_agent(),
        task_name="write_professional_summary_task",
        context=context,
        output_model=ProfessionalSummary,
        run_id=run_id,
    )


__all__ = ["write_professional_summary"]
