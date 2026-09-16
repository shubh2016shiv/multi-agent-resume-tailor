"""Graph node that builds the resume-to-job alignment strategy."""

from src.agents.gap_analysis import create_gap_analysis_agent
from src.data_models.strategy import AlignmentStrategy
from src.formatters.gap_analysis_formatter import format_gap_analysis_context
from src.orchestration.crew_task_execution import run_agent_task
from src.orchestration.nodes.node_contract import StateUpdate
from src.orchestration.nodes.node_logging import log_node_execution
from src.orchestration.state import ResumeEnhancementPipelineState, require
from src.tools.engines.job_matching import match_resume_to_job


@log_node_execution("run_gap_analysis")
def run_gap_analysis(state: ResumeEnhancementPipelineState) -> StateUpdate:
    """Calculate resume-job gaps and return the resulting alignment strategy."""
    resume = require(state["resume"], "resume")
    job = require(state["job_description"], "job_description")
    match_report = match_resume_to_job(resume, job)
    context = format_gap_analysis_context(
        resume=resume,
        job_description=job,
        match_report=match_report,
        format_type="toon",
    )
    strategy = _request_alignment_strategy(context, state["run_id"])
    return {
        "requirement_match_report": match_report,
        "alignment_strategy": strategy,
    }


def _request_alignment_strategy(context: str, run_id: str) -> AlignmentStrategy:
    """Run the configured gap-analysis agent and return its structured strategy."""
    return run_agent_task(
        agent=create_gap_analysis_agent(),
        task_name="create_alignment_strategy_task",
        context=context,
        output_model=AlignmentStrategy,
        run_id=run_id,
    )


__all__ = ["run_gap_analysis"]
