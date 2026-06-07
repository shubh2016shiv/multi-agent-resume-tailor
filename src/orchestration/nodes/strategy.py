"""Stage 2 strategy node for the resume enhancement graph."""

from src.agents.gap_analysis import create_gap_analysis_agent
from src.data_models.strategy import AlignmentStrategy
from src.formatters.gap_analysis_formatter import format_gap_analysis_context
from src.orchestration.crew_task_execution import run_agent_task
from src.orchestration.state import ResumeEnhancementPipelineState


def run_gap_analysis(state: ResumeEnhancementPipelineState) -> dict:
    """Identify resume/job gaps and produce a tailoring strategy.

    Reads: resume and job_description, both set by Stage 1.
    Writes: alignment_strategy.
    Returns: partial state with the AlignmentStrategy for downstream writers.
    """
    context = format_gap_analysis_context(
        resume=state["resume"],
        job_description=state["job_description"],
        format_type="toon",
    )
    agent = create_gap_analysis_agent()
    alignment_strategy = run_agent_task(
        agent=agent,
        task_name="create_alignment_strategy_task",
        context=context,
        output_model=AlignmentStrategy,
    )
    return {"alignment_strategy": alignment_strategy}
