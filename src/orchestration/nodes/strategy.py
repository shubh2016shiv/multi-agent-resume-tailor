"""Stage 2 strategy node for the resume enhancement graph."""

from src.agents.gap_analysis import create_gap_analysis_agent
from src.data_models.strategy import AlignmentStrategy
from src.formatters.gap_analysis_formatter import format_gap_analysis_context
from src.orchestration.crew_task_execution import run_agent_task
from src.orchestration.state import ResumeEnhancementPipelineState
from src.tools.job_matching import match_resume_to_job


def run_gap_analysis(state: ResumeEnhancementPipelineState) -> dict:
    """Identify resume/job gaps and produce a tailoring strategy.

    Reads: resume and job_description, both set by Stage 1.
    Writes: requirement_match_report and alignment_strategy.
    Returns: partial state with both for downstream writers.

    The match (requirement coverage + keyword coverage) is computed here in code on
    the typed objects, then both persisted as a typed artifact and rendered into the
    agent's context. The agent reads those pre-computed facts -- it never reconstructs
    the resume to call a tool, which is what previously looped the stage to timeout.
    """
    assert state["resume"] is not None, "resume must be set before gap analysis"
    assert state["job_description"] is not None, "job_description must be set before gap analysis"
    match_report = match_resume_to_job(state["resume"], state["job_description"])
    context = format_gap_analysis_context(
        resume=state["resume"],
        job_description=state["job_description"],
        match_report=match_report,
        format_type="toon",
    )
    agent = create_gap_analysis_agent()
    alignment_strategy = run_agent_task(
        agent=agent,
        task_name="create_alignment_strategy_task",
        context=context,
        output_model=AlignmentStrategy,
    )
    return {
        "requirement_match_report": match_report,
        "alignment_strategy": alignment_strategy,
    }
