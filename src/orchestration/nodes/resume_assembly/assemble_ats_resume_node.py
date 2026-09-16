"""Graph node that assembles the optimized sections into one ATS resume."""

from src.agents.ats_optimizer import create_ats_optimizer_agent
from src.agents.ats_optimizer.models import AtsOptimizedResume
from src.data_models.resume import Experience
from src.formatters.ats_optimization_formatter import format_ats_optimization_context
from src.orchestration.crew_task_execution import run_agent_task
from src.orchestration.nodes.node_contract import StateUpdate
from src.orchestration.nodes.node_logging import log_node_execution
from src.orchestration.state import ResumeEnhancementPipelineState, require


@log_node_execution("assemble_ats_resume")
def assemble_ats_resume(state: ResumeEnhancementPipelineState) -> StateUpdate:
    """Assemble optimized sections while preserving verified experience entries."""
    experience = require(state["optimized_experience"], "optimized_experience")
    context = format_ats_optimization_context(
        professional_summary=require(state["professional_summary"], "professional_summary"),
        optimized_experience=experience,
        optimized_skills=require(state["optimized_skills"], "optimized_skills"),
        original_resume=require(state["resume"], "resume"),
        job_description=require(state["job_description"], "job_description"),
        format_type="toon",
    )
    assembled = _request_ats_resume(context, state["run_id"])
    verified = _use_verified_experience(assembled, experience.optimized_experiences)
    return {"optimized_resume": verified}


def _request_ats_resume(context: str, run_id: str) -> AtsOptimizedResume:
    """Run the configured ATS optimizer and return its structured resume."""
    return run_agent_task(
        agent=create_ats_optimizer_agent(),
        task_name="optimize_ats_resume_task",
        context=context,
        output_model=AtsOptimizedResume,
        run_id=run_id,
    )


def _use_verified_experience(
    assembled: AtsOptimizedResume,
    verified_experience: list[Experience],
) -> AtsOptimizedResume:
    """Replace assembler-generated experience with truth-checked upstream entries."""
    final_resume = assembled.final_resume.model_copy(
        update={"work_experience": verified_experience}
    )
    return assembled.model_copy(update={"final_resume": final_resume})


__all__ = ["assemble_ats_resume"]
