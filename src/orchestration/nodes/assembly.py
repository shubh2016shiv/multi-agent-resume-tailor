"""Stage 4 ATS resume assembly node."""

import time

from src.agents.ats_optimizer import create_ats_optimizer_agent
from src.agents.ats_optimizer.models import AtsOptimizedResume
from src.core.logger import get_logger
from src.data_models.resume import Experience
from src.formatters.ats_optimization_formatter import format_ats_optimization_context
from src.orchestration.crew_task_execution import run_agent_task
from src.orchestration.state import ResumeEnhancementPipelineState

logger = get_logger(__name__)


def assemble_ats_resume(state: ResumeEnhancementPipelineState) -> dict:
    """Assemble all optimized sections into a single ATS-compliant resume.

    Reads: professional_summary, optimized_experience, optimized_skills,
    original resume, and job description.
    Writes: optimized_resume.
    Returns: partial state with the typed AtsOptimizedResume.
    """
    start_time = time.monotonic()
    logger.info(
        "pipeline_stage_started",
        stage="assemble_ats_resume",
        run_id=state["run_id"],
    )
    assert state["professional_summary"] is not None, (
        "professional_summary must be set before assembly"
    )
    assert state["optimized_experience"] is not None, (
        "optimized_experience must be set before assembly"
    )
    assert state["optimized_skills"] is not None, "optimized_skills must be set before assembly"
    assert state["resume"] is not None, "resume must be set before assembly"
    assert state["job_description"] is not None, "job_description must be set before assembly"
    context = format_ats_optimization_context(
        professional_summary=state["professional_summary"],
        optimized_experience=state["optimized_experience"],
        optimized_skills=state["optimized_skills"],
        original_resume=state["resume"],
        job_description=state["job_description"],
        format_type="toon",
    )
    agent = create_ats_optimizer_agent()
    optimized_resume = run_agent_task(
        agent=agent,
        task_name="optimize_ats_resume_task",
        context=context,
        output_model=AtsOptimizedResume,
    )
    verified_resume = _preserve_verified_experience(
        optimized_resume,
        state["optimized_experience"].optimized_experiences,
    )
    duration_ms = round((time.monotonic() - start_time) * 1000)
    logger.info(
        "pipeline_stage_completed",
        stage="assemble_ats_resume",
        run_id=state["run_id"],
        duration_ms=duration_ms,
    )
    return {"optimized_resume": verified_resume}


def _preserve_verified_experience(
    optimized_resume: AtsOptimizedResume,
    verified_experiences: list[Experience],
) -> AtsOptimizedResume:
    """Replace assembler-written experience with the verified upstream entries."""
    final_resume = optimized_resume.final_resume.model_copy(
        update={"work_experience": verified_experiences}
    )
    return optimized_resume.model_copy(update={"final_resume": final_resume})
