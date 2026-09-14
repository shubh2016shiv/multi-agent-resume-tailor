"""Turn what the graph produced into this project's OrchestrationResult, and save it.

There is no LangGraph pattern to learn in this file -- only how THIS pipeline turns
a raw state dict into its own public result type. runner.py calls
finalize_pipeline_output() exactly once, right after invoke() returns; everything
else here is that one function's own machinery, pulled into named pieces instead of
nested inside it.

THE ONE DECISION THIS FILE MAKES: paused, or completed?
    finalize_pipeline_output() checks for LangGraph's own "__interrupt__" marker in
    the output. If it is there, the run paused mid-way, and
    _build_paused_orchestration_result() builds a result naming what still needs
    the candidate's input. Otherwise the run reached the end of the graph, and
    _build_completed_orchestration_result() builds the full result -- including the
    run's final disposition, which is a judgment call this file does not make
    itself; it only calls derive_run_disposition() (see human_review_policy.py).
"""

import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from src.core.logger import get_logger
from src.core.settings import get_config
from src.data_models.orchestration import OrchestrationResult, RunDisposition
from src.hitl.professional_experience.models import ExperienceClarificationPausedRunManifest
from src.hitl.professional_experience.persistence import PausedRunLayout, save_paused_run_state
from src.orchestration.human_review_policy import derive_run_disposition
from src.orchestration.state import ResumeEnhancementPipelineState, require
from src.tools.engines.document_rendering.output_paths import resume_output_dir

logger = get_logger(__name__)


def finalize_pipeline_output(
    pipeline: CompiledStateGraph,
    config: RunnableConfig,
    output: dict,
    run_id: str,
    resume_path: str,
    jd_path: str,
    paused_run_layout: PausedRunLayout | None,
) -> OrchestrationResult:
    """Convert an invoke() result into a paused or completed orchestration result.

    A run that pauses again after a resume reuses the same paused-run directory,
    so the candidate always has exactly one folder to work in.
    """
    if "__interrupt__" in output:  # LangGraph paused on an interrupt boundary
        snapshot = pipeline.get_state(config)
        snapshot_state = cast(ResumeEnhancementPipelineState, snapshot.values)
        layout = paused_run_layout or PausedRunLayout.at(
            _paused_run_directory(snapshot_state, run_id)
        )
        paused_at = datetime.now(UTC)
        manifest = ExperienceClarificationPausedRunManifest(
            run_id=run_id,
            resume_path=resume_path,
            jd_path=jd_path,
            paused_at=paused_at,
            expires_at=paused_at + timedelta(hours=get_config().workflow.clarification_ttl_hours),
        )
        save_paused_run_state(
            layout,
            manifest,
            snapshot_state.get("experience_clarifications") or [],
        )
        result = _build_paused_orchestration_result(
            snapshot_state,
            paused_run_path=str(layout.root),
        )
        _persist_result(result)
        return result

    final_state = cast(ResumeEnhancementPipelineState, output)
    result = _build_completed_orchestration_result(final_state)
    _persist_result(result)
    return result


def _build_paused_orchestration_result(
    state: ResumeEnhancementPipelineState,
    paused_run_path: str,
) -> OrchestrationResult:
    """Build the public result for a run paused before ATS assembly."""
    clarifications_requested = state.get("experience_clarifications") or []
    return OrchestrationResult(
        original_resume=require(state["resume"], "resume"),
        job_description=require(state["job_description"], "job_description"),
        strategy=require(state["alignment_strategy"], "alignment_strategy"),
        optimized_resume=None,
        quality_report=None,
        rendered_artifacts=None,
        clarifications_requested=clarifications_requested,
        disposition=RunDisposition.NEEDS_CANDIDATE_INPUT,
        paused_run_path=paused_run_path,
    )


def _build_completed_orchestration_result(
    state: ResumeEnhancementPipelineState,
) -> OrchestrationResult:
    """Build the public result for a completed end-to-end run."""
    clarifications_requested = state.get("experience_clarifications") or []
    quality_report = require(state["quality_report"], "quality_report")
    return OrchestrationResult(
        original_resume=require(state["resume"], "resume"),
        job_description=require(state["job_description"], "job_description"),
        strategy=require(state["alignment_strategy"], "alignment_strategy"),
        optimized_resume=require(state["optimized_resume"], "optimized_resume"),
        quality_report=quality_report,
        rendered_artifacts=state["rendered_artifacts"],
        clarifications_requested=clarifications_requested,
        disposition=derive_run_disposition(
            human_review_required=state["human_review_required"],
            quality_gate_passed=quality_report.passes_quality_gate,
            has_candidate_questions=bool(clarifications_requested),
        ),
        paused_run_path=None,
    )


def _persist_result(result: OrchestrationResult) -> None:
    """Persist one orchestration result next to the paused run or rendered artifacts.

    The clarification sheet is deliberately NOT written here. A paused run already
    got one from save_paused_run_state, and writing it again produced two
    identical writes to the same path. A completed run must not get one at all:
    its folder has no manifest and no checkpoint, so a sheet there would invite
    the candidate to answer questions that can never be resumed.
    """
    output_dir = _result_output_dir(result)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    logger.info(
        "run_result_saved",
        path=str(path),
        disposition=result.disposition.value,
    )


def _result_output_dir(result: OrchestrationResult) -> Path:
    """Return the directory where one run result should be persisted."""
    if result.paused_run_path:
        return Path(result.paused_run_path)
    base_dir = Path(get_config().file_paths.output_dir)
    resume_for_path = (
        result.optimized_resume.final_resume
        if result.optimized_resume is not None
        else result.original_resume
    )
    return resume_output_dir(resume_for_path, result.job_description, base_dir)


def _paused_run_directory(
    state: ResumeEnhancementPipelineState,
    run_id: str,
) -> Path:
    """Return the local folder where one paused clarification run should live."""
    parent_dir = resume_output_dir(
        require(state["resume"], "resume"),
        require(state["job_description"], "job_description"),
        Path(get_config().file_paths.output_dir),
    )
    return parent_dir / f"paused_run_{run_id}"


def log_pipeline_completion(
    run_id: str,
    result: OrchestrationResult,
    start_time: float,
) -> None:
    """Log the final public disposition of a run."""
    duration_ms = round((time.monotonic() - start_time) * 1000)
    logger.info(
        "pipeline_run_completed",
        run_id=run_id,
        gate_passed=(
            result.quality_report.passes_quality_gate if result.quality_report is not None else None
        ),
        disposition=result.disposition.value,
        duration_ms=duration_ms,
    )
