"""Public entry points for the resume enhancement pipeline."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from src.core.logger import get_logger
from src.core.pii_mapping_store import delete_pii_mapping
from src.core.settings import get_config
from src.data_models.orchestration import OrchestrationResult, RunDisposition
from src.hitl.professional_experience.models import (
    ExperienceClarificationPausedRunManifest,
)
from src.hitl.professional_experience.persistence import (
    SHEET_FILENAME,
    PausedRunLayout,
    archive_checkpoint_database,
    load_paused_run,
    read_answered_clarifications,
    save_paused_run_state,
)
from src.observability import init_observability
from src.orchestration.checkpointing import (
    close_checkpoint_database,
    open_checkpoint_database,
)
from src.orchestration.graph import build_resume_enhancement_graph
from src.orchestration.human_review_policy import derive_run_disposition
from src.orchestration.state import (
    ResumeEnhancementPipelineState,
    new_pipeline_state,
    require,
)
from src.tools.engines.document_rendering.output_paths import resume_output_dir

logger = get_logger(__name__)

ProgressCallback = Callable[[str, str], None]


@dataclass(frozen=True)
class _RunPlan:
    """One execution's inputs, plus everything that differs between the two modes.

    Exactly one of in_flight_checkpoint_db / paused_run_layout is set:
    in_flight_checkpoint_db for a fresh run, whose checkpoint is archived into a
    paused-run directory or deleted when the run ends; paused_run_layout for a
    resume, whose checkpoint already lives in that directory.
    """

    run_id: str
    resume_path: str
    jd_path: str
    checkpointer: SqliteSaver
    pipeline_input: Any
    started_log_fields: dict[str, Any]
    in_flight_checkpoint_db: Path | None = None
    paused_run_layout: PausedRunLayout | None = None


def tailor_resume(
    resume_path: str,
    jd_path: str,
    progress_callback: ProgressCallback | None = None,
) -> OrchestrationResult:
    """Run a fresh pipeline execution from source documents."""
    init_observability("resume-tailor-agents")
    run_id = uuid4().hex
    checkpoint_db_path = _in_flight_checkpoint_db_path(run_id)
    return _execute_run(
        _RunPlan(
            run_id=run_id,
            resume_path=resume_path,
            jd_path=jd_path,
            checkpointer=open_checkpoint_database(checkpoint_db_path),
            pipeline_input=new_pipeline_state(run_id, resume_path, jd_path),
            started_log_fields={"resume_mode": "fresh"},
            in_flight_checkpoint_db=checkpoint_db_path,
        ),
        progress_callback,
    )


# HITL COMPONENT 6 -- RESUME. See
# src/hitl/professional_experience/README.md#9-component-6--resume-mechanism
def resume_paused_run(
    paused_run_path: str,
    progress_callback: ProgressCallback | None = None,
) -> OrchestrationResult:
    """Resume a previously paused professional-experience clarification run.

    Everything that can refuse the resume is checked before any pipeline
    machinery is opened: an expired run and an unanswered sheet both fail here,
    cheaply, instead of after a checkpoint connection and a graph compile.
    """
    init_observability("resume-tailor-agents")
    layout, manifest = load_paused_run(paused_run_path)
    if manifest.is_expired:
        raise ValueError(
            f"This paused run expired on {manifest.expires_at.isoformat()} and can no "
            "longer be resumed. Start a fresh run to tailor this resume again."
        )
    answered_clarifications = read_answered_clarifications(layout)
    if not answered_clarifications:
        raise ValueError(
            f"{SHEET_FILENAME} has no answered questions yet; answer at least "
            "one clarification before resuming the paused run."
        )

    return _execute_run(
        _RunPlan(
            run_id=manifest.run_id,
            resume_path=manifest.resume_path,
            jd_path=manifest.jd_path,
            checkpointer=open_checkpoint_database(layout.checkpoint_db),
            pipeline_input=Command(
                resume={"status": "candidate_answers_submitted"},
                update={"clarification_answers": answered_clarifications},
            ),
            started_log_fields={
                "resume_mode": "paused_run_resume",
                "paused_run_path": paused_run_path,
                "answered_clarifications": len(answered_clarifications),
            },
            paused_run_layout=layout,
        ),
        progress_callback,
    )


def _execute_run(
    plan: _RunPlan,
    progress_callback: ProgressCallback | None,
) -> OrchestrationResult:
    """Compile, invoke, and settle one run -- the skeleton both entry points share.

    The checkpointer is opened by the caller (so a failure to open it never reaches
    the finally block below, which would try to close it) and is always closed here.
    """
    start_time = time.monotonic()
    result: OrchestrationResult | None = None
    pipeline = build_resume_enhancement_graph(checkpointer=plan.checkpointer)
    config: RunnableConfig = {"configurable": {"thread_id": plan.run_id}}
    logger.info(
        "pipeline_run_started",
        run_id=plan.run_id,
        resume_path=plan.resume_path,
        jd_path=plan.jd_path,
        **plan.started_log_fields,
    )
    try:
        output = _invoke_pipeline(pipeline, plan.pipeline_input, config, progress_callback)
        result = _finalize_pipeline_output(
            pipeline=pipeline,
            config=config,
            output=output,
            run_id=plan.run_id,
            resume_path=plan.resume_path,
            jd_path=plan.jd_path,
            paused_run_layout=plan.paused_run_layout,
        )
        _log_pipeline_completion(plan.run_id, result, start_time)
        return result
    except Exception:
        logger.exception(
            "pipeline_run_failed",
            run_id=plan.run_id,
            duration_ms=round((time.monotonic() - start_time) * 1000),
        )
        raise
    finally:
        close_checkpoint_database(plan.checkpointer)
        _settle_run_state(plan, result)


def _settle_run_state(plan: _RunPlan, result: OrchestrationResult | None) -> None:
    """Retire or preserve this run's checkpoint and PII mapping.

    The two modes deliberately disagree about failure. A crashed FRESH run has no
    pause anyone could resume, so both are removed. A crashed RESUME leaves its
    paused run intact, so both must survive -- deleting the mapping there would make
    every later resume unable to rehydrate PII.
    """
    if plan.paused_run_layout is not None:
        _settle_resumed_run_checkpoint(plan.paused_run_layout, result)
        needs_pii_cleanup = _should_cleanup_pii_mapping_after_resume(result)
    elif plan.in_flight_checkpoint_db is not None:
        _settle_fresh_run_checkpoint(plan.in_flight_checkpoint_db, result)
        needs_pii_cleanup = _should_cleanup_pii_mapping(result)
    else:
        raise RuntimeError(
            "_RunPlan carries neither an in-flight checkpoint path nor a paused-run layout."
        )
    if needs_pii_cleanup:
        _cleanup_pii_mapping(plan.run_id)


def _invoke_pipeline(
    pipeline: CompiledStateGraph,
    pipeline_input: Any,
    config: RunnableConfig,
    progress_callback: ProgressCallback | None,
) -> dict:
    """Invoke normally or stream task lifecycle events to a caller."""
    if progress_callback is None:
        return cast(dict, pipeline.invoke(pipeline_input, config=config))
    output: dict | None = None
    for mode, event in pipeline.stream(
        pipeline_input, config=config, stream_mode=["tasks", "values"]
    ):
        if mode == "tasks":
            _report_task_event(progress_callback, cast(dict, event))
        else:
            output = cast(dict, event)
    if output is None:
        raise RuntimeError("Pipeline completed without emitting state")
    return output


def _report_task_event(callback: ProgressCallback, event: dict[str, Any]) -> None:
    """Translate one LangGraph task event into a stable UI lifecycle event."""
    if "result" not in event and "error" not in event:
        callback("started", str(event["name"]))
        return
    callback("failed" if event.get("error") else "completed", str(event["name"]))


def _in_flight_checkpoint_db_path(run_id: str) -> Path:
    """Where a fresh run keeps its checkpoint DB until it either pauses or ends.

    A run that pauses has this file moved into its paused-run directory; any
    other outcome deletes it.
    """
    return Path(get_config().file_paths.output_dir) / "checkpoints" / f"{run_id}.sqlite3"


def _settle_fresh_run_checkpoint(
    checkpoint_db_path: Path,
    result: OrchestrationResult | None,
) -> None:
    """Archive the checkpoint DB with its paused run, or delete it for any other outcome.

    Precondition: the checkpointer's connection is closed. A terminal or failed
    fresh run has nothing to resume, so its checkpoint history is removed.
    """
    if result is not None and result.paused_run_path:
        archive_checkpoint_database(
            checkpoint_db_path,
            PausedRunLayout.at(result.paused_run_path),
        )
        return
    checkpoint_db_path.unlink(missing_ok=True)


def _settle_resumed_run_checkpoint(
    layout: PausedRunLayout,
    result: OrchestrationResult | None,
) -> None:
    """Delete the paused run's checkpoint DB only once the run truly completed.

    A failed or re-paused resume keeps the database in place so the candidate
    can fix the sheet (or answer the new questions) and resume again.
    """
    if result is not None and result.paused_run_path is None:
        layout.checkpoint_db.unlink(missing_ok=True)


def _finalize_pipeline_output(
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


def _should_cleanup_pii_mapping(result: OrchestrationResult | None) -> bool:
    """Delete run-local PII state only after a terminal run result.

    A paused run must keep its placeholder mapping alive because rehydrate_pii
    still needs it after the candidate resumes the workflow.
    """
    return result is None or result.disposition is not RunDisposition.NEEDS_CANDIDATE_INPUT


def _should_cleanup_pii_mapping_after_resume(result: OrchestrationResult | None) -> bool:
    """Keep the PII mapping whenever the paused run is still resumable.

    Unlike a fresh run, a failed resume (bad sheet, transient error) leaves the
    paused run intact -- deleting the mapping here would make every later resume
    unable to rehydrate PII. Only a genuinely terminal result cleans up.
    """
    return result is not None and result.disposition is not RunDisposition.NEEDS_CANDIDATE_INPUT


def _log_pipeline_completion(
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


def _cleanup_pii_mapping(run_id: str) -> None:
    """Delete the run-local PII mapping when redaction is enabled."""
    if get_config().feature_flags.enable_pii_redaction:
        delete_pii_mapping(run_id)
