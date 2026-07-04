"""Public entry points for the resume enhancement pipeline."""

import time
from datetime import datetime
from pathlib import Path
from typing import cast
from uuid import uuid4

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
    archive_checkpoint_database,
    close_checkpoint_database,
    load_answered_clarifications,
    load_paused_run_state,
    open_checkpoint_database,
    save_clarification_sheet,
    save_paused_run_state,
)
from src.observability import init_observability
from src.orchestration.graph import build_resume_enhancement_graph
from src.orchestration.human_review_policy import derive_run_disposition
from src.orchestration.state import ResumeEnhancementPipelineState
from src.tools.engines.document_rendering.output_paths import resume_output_dir

logger = get_logger(__name__)

init_observability("resume-tailor-agents")


def tailor_resume(
    resume_path: str,
    jd_path: str,
) -> OrchestrationResult:
    """Run a fresh pipeline execution from source documents."""
    run_id = uuid4().hex
    start_time = time.monotonic()
    result: OrchestrationResult | None = None
    checkpoint_db_path = _in_flight_checkpoint_db_path(run_id)
    checkpointer = open_checkpoint_database(checkpoint_db_path)
    pipeline = _build_pipeline(checkpointer)
    config = {"configurable": {"thread_id": run_id}}
    logger.info(
        "pipeline_run_started",
        run_id=run_id,
        resume_path=resume_path,
        jd_path=jd_path,
        resume_mode="fresh",
    )
    initial_state: ResumeEnhancementPipelineState = {
        "run_id": run_id,
        "resume_path": resume_path,
        "jd_path": jd_path,
        "clarification_answers": [],
        "resume": None,
        "job_description": None,
        "requirement_match_report": None,
        "alignment_strategy": None,
        "professional_summary": None,
        "optimized_experience": None,
        "experience_clarifications": None,
        "optimized_skills": None,
        "optimized_resume": None,
        "quality_report": None,
        "rendered_structure_evaluation": None,
        "human_review_required": False,
        "rendered_artifacts": None,
    }
    try:
        output = pipeline.invoke(initial_state, config=config)
        result = _finalize_pipeline_output(
            pipeline=pipeline,
            config=config,
            output=output,
            run_id=run_id,
            resume_path=resume_path,
            jd_path=jd_path,
        )
        _log_pipeline_completion(run_id, result, start_time)
        return result
    except Exception:
        duration_ms = round((time.monotonic() - start_time) * 1000)
        logger.exception(
            "pipeline_run_failed",
            run_id=run_id,
            duration_ms=duration_ms,
        )
        raise
    finally:
        close_checkpoint_database(checkpointer)
        _settle_fresh_run_checkpoint(checkpoint_db_path, result)
        if _should_cleanup_pii_mapping(result):
            _cleanup_pii_mapping(run_id)


def resume_paused_run(
    paused_run_path: str,
) -> OrchestrationResult:
    """Resume a previously paused professional-experience clarification run."""
    paused_run_dir = Path(paused_run_path)
    manifest, checkpointer = load_paused_run_state(paused_run_path)

    start_time = time.monotonic()
    result: OrchestrationResult | None = None
    pipeline = _build_pipeline(checkpointer)
    config = {"configurable": {"thread_id": manifest.run_id}}
    try:
        answered_clarifications = load_answered_clarifications(
            str(paused_run_dir / manifest.clarifications_filename)
        )
        if not answered_clarifications:
            raise ValueError(
                "clarifications_sheet.json has no answered questions yet; answer at least "
                "one clarification before resuming the paused run."
            )
        logger.info(
            "pipeline_run_started",
            run_id=manifest.run_id,
            resume_path=manifest.resume_path,
            jd_path=manifest.jd_path,
            resume_mode="paused_run_resume",
            paused_run_path=paused_run_path,
            answered_clarifications=len(answered_clarifications),
        )
        output = pipeline.invoke(
            Command(
                resume={"status": "candidate_answers_submitted"},
                update={"clarification_answers": answered_clarifications},
            ),
            config=config,
        )
        result = _finalize_pipeline_output(
            pipeline=pipeline,
            config=config,
            output=output,
            run_id=manifest.run_id,
            resume_path=manifest.resume_path,
            jd_path=manifest.jd_path,
            paused_run_path=paused_run_path,
        )
        _log_pipeline_completion(manifest.run_id, result, start_time)
        return result
    except Exception:
        duration_ms = round((time.monotonic() - start_time) * 1000)
        logger.exception(
            "pipeline_run_failed",
            run_id=manifest.run_id,
            duration_ms=duration_ms,
        )
        raise
    finally:
        close_checkpoint_database(checkpointer)
        _settle_resumed_run_checkpoint(
            paused_run_dir / manifest.checkpoint_db_filename, result
        )
        if _should_cleanup_pii_mapping_after_resume(result):
            _cleanup_pii_mapping(manifest.run_id)


def _build_pipeline(checkpointer: SqliteSaver) -> CompiledStateGraph:
    """Compile a pipeline graph bound to one run-local checkpointer."""
    return build_resume_enhancement_graph(checkpointer=checkpointer)


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
        archive_checkpoint_database(checkpoint_db_path, Path(result.paused_run_path))
        return
    checkpoint_db_path.unlink(missing_ok=True)


def _settle_resumed_run_checkpoint(
    checkpoint_db_path: Path,
    result: OrchestrationResult | None,
) -> None:
    """Delete the paused run's checkpoint DB only once the run truly completed.

    A failed or re-paused resume keeps the database in place so the candidate
    can fix the sheet (or answer the new questions) and resume again.
    """
    if result is not None and result.paused_run_path is None:
        checkpoint_db_path.unlink(missing_ok=True)


def _finalize_pipeline_output(
    pipeline: CompiledStateGraph,
    config: dict,
    output: dict,
    run_id: str,
    resume_path: str,
    jd_path: str,
    paused_run_path: str | None = None,
) -> OrchestrationResult:
    """Convert a fresh invoke() result into a paused or completed orchestration result."""
    if _pipeline_interrupted(output):
        snapshot = pipeline.get_state(config)
        snapshot_state = cast(ResumeEnhancementPipelineState, snapshot.values)
        effective_paused_run_path = paused_run_path or _paused_run_directory(
            snapshot_state,
            run_id,
        )
        manifest = ExperienceClarificationPausedRunManifest(
            run_id=run_id,
            resume_path=resume_path,
            jd_path=jd_path,
        )
        save_paused_run_state(
            Path(effective_paused_run_path),
            manifest,
            snapshot_state.get("experience_clarifications") or [],
        )
        result = _build_paused_orchestration_result(
            snapshot_state,
            paused_run_path=effective_paused_run_path,
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
    assert state["resume"] is not None, "Paused run is missing the parsed resume"
    assert state["job_description"] is not None, "Paused run is missing the parsed job description"
    assert state["alignment_strategy"] is not None, "Paused run is missing the alignment strategy"
    clarifications_requested = state.get("experience_clarifications") or []
    return OrchestrationResult(
        original_resume=state["resume"],
        job_description=state["job_description"],
        strategy=state["alignment_strategy"],
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
    assert state["resume"] is not None, "Pipeline completed but 'resume' is still None"
    assert state["job_description"] is not None, (
        "Pipeline completed but 'job_description' is still None"
    )
    assert state["alignment_strategy"] is not None, (
        "Pipeline completed but 'alignment_strategy' is still None"
    )
    assert state["optimized_resume"] is not None, (
        "Pipeline completed but 'optimized_resume' is still None"
    )
    assert state["quality_report"] is not None, (
        "Pipeline completed but 'quality_report' is still None"
    )
    clarifications_requested = state.get("experience_clarifications") or []
    return OrchestrationResult(
        original_resume=state["resume"],
        job_description=state["job_description"],
        strategy=state["alignment_strategy"],
        optimized_resume=state["optimized_resume"],
        quality_report=state["quality_report"],
        rendered_artifacts=state["rendered_artifacts"],
        clarifications_requested=clarifications_requested,
        disposition=derive_run_disposition(
            human_review_required=state["human_review_required"],
            quality_gate_passed=state["quality_report"].passes_quality_gate,
            has_candidate_questions=bool(clarifications_requested),
        ),
        paused_run_path=None,
    )


def _persist_result(result: OrchestrationResult) -> str:
    """Persist one orchestration result next to the paused run or rendered artifacts."""
    output_dir = _result_output_dir(result)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    logger.info(
        "run_result_saved",
        path=str(path),
        disposition=result.disposition.value,
    )
    if result.clarifications_requested:
        save_clarification_sheet(output_dir, result.clarifications_requested)
    return str(path)


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
) -> str:
    """Return the local folder where one paused clarification run should live."""
    assert state["resume"] is not None, "Paused run is missing the parsed resume"
    assert state["job_description"] is not None, "Paused run is missing the parsed job description"
    base_dir = Path(get_config().file_paths.output_dir)
    parent_dir = resume_output_dir(state["resume"], state["job_description"], base_dir)
    return str(parent_dir / f"paused_run_{run_id}")


def _pipeline_interrupted(output: dict) -> bool:
    """Return whether LangGraph paused the run on an interrupt boundary."""
    return "__interrupt__" in output


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
            result.quality_report.passes_quality_gate
            if result.quality_report is not None
            else None
        ),
        disposition=result.disposition.value,
        duration_ms=duration_ms,
    )


def _cleanup_pii_mapping(run_id: str) -> None:
    """Delete the run-local PII mapping when redaction is enabled."""
    if get_config().feature_flags.enable_pii_redaction:
        delete_pii_mapping(run_id)
