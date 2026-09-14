"""What happens to a run's checkpoint file and PII mapping once the run ends.

runner.py calls settle_run_state() once, in its finally block, after the checkpointer's
connection is already closed (a locked SQLite file cannot be moved or deleted while a
connection to it is still open). Everything else in this file is that one function's
own machinery, pulled out into named, individually testable pieces.

THE RULE WORTH LEARNING, IF YOU ARE BUILDING YOUR OWN RESUMABLE PIPELINE
    A run that ended with something still resumable (a paused run) must KEEP its
    durable state; a run that ended with nothing left to resume must not. Getting
    this backwards either leaks a checkpoint file forever, or deletes a paused run's
    state out from under a candidate who has not answered the questions yet.
    settle_run_state() below is where that rule gets applied; the two
    should_cleanup_pii_mapping* functions are where it is spelled out as a plain
    True/False decision, one case at a time.
"""

from pathlib import Path

from src.core.logger import get_logger
from src.core.pii_mapping_store import delete_pii_mapping
from src.core.settings import get_config
from src.data_models.orchestration import OrchestrationResult, RunDisposition
from src.hitl.professional_experience.persistence import (
    PausedRunLayout,
    archive_checkpoint_database,
)

logger = get_logger(__name__)


def settle_run_state(
    run_id: str,
    result: OrchestrationResult | None,
    *,
    in_flight_checkpoint_db: Path | None,
    paused_run_layout: PausedRunLayout | None,
) -> None:
    """Retire or preserve this run's checkpoint and PII mapping, then act on it.

    Exactly one of the two keyword arguments is set, and that is what tells this
    function which of the two run modes it is settling:
      * in_flight_checkpoint_db -- a fresh run. Its checkpoint sits in a scratch
        directory and moves into a paused-run folder, or is deleted, when the run
        ends (see _settle_fresh_run_checkpoint).
      * paused_run_layout -- a resume. Its checkpoint already lives in the
        paused-run folder, and stays there unless the run truly finished (see
        _settle_resumed_run_checkpoint).
    """
    if paused_run_layout is not None:
        _settle_resumed_run_checkpoint(paused_run_layout, result)
        needs_pii_cleanup = _should_cleanup_pii_mapping_after_resume(result)
    elif in_flight_checkpoint_db is not None:
        _settle_fresh_run_checkpoint(in_flight_checkpoint_db, result)
        needs_pii_cleanup = _should_cleanup_pii_mapping(result)
    else:
        raise RuntimeError(
            "settle_run_state got neither an in-flight checkpoint path nor a "
            "paused-run layout -- the caller built its run description wrong."
        )
    if needs_pii_cleanup:
        _cleanup_pii_mapping(run_id)


def in_flight_checkpoint_db_path(run_id: str) -> Path:
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


def _cleanup_pii_mapping(run_id: str) -> None:
    """Delete the run-local PII mapping when redaction is enabled."""
    if get_config().feature_flags.enable_pii_redaction:
        delete_pii_mapping(run_id)
