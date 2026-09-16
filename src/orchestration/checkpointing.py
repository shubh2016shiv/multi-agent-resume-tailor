"""Open, close, preserve, and retire LangGraph checkpoint databases."""

import sqlite3
from pathlib import Path

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

from src.core.settings import get_config
from src.data_models.orchestration import OrchestrationResult
from src.hitl.professional_experience.persistence import (
    PausedRunLayout,
    archive_checkpoint_database,
)
from src.orchestration.checkpoint_allowlist import CHECKPOINT_ALLOWED_MSGPACK_MODULES


def open_checkpoint_database(db_path: Path) -> SqliteSaver:
    """Open the checkpoint file for one run, creating it and its directory if needed.

    check_same_thread=False because the graph runs nodes in worker threads, so the
    connection is used from a different thread than the one that opened it. That is
    safe here: SqliteSaver serializes its own writes.

    Pair every call with close_checkpoint_database. The file cannot be moved or
    deleted on Windows while the connection is open.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(db_path), check_same_thread=False)
    # serde = the serializer LangGraph uses to turn state into bytes and back. Handing
    # it an explicit allowlist is what restricts which classes a load may rebuild.
    serde = JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_ALLOWED_MSGPACK_MODULES)
    return SqliteSaver(connection, serde=serde)


def close_checkpoint_database(checkpointer: SqliteSaver) -> None:
    """Close the connection, releasing the file so it can be moved or deleted."""
    checkpointer.conn.close()


def checkpoint_path_for_run(run_id: str) -> Path:
    """Return the temporary checkpoint path for a new pipeline run."""
    output_dir = Path(get_config().file_paths.output_dir)
    return output_dir / "checkpoints" / f"{run_id}.sqlite3"


def settle_checkpoint(
    result: OrchestrationResult | None,
    *,
    in_flight_path: Path | None,
    paused_layout: PausedRunLayout | None,
) -> None:
    """Archive a paused checkpoint or delete one that is no longer resumable."""
    if paused_layout is not None:
        _settle_resumed_checkpoint(paused_layout, result)
    elif in_flight_path is not None:
        _settle_fresh_checkpoint(in_flight_path, result)
    else:
        raise RuntimeError("A checkpoint path or paused-run layout is required.")


def _settle_fresh_checkpoint(
    checkpoint_path: Path,
    result: OrchestrationResult | None,
) -> None:
    """Archive a paused run's checkpoint; otherwise delete it."""
    if result is not None and result.paused_run_path:
        archive_checkpoint_database(
            checkpoint_path,
            PausedRunLayout.at(result.paused_run_path),
        )
        return
    checkpoint_path.unlink(missing_ok=True)


def _settle_resumed_checkpoint(
    layout: PausedRunLayout,
    result: OrchestrationResult | None,
) -> None:
    """Delete a resumed checkpoint only after the run completes."""
    if result is not None and result.paused_run_path is None:
        layout.checkpoint_db.unlink(missing_ok=True)
