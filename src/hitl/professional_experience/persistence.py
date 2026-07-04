"""File-backed persistence for paused professional-experience clarification runs.

A paused run is one self-contained directory:
  clarifications_sheet.json   the questions the candidate edits
  paused_run_manifest.json    run identity + filenames needed to resume
  checkpoints.sqlite3         the LangGraph checkpoint history (SqliteSaver)

The checkpoint database is written by LangGraph's own SqliteSaver -- the library's
supported durable checkpointer -- so no LangGraph internals are serialized here.
The runner opens the database at run start, and this module moves it into the
paused-run directory when a run pauses (see archive_checkpoint_database).
"""

import json
import shutil
import sqlite3
from pathlib import Path

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

from src.core.logger import get_logger
from src.hitl.professional_experience.checkpoint_types import (
    CHECKPOINT_ALLOWED_MSGPACK_MODULES,
)
from src.hitl.professional_experience.models import (
    ExperienceBulletClarification,
    ExperienceClarificationPausedRunManifest,
)

logger = get_logger(__name__)

MANIFEST_FILENAME = "paused_run_manifest.json"
# Must match the ExperienceClarificationPausedRunManifest.checkpoint_db_filename default.
CHECKPOINT_DB_FILENAME = "checkpoints.sqlite3"


def save_clarification_sheet(
    output_dir: Path,
    clarifications: list[ExperienceBulletClarification],
) -> str | None:
    """Write the candidate's editable clarification sheet into output_dir."""
    if not clarifications:
        return None
    sheet = {
        "_instructions": (
            "Answer the questions below in your own words using only real facts from "
            "your work. Add systems, outcomes, users, or scale only when they are "
            "true. Then resume the paused run from this folder."
        ),
        "clarifications": [clarification.model_dump(mode="json") for clarification in clarifications],
    }
    path = output_dir / "clarifications_sheet.json"
    path.write_text(json.dumps(sheet, indent=2), encoding="utf-8")
    logger.info(
        "clarification_sheet_saved",
        path=str(path),
        questions=len(clarifications),
    )
    return str(path)


def load_answered_clarifications(
    clarifications_path: str | None,
) -> list[ExperienceBulletClarification]:
    """Load answered clarifications from a clarification sheet path."""
    if clarifications_path is None:
        return []
    sheet = json.loads(Path(clarifications_path).read_text(encoding="utf-8"))
    clarifications = [
        ExperienceBulletClarification.model_validate(entry) for entry in sheet["clarifications"]
    ]
    return [
        clarification for clarification in clarifications if clarification.answer.strip()
    ]


def open_checkpoint_database(db_path: Path) -> SqliteSaver:
    """Open (creating if needed) the SQLite-backed LangGraph checkpointer for one run.

    The connection allows cross-thread use because graph nodes run in worker
    threads; SqliteSaver serializes its own writes internally. Close it with
    close_checkpoint_database when the run ends.

    The serde carries an explicit allowlist of every pipeline model and enum
    (see checkpoint_types.py) so a paused run can resume without the
    "unregistered type" warning today, and without a hard failure once
    LangGraph enforces LANGGRAPH_STRICT_MSGPACK by default.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(db_path), check_same_thread=False)
    serde = JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_ALLOWED_MSGPACK_MODULES)
    return SqliteSaver(connection, serde=serde)


def close_checkpoint_database(checkpointer: SqliteSaver) -> None:
    """Close the checkpointer's SQLite connection so its file can be moved or deleted."""
    checkpointer.conn.close()


def save_paused_run_state(
    paused_run_dir: Path,
    manifest: ExperienceClarificationPausedRunManifest,
    clarifications: list[ExperienceBulletClarification],
) -> str:
    """Write the clarification sheet and manifest of a paused run into its directory.

    The checkpoint database is not written here -- it is produced by SqliteSaver
    during the run and moved in by archive_checkpoint_database once the runner
    has closed it.
    """
    paused_run_dir.mkdir(parents=True, exist_ok=True)
    save_clarification_sheet(paused_run_dir, clarifications)
    (paused_run_dir / MANIFEST_FILENAME).write_text(
        manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )
    logger.info(
        "paused_run_state_saved",
        paused_run_dir=str(paused_run_dir),
        questions=len(clarifications),
    )
    return str(paused_run_dir)


def archive_checkpoint_database(
    db_path: Path,
    paused_run_dir: Path,
) -> None:
    """Move a closed checkpoint database into the paused-run directory.

    No-op when the database already lives there (a run that paused a second time
    after a resume). Precondition: the checkpointer's connection is closed.
    """
    target = paused_run_dir / CHECKPOINT_DB_FILENAME
    if db_path == target:
        return
    paused_run_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(db_path), str(target))
    logger.info("checkpoint_database_archived", path=str(target))


def load_paused_run_state(
    paused_run_path: str,
) -> tuple[ExperienceClarificationPausedRunManifest, SqliteSaver]:
    """Load the manifest and reopen the checkpoint database of a paused run."""
    paused_run_dir = Path(paused_run_path)
    manifest = ExperienceClarificationPausedRunManifest.model_validate_json(
        (paused_run_dir / MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    db_path = paused_run_dir / manifest.checkpoint_db_filename
    if not db_path.is_file():
        raise FileNotFoundError(
            f"Paused run at {paused_run_path} has no checkpoint database "
            f"({manifest.checkpoint_db_filename}); it cannot be resumed."
        )
    return manifest, open_checkpoint_database(db_path)
