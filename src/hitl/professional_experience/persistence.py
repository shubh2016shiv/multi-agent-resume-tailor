"""File-backed persistence for paused professional-experience clarification runs.

WHAT A PAUSED RUN LOOKS LIKE ON DISK
-----------------------------------
One self-contained directory, `paused_run_<run_id>/`, holding four files:

    clarifications_sheet.json   the questions -- the candidate writes answers here (MUTABLE)
    paused_run_manifest.json    run identity + the deadline to answer by
    answers_audit.jsonl         every answer ever accepted, one per line (APPEND-ONLY)
    checkpoints.sqlite3         the full LangGraph state (written by the runner, not here)

WHO CALLS WHAT (the two directions across the pause)
---------------------------------------------------
    PAUSING  ->  finalize_pipeline_output()  [orchestration/run_results.py]
                   calls save_paused_run_state(layout, manifest, clarifications)
                     -> write_clarification_sheet()   writes the sheet
                     -> (manifest written inline)

    ANSWERING -> web_app._save_clarification_answers() [web_app/server.py]
              -> (or the candidate edits the sheet by hand for the CLI flow)
                   calls record_clarification_answers(layout, {bullet_id: text}, answered_by=...)
                     -> read_clarification_sheet()     load current questions
                     -> append_answer_records()        write audit FIRST
                     -> write_clarification_sheet()     then the mutable sheet

    RESUMING  ->  runner.resume_paused_run()          [orchestration/runner.py]
                   calls load_paused_run(path)          -> (layout, manifest); refuses if unresumable
                   calls read_answered_clarifications() -> the answers to feed back in

THE ONE RULE THIS MODULE ENFORCES
---------------------------------
PausedRunLayout is the ONLY code that knows the four file names, and the sheet
functions below are the ONLY code that (de)serializes the sheet. This is not
style. The one time the sheet's shape was known in two places -- written here,
hand-parsed in the web layer -- the two disagreed: the sheet is an OBJECT with a
"clarifications" key, the web layer indexed it like a LIST, and every browser
submission returned HTTP 500 while the CLI path worked fine. A single accessor
makes that class of bug unrepresentable.
"""

import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from src.core.logger import get_logger
from src.hitl.professional_experience.models import (
    ClarificationAnswerRecord,
    ExperienceBulletClarification,
    ExperienceClarificationPausedRunManifest,
)

logger = get_logger(__name__)

# The four file names. Referenced ONLY through PausedRunLayout below -- no caller
# ever writes these literals, so the directory shape has exactly one definition.
SHEET_FILENAME = "clarifications_sheet.json"
MANIFEST_FILENAME = "paused_run_manifest.json"
AUDIT_FILENAME = "answers_audit.jsonl"
CHECKPOINT_DB_FILENAME = "checkpoints.sqlite3"

# Shown at the top of the sheet. Matters because the CLI flow is a human opening
# this JSON in an editor -- the file has to explain itself.
SHEET_INSTRUCTIONS = (
    "Answer the questions below in your own words using only real facts from your "
    "work. Add systems, outcomes, users, or scale only when they are true. Leave "
    "any question blank to skip it, then resume this paused run from this folder."
)


# =============================================================================
# PausedRunLayout  --  the single source of truth for the directory's shape
# -----------------------------------------------------------------------------
# A frozen dataclass wrapping one Path. Every other function in this module, and
# the runner and web app, take a PausedRunLayout and ask it for a file path
# (layout.sheet, layout.checkpoint_db, ...) instead of composing one. That is
# how "no caller hardcodes a filename" is guaranteed rather than hoped for.
#
# CONSTRUCTED BY : PausedRunLayout.at(path) -- used by runner.py and web_app/server.py
# =============================================================================
@dataclass(frozen=True)
class PausedRunLayout:
    """The file layout of one paused-run directory.

    Names live here once so a resume never has to guess the directory's shape,
    and so no caller can drift from it by hardcoding a literal.
    """

    root: Path

    @classmethod
    def at(cls, paused_run_path: str | Path) -> "PausedRunLayout":
        """Return the layout for an existing or about-to-be-created paused run."""
        return cls(root=Path(paused_run_path))

    # Each property is `root / <one filename constant>`. Nothing more.
    @property
    def sheet(self) -> Path:
        return self.root / SHEET_FILENAME

    @property
    def manifest(self) -> Path:
        return self.root / MANIFEST_FILENAME

    @property
    def audit_log(self) -> Path:
        return self.root / AUDIT_FILENAME

    @property
    def checkpoint_db(self) -> Path:
        return self.root / CHECKPOINT_DB_FILENAME


# =============================================================================
# write_clarification_sheet()  --  serialise questions -> clarifications_sheet.json
# -----------------------------------------------------------------------------
# The ONLY function that writes the sheet. HITL Component 5 (file half); the web
# layer is the other half and calls record_clarification_answers(), never this.
#
# CALLED BY : save_paused_run_state()          (this file, on pause)
#             record_clarification_answers()   (this file, after recording answers)
# RETURNS   : the sheet's path as a string (for logging/CLI display)
# =============================================================================
def write_clarification_sheet(
    layout: PausedRunLayout,
    clarifications: list[ExperienceBulletClarification],
) -> str:
    """Write the candidate's editable clarification sheet.

    The sheet stays human-editable on purpose: the CLI flow (`--resume-from`) is
    a candidate opening this file in an editor, so it carries its own instructions.
    """
    # STEP 1: ensure the directory exists (first pause of a run creates it).
    layout.root.mkdir(parents=True, exist_ok=True)

    # STEP 2: build the on-disk shape. THIS IS THE CONTRACT the web bug violated:
    #   top level is an OBJECT -- "_instructions" (human help) + "clarifications"
    #   (the list). Anyone treating the file as a bare list is reading it wrong.
    sheet = {
        "_instructions": SHEET_INSTRUCTIONS,
        "clarifications": [
            clarification.model_dump(mode="json") for clarification in clarifications
        ],
    }

    # STEP 3: write it, pretty-printed (a human may read it), then log.
    layout.sheet.write_text(json.dumps(sheet, indent=2), encoding="utf-8")
    logger.info(
        "clarification_sheet_saved",
        path=str(layout.sheet),
        questions=len(clarifications),
    )
    return str(layout.sheet)


# =============================================================================
# read_clarification_sheet()  --  deserialise clarifications_sheet.json -> objects
# -----------------------------------------------------------------------------
# The ONLY function that reads the sheet. Reads sheet["clarifications"] -- the
# KEY, not a positional index -- which is the fix for the HTTP 500 described in
# the module docstring.
#
# CALLED BY : read_answered_clarifications()      (this file)
#             record_clarification_answers()      (this file)
# RAISES    : FileNotFoundError if the sheet is missing (nothing to answer)
# =============================================================================
def read_clarification_sheet(
    layout: PausedRunLayout,
) -> list[ExperienceBulletClarification]:
    """Read every question in the sheet, answered or not."""
    if not layout.sheet.is_file():
        raise FileNotFoundError(
            f"Paused run at {layout.root} has no {SHEET_FILENAME}; there is nothing to answer."
        )
    # sheet is {"_instructions": ..., "clarifications": [...]} -- index the KEY.
    sheet = json.loads(layout.sheet.read_text(encoding="utf-8"))
    # model_validate rebuilds each dict into a typed object, re-checking the schema.
    return [
        ExperienceBulletClarification.model_validate(entry) for entry in sheet["clarifications"]
    ]


# =============================================================================
# read_answered_clarifications()  --  the subset the resume actually needs
# -----------------------------------------------------------------------------
# CALLED BY : runner.resume_paused_run()  [orchestration/runner.py] -- if this
#             returns empty, the resume is refused ("answer at least one first").
# =============================================================================
def read_answered_clarifications(
    layout: PausedRunLayout,
) -> list[ExperienceBulletClarification]:
    """Read only the questions the candidate actually answered."""
    # is_answered is `bool(self.answer.strip())` -- see models.py. Whitespace is
    # not an answer, so a sheet touched but not filled resumes to nothing.
    return [
        clarification
        for clarification in read_clarification_sheet(layout)
        if clarification.is_answered
    ]


# =============================================================================
# record_clarification_answers()  --  the write path for BOTH surfaces
# -----------------------------------------------------------------------------
# Takes {bullet_id: answer_text}. Both the web endpoint and (conceptually) the
# CLI edit funnel through here. Does three things atomically-ish: validates the
# ids, writes the immutable audit log, then rewrites the mutable sheet.
#
# CALLED BY : web_app._save_clarification_answers()  [web_app/server.py]
# CALLS     : read_clarification_sheet(), append_answer_records(), write_clarification_sheet()
# RAISES    : ValueError on an unknown bullet_id, or if no non-blank answer given
# RETURNS   : the full updated question list (answered + still-blank)
# =============================================================================
def record_clarification_answers(
    layout: PausedRunLayout,
    answers: Mapping[str, str],  # {bullet_id -> answer text}; NEVER positional
    *,
    answered_by: str,  # honest label of the submitting surface
) -> list[ExperienceBulletClarification]:
    """Record candidate answers into the sheet and the audit log.

    Answers are keyed by bullet_id, never by position. A positional key would
    couple whatever renders the questions to whatever wrote the file, with
    nothing keeping the two orderings in step; bullet_id is already the stable
    identity used to route an answer back to its role after the resume.

    Previously answered questions are preserved, so a candidate may answer in
    several sittings. Blank answers are ignored rather than recorded as answers.
    """
    # ---- STEP 1: load current state and reject unknown ids -----------------
    # An answer naming a bullet_id not in the sheet is a caller bug (wrong run,
    # stale UI). Fail loudly rather than silently dropping it.
    clarifications = read_clarification_sheet(layout)
    known_bullet_ids = {clarification.bullet_id for clarification in clarifications}
    unknown_bullet_ids = sorted(set(answers) - known_bullet_ids)
    if unknown_bullet_ids:
        raise ValueError(f"No such question(s) in this paused run: {unknown_bullet_ids}.")

    # ---- STEP 2: build the updated list + the new audit entries -----------
    # Walk every EXISTING question (so prior answers are preserved), and for each
    # one that has a fresh non-blank answer: stamp it and queue an audit record.
    recorded_at = datetime.now(UTC)
    updated: list[ExperienceBulletClarification] = []
    new_records: list[ClarificationAnswerRecord] = []
    for clarification in clarifications:
        answer = (answers.get(clarification.bullet_id) or "").strip()
        if not answer:
            updated.append(clarification)  # keep as-is (blank now, or answered earlier)
            continue
        updated.append(
            clarification.model_copy(
                update={
                    "answer": answer,
                    "answered_at": recorded_at,
                    "answered_by": answered_by,
                }
            )
        )
        new_records.append(
            ClarificationAnswerRecord(
                bullet_id=clarification.bullet_id,
                answer=answer,
                answered_by=answered_by,
                answered_at=recorded_at,
            )
        )

    # ---- STEP 3: refuse an empty submission -------------------------------
    # Resuming with zero answers would burn an LLM pass for nothing.
    if not any(clarification.is_answered for clarification in updated):
        raise ValueError("Answer at least one clarification before continuing.")

    # ---- STEP 4: persist -- AUDIT FIRST, then the working copy ------------
    # Ordering is deliberate: if the process dies between these two lines, the
    # record of what the candidate submitted still exists. The reverse ordering
    # could lose the audit while keeping the mutation.
    append_answer_records(layout, new_records)
    write_clarification_sheet(layout, updated)

    logger.info(
        "clarification_answers_recorded",
        paused_run_path=str(layout.root),
        answered_now=len(new_records),
        answered_total=sum(1 for c in updated if c.is_answered),
    )
    return updated


# =============================================================================
# append_answer_records()  --  write to the immutable audit log
# -----------------------------------------------------------------------------
# JSON Lines, opened in APPEND mode, never rewritten. The sheet beside it is the
# mutable working copy; this file is the trustworthy history -- correcting an
# answer leaves BOTH entries here.
#
# CALLED BY : record_clarification_answers()  (STEP 4)
# =============================================================================
def append_answer_records(
    layout: PausedRunLayout,
    records: list[ClarificationAnswerRecord],
) -> None:
    """Append answers to the immutable audit log."""
    if not records:
        return
    layout.root.mkdir(parents=True, exist_ok=True)
    # "a" = append. One JSON object per line. No line is ever modified or removed.
    with layout.audit_log.open("a", encoding="utf-8") as audit_file:
        for record in records:
            audit_file.write(record.model_dump_json() + "\n")


# =============================================================================
# save_paused_run_state()  --  called at the moment the pipeline pauses
# -----------------------------------------------------------------------------
# Writes the sheet and the manifest into a (possibly new) paused-run directory.
# Does NOT write checkpoints.sqlite3 -- the runner's checkpointer owns that file
# while the graph runs, and archive_checkpoint_database() moves it in afterward.
#
# CALLED BY : finalize_pipeline_output()  [orchestration/run_results.py],
#             when output contains "__interrupt__".
# CALLS     : write_clarification_sheet()
# RETURNS   : the paused-run directory path as a string
# =============================================================================
def save_paused_run_state(
    layout: PausedRunLayout,
    manifest: ExperienceClarificationPausedRunManifest,
    clarifications: list[ExperienceBulletClarification],
) -> str:
    """Write the sheet and manifest of a newly paused run."""
    layout.root.mkdir(parents=True, exist_ok=True)
    write_clarification_sheet(layout, clarifications)  # the questions
    layout.manifest.write_text(  # identity + deadline
        manifest.model_dump_json(indent=2), encoding="utf-8"
    )
    logger.info(
        "paused_run_state_saved",
        paused_run_dir=str(layout.root),
        questions=len(clarifications),
        expires_at=manifest.expires_at.isoformat(),
    )
    return str(layout.root)


# =============================================================================
# load_paused_run()  --  the guarded entry to a resume
# -----------------------------------------------------------------------------
# Loads the manifest and verifies the directory holds the TWO things a resume
# cannot proceed without: the manifest (identifies the LangGraph thread) and the
# checkpoint db (where that thread's state lives). Raises BEFORE the runner opens
# a database connection or compiles a graph -- failures stay cheap and legible.
#
# CALLED BY : runner.resume_paused_run()  [orchestration/runner.py], first line.
# RETURNS   : (layout, manifest). The caller then checks manifest.is_expired.
# RAISES    : FileNotFoundError if manifest or checkpoint db is missing
# =============================================================================
def load_paused_run(
    paused_run_path: str,
) -> tuple[PausedRunLayout, ExperienceClarificationPausedRunManifest]:
    """Load one paused run's layout and manifest, verifying it can still be resumed."""
    layout = PausedRunLayout.at(paused_run_path)

    # CHECK 1: the manifest exists.
    if not layout.manifest.is_file():
        raise FileNotFoundError(
            f"Paused run at {paused_run_path} has no {MANIFEST_FILENAME}; it cannot be resumed."
        )
    manifest = ExperienceClarificationPausedRunManifest.model_validate_json(
        layout.manifest.read_text(encoding="utf-8")
    )

    # CHECK 2: the checkpoint history exists. Without it there is no thread to
    # resume, only a manifest pointing at nothing.
    if not layout.checkpoint_db.is_file():
        raise FileNotFoundError(
            f"Paused run at {paused_run_path} has no {CHECKPOINT_DB_FILENAME}; "
            "it cannot be resumed."
        )

    return layout, manifest


# =============================================================================
# archive_checkpoint_database()  --  move the finished-writing db into the dir
# -----------------------------------------------------------------------------
# During a fresh run the checkpointer writes to .../checkpoints/<run_id>.sqlite3.
# When that run pauses, the runner closes the connection and calls this to move
# the file into paused_run_<id>/checkpoints.sqlite3, so the paused run is fully
# self-contained.
#
# CALLED BY : runner._settle_fresh_run_checkpoint()  [orchestration/runner.py]
# PRECONDITION : the SqliteSaver connection is already closed (else the move fails
#                on Windows / leaves a locked handle).
# =============================================================================
def archive_checkpoint_database(db_path: Path, layout: PausedRunLayout) -> None:
    """Move a closed checkpoint database into its paused-run directory."""
    # No-op if it is already there -- a run that paused a second time after a
    # resume already has its db inside the directory.
    if db_path == layout.checkpoint_db:
        return
    layout.root.mkdir(parents=True, exist_ok=True)
    shutil.move(str(db_path), str(layout.checkpoint_db))
    logger.info("checkpoint_database_archived", path=str(layout.checkpoint_db))
