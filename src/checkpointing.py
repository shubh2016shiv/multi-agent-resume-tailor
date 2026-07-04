"""
Debug checkpoint writer for LLM agent input/output.

Saves the exact task_description each agent receives just before the CrewAI
kickoff() call, and the raw + validated result right after, so you can inspect
exactly what an agent's LLM call saw and returned.

Usage: enabled via env var DEBUG_CHECKPOINTS=1.

Layout — ONE timestamped folder per pipeline run, keyed by run_id:

    checkpoints/
      2026-07-04_18-14-42_c1cd7350/          <- one folder for the whole run
        01_job_description_extractor__INPUT.txt
        01_job_description_extractor__OUTPUT.txt
        02_resume_to_job_alignment_strategist__INPUT.txt
        02_resume_to_job_alignment_strategist__OUTPUT.txt
        03_career_narrative_specialist__INPUT.txt
        03_career_narrative_specialist__OUTPUT.txt
        ...

- Every LLM call gets its own INPUT/OUTPUT file pair, so input vs output is
  obvious from the filename alone.
- A per-run sequence number prefixes each pair, so calls are ordered and the
  same agent invoked several times (e.g. the per-role experience rewrites) never
  overwrites itself.
- The agent's role is cleaned for the filename: the parenthetical qualifier
  (e.g. "(domain-agnostic, source-faithful)") is dropped.

This module has zero production-path side-effects when disabled. It writes
synchronously (input before kickoff, output after) so the INPUT file is always
present even when the agent crashes mid-call.
"""

import os
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Module-level flag — checked once per call, no import-time overhead.
_ENABLED: bool | None = None  # None = not yet resolved
_WRITE_LOCK = threading.Lock()  # guards the run-dir registry and per-run counters

_CHECKPOINT_ROOT = Path("checkpoints")

# run_id -> the single timestamped folder that holds every checkpoint for that run.
_RUN_DIRS: dict[str, Path] = {}
# run_id -> next sequence number to prefix the next call's file pair.
_RUN_SEQ: dict[str, int] = {}


def _is_enabled() -> bool:
    """Resolve and cache whether checkpoint writing is active.

    Priority: DEBUG_CHECKPOINTS env var > default (disabled).
    Cached after first call so the env lookup is not repeated per agent call.
    """
    global _ENABLED
    if _ENABLED is None:
        _ENABLED = os.environ.get("DEBUG_CHECKPOINTS", "0").strip() in {"1", "true", "yes"}
    return _ENABLED


def _clean_agent_name(agent_role: str) -> str:
    """Turn an agent role into a clean, filesystem-safe slug.

    Drops the parenthetical qualifier so "Professional Summary Writer
    (domain-agnostic)" becomes "professional_summary_writer" rather than
    "professional_summary_writer_domain_agnostic".
    """
    role = agent_role.split("(", 1)[0]
    slug = re.sub(r"[^A-Za-z0-9]+", "_", role)
    return slug.strip("_").lower()


def _run_dir_and_seq(run_id: str) -> tuple[Path, int]:
    """Return this run's single checkpoint folder and the next sequence number.

    The folder is created once per run_id (the first time a checkpoint fires for
    that run) and reused for every subsequent agent in the same run. Caller must
    hold _WRITE_LOCK.
    """
    run_dir = _RUN_DIRS.get(run_id)
    if run_dir is None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = _CHECKPOINT_ROOT / f"{ts}_{run_id[:8]}"
        _RUN_DIRS[run_id] = run_dir
        _RUN_SEQ[run_id] = 0

    _RUN_SEQ[run_id] += 1
    return run_dir, _RUN_SEQ[run_id]


@dataclass(frozen=True)
class CheckpointHandle:
    """Where this call's OUTPUT half will be written after the LLM returns."""

    output_path: Path


def _build_input_header(run_id: str, agent_role: str, task_name: str, output_model_name: str) -> str:
    """Build the metadata header for the INPUT file."""
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    sep = "=" * 60
    return (
        f"{sep}\n"
        f"  AGENT CHECKPOINT -- INPUT (context sent to the LLM)\n"
        f"{sep}\n"
        f"  run_id       : {run_id}\n"
        f"  agent_role   : {agent_role}\n"
        f"  task_name    : {task_name}\n"
        f"  output_model : {output_model_name}\n"
        f"  timestamp    : {ts}\n"
        f"{sep}\n\n"
    )


def _build_output_header() -> str:
    """Build the metadata header for the OUTPUT file."""
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    sep = "=" * 60
    return (
        f"{sep}\n"
        f"  AGENT CHECKPOINT -- OUTPUT (what the LLM returned)\n"
        f"{sep}\n"
        f"  timestamp    : {ts}\n"
        f"{sep}\n\n"
    )


def save_agent_input_checkpoint(
    run_id: str,
    agent_role: str,
    task_name: str,
    output_model_name: str,
    task_description: str,
) -> CheckpointHandle | None:
    """Write the full task_description to this call's INPUT file.

    Called just before Crew.kickoff(). No-ops when DEBUG_CHECKPOINTS is not set.

    Returns a handle carrying the sibling OUTPUT path, to pass to
    save_agent_output_checkpoint once the LLM call returns. None when disabled.
    """
    if not _is_enabled():
        return None

    with _WRITE_LOCK:
        run_dir, seq = _run_dir_and_seq(run_id)
        base = f"{seq:02d}_{_clean_agent_name(agent_role)}"
        input_path = run_dir / f"{base}__INPUT.txt"
        output_path = run_dir / f"{base}__OUTPUT.txt"
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            header = _build_input_header(run_id, agent_role, task_name, output_model_name)
            input_path.write_text(header + task_description, encoding="utf-8")
            logger.debug(
                "debug_checkpoint_input_saved",
                path=str(input_path),
                task_name=task_name,
                agent_role=agent_role,
            )
        except OSError as exc:
            # Never let checkpoint I/O break the pipeline.
            logger.warning("debug_checkpoint_write_failed", path=str(input_path), error=str(exc))
            return None

    return CheckpointHandle(output_path=output_path)


def save_agent_output_checkpoint(
    checkpoint: CheckpointHandle | None,
    raw_output: str,
    validated_output: Any = None,
) -> None:
    """Write the LLM's raw (and validated) output to this call's OUTPUT file.

    Called right after Crew.kickoff() resolves and output validation completes.
    No-ops when checkpointing was disabled at input time (checkpoint is None).
    """
    if checkpoint is None:
        return

    body = _build_output_header() + raw_output
    if validated_output is not None:
        body += "\n\n----- validated output_model JSON -----\n"
        body += validated_output.model_dump_json(indent=2)

    with _WRITE_LOCK:
        try:
            checkpoint.output_path.write_text(body, encoding="utf-8")
            logger.debug("debug_checkpoint_output_saved", path=str(checkpoint.output_path))
        except OSError as exc:
            # Never let checkpoint I/O break the pipeline.
            logger.warning(
                "debug_checkpoint_write_failed",
                path=str(checkpoint.output_path),
                error=str(exc),
            )
