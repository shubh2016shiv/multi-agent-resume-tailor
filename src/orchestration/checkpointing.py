"""The save file that lets a paused run be resumed later.

WHY THIS EXISTS
    A run can stop in the middle. When the experience stage finds a bullet it cannot
    improve without a fact only the candidate has, the graph pauses and the process
    exits. The candidate answers a question sheet hours later, and a NEW process has
    to pick the run up exactly where it stopped -- with the parsed resume, the job
    description, the strategy, and every rewritten bullet still in hand.

    Nothing survives a process exit in memory, so before pausing, LangGraph writes the
    whole pipeline state to a SQLite file. That file is the checkpoint, and this module
    opens and closes it. Every run gets one, whether or not it ever pauses.

WHO USES IT
    runner.py opens it before compiling the graph, passes it to the graph, and closes
    it when the run ends. Where the file then goes -- archived into a paused-run
    directory, or deleted -- is runner.py's decision, not this module's.

THE PART THAT LOOKS STRANGE: THE ALLOWLIST
    State is full of Pydantic objects (Resume, JobDescription, ...), and a SQLite file
    holds bytes. So LangGraph flattens each object to bytes plus the name of the class
    it came from, and on resume it imports that class by name and rebuilds the object.

    Importing a class named in a file is dangerous: edit the file, name any class you
    like, and loading it runs that class's code (CVE-2026-28277). So LangGraph will
    only rebuild classes you listed in advance. checkpoint_allowlist.py is that list.

    Practical consequence: if you add a Pydantic model to the pipeline state, add it to
    that list too, or a resumed run will fail to load. A test enforces this, so you do
    not have to remember -- see checkpoint_allowlist.py for how.
"""

import sqlite3
from pathlib import Path

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

from src.orchestration.checkpoint_allowlist import CHECKPOINT_ALLOWED_MSGPACK_MODULES


# HITL COMPONENT 4 -- DURABLE STATE (CHECKPOINT). Generic pipeline plumbing
# (every run checkpoints, HITL or not), but it's the mechanism the pause in
# src/orchestration/nodes/experience/node.py relies on. See
# src/hitl/professional_experience/README.md#7-component-4--durable-state-checkpoint
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
