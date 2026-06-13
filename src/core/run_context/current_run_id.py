"""The current pipeline run's identity, bound for the duration of one run.

A CrewAI tool runs inside an agent kickoff and cannot read the LangGraph pipeline
state. This module carries the run_id across that boundary so the document-ingestion
tools can key their Redis PII mapping. This is run identity, distinct from logging's
correlation_id.

Why a module global and not a ContextVar: CrewAI executes tool calls in worker
threads that do NOT inherit the binding thread's contextvars (a new thread starts
with the var at its default), so a ContextVar bound in the ingestion node is unset
inside the tool. A module global is shared across all threads in the process, so it
survives that boundary. The ingestion node binds it around a single, synchronous
kickoff, so there is one binding active at a time.
"""

import threading
from collections.abc import Iterator
from contextlib import contextmanager

from src.core.run_context.exceptions import MissingRunIdError

# None when no run is bound. Guarded by a lock so the set/restore pair around a
# kickoff is atomic against the worker threads that read it.
_lock = threading.Lock()
_current_run_id: str | None = None

# TODO: Concurrent tailor_resume() calls in one process would share this global.
#       Proposed: a thread-local keyed by the binding thread, or pass run_id through
#       CrewAI's task context once it supports thread-context propagation.
#       Deferred: the pipeline runs one resume extraction at a time per process today.


@contextmanager
def bind_run_id(run_id: str) -> Iterator[None]:
    """Bind run_id as the current run for the duration of the block.

    Expects a non-empty run_id. Tools called inside the block (even when CrewAI
    runs them in a worker thread) can read it with get_current_run_id(). Restores
    the prior value on exit, so nested or sequential runs do not leak into each other.

    Raises:
        ValueError: if run_id is empty.
    """
    if not run_id:
        raise ValueError("run_id must be a non-empty string")
    global _current_run_id
    with _lock:
        previous_run_id = _current_run_id
        _current_run_id = run_id
    try:
        yield
    finally:
        with _lock:
            _current_run_id = previous_run_id


def get_current_run_id() -> str:
    """Return the run_id bound by the enclosing bind_run_id() block.

    Precondition: called inside a bind_run_id() block (possibly from another thread).

    Returns:
        The bound run_id.

    Raises:
        MissingRunIdError: if no run_id is bound (fail closed, never guess one).
    """
    with _lock:
        run_id = _current_run_id
    if run_id is None:
        raise MissingRunIdError(
            "No run_id is bound; call get_current_run_id() only inside a bind_run_id() block"
        )
    return run_id
