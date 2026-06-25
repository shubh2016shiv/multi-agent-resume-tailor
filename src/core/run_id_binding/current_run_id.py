"""Bind the current orchestration run_id so ingestion tools can read it.

This module teaches one narrow flow:
1. ``tailor_resume()`` creates a pipeline ``run_id``.
2. LangGraph state carries that ``run_id`` through orchestration.
3. The ingestion node binds the ``run_id`` around the CrewAI kickoff.
4. Ingestion tools read the bound ``run_id`` from this module.
5. The PII mapping store uses that ``run_id`` to save and validate mappings.

This is pipeline run identity, distinct from logging's ``correlation_id``.

Why a module global and not a ``ContextVar``: CrewAI executes tool calls in worker
threads that do not inherit the binding thread's contextvars. A module global is
shared across threads in the same process, so the tool thread can still read the
currently bound pipeline ``run_id``.
"""

import threading
from collections.abc import Iterator
from contextlib import contextmanager

from src.core.run_id_binding.exceptions import MissingRunIdError

# Guards the bind/restore pair so tool threads never observe a half-updated value.
_run_id_binding_lock = threading.Lock()

# Holds the current pipeline run_id, not the logging correlation_id.
_bound_run_id: str | None = None

# WARNING: concurrent tailor_resume() calls in the same process would share this
# module-global binding.
# TODO: isolate concurrent runs per caller thread, or pass run_id through CrewAI's
#       task context once thread-context propagation is supported.
#       Deferred because: this process runs one resume-ingestion kickoff at a time today.


@contextmanager
def bind_run_id(run_id: str) -> Iterator[None]:
    """Bind a pipeline run_id for the duration of one ingestion scope.

    Expects a non-empty ``run_id``. Tools called inside the block, including
    CrewAI tools running in worker threads, can read it with
    ``get_current_run_id()``. On exit, the previously bound run_id is restored
    so nested or sequential scopes do not leak into one another.

    Raises:
        ValueError: If ``run_id`` is empty.
    """
    ####################################################
    # STEP 1: REJECT AN EMPTY RUN ID BEFORE TOUCHING SHARED STATE#
    ####################################################
    if not run_id:
        raise ValueError("run_id must be a non-empty string")

    ####################################################
    # STEP 2: SWAP IN THE NEW RUN ID AND SAVE THE PRIOR ONE#
    ####################################################
    global _bound_run_id
    with _run_id_binding_lock:
        previously_bound_run_id = _bound_run_id
        _bound_run_id = run_id

    try:
        ####################################################
        # STEP 3: KEEP THE RUN ID AVAILABLE FOR THE WHOLE CALLER SCOPE#
        ####################################################
        yield
    finally:
        ####################################################
        # STEP 4: ALWAYS RESTORE THE PRIOR RUN ID AFTER THE SCOPE ENDS#
        ####################################################
        with _run_id_binding_lock:
            _bound_run_id = previously_bound_run_id


def get_current_run_id() -> str:
    """Return the pipeline run_id bound by the current ingestion scope.

    Precondition: called inside a ``bind_run_id()`` block, possibly from a
    CrewAI worker thread started inside that block.

    Returns:
        The currently bound pipeline ``run_id``.

    Raises:
        MissingRunIdError: If no pipeline ``run_id`` is currently bound.
    """
    ####################################################
    # STEP 1: READ THE CURRENTLY BOUND RUN ID UNDER THE LOCK#
    ####################################################
    with _run_id_binding_lock:
        bound_run_id = _bound_run_id

    ####################################################
    # STEP 2: FAIL CLOSED WHEN THE CALLER IS OUTSIDE A BINDING SCOPE#
    ####################################################
    if bound_run_id is None:
        raise MissingRunIdError(
            "No run_id is bound; call get_current_run_id() only inside a bind_run_id() block"
        )

    ####################################################
    # STEP 3: RETURN THE RUN ID TO THE TOOL OR CALLER#
    ####################################################
    return bound_run_id
