"""Public facade for the run-id binding package.

Example:
    from src.core.run_id_binding import bind_run_id, get_current_run_id

This package owns one concern only: exposing the current orchestration ``run_id``
to CrewAI ingestion tools while the ingestion node is running. It is not general
runtime context, logging context, tracing context, or orchestration state.
"""

from src.core.run_id_binding.current_run_id import bind_run_id, get_current_run_id
from src.core.run_id_binding.exceptions import MissingRunIdError

__all__ = ["MissingRunIdError", "bind_run_id", "get_current_run_id"]
