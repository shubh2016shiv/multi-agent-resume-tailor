"""Public facade for the run-context package.

    from src.core.run_context import bind_run_id, get_current_run_id

Owns the current pipeline run's identity (run_id), made available to CrewAI tools
that cannot read LangGraph pipeline state directly.
"""

from src.core.run_context.current_run_id import bind_run_id, get_current_run_id
from src.core.run_context.exceptions import MissingRunIdError

__all__ = ["MissingRunIdError", "bind_run_id", "get_current_run_id"]
