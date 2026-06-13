"""Exceptions for the run-context package."""


class MissingRunIdError(RuntimeError):
    """Raised when the current run_id is read outside a bind_run_id() block."""
