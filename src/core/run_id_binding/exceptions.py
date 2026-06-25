"""Exceptions for the run-id binding package."""


class MissingRunIdError(RuntimeError):
    """Raised when code asks for the current run_id outside a binding scope."""
