"""Package-specific exceptions for runtime settings."""


class ConfigurationError(RuntimeError):
    """Raised when a project configuration file cannot be loaded safely."""
