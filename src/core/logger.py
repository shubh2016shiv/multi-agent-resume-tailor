"""
Structured Logging with structlog
==================================

Industry-standard structured logging using the structlog library.

WHY structlog?
--------------
- Battle-tested library used by companies like Stripe, Lyft, and others
- Structured logging makes logs queryable and machine-readable
- Automatic context binding (environment, version, user_id, etc.)
- Clean API with minimal boilerplate

Usage:
------
```python
from src.core.logger import get_logger

logger = get_logger(__name__)
logger.info("task_started", task_id="123", user_id=456)
logger.error("task_failed", task_id="123", error="timeout")
```
"""

import logging
import sys
from pathlib import Path
from typing import Any

import structlog

from src.core.settings import get_config

_configured = False
_configured_signature: tuple[str, str, str | None, str] | None = None


def _ensure_windows_utf8_streams() -> None:
    """Reconfigure Windows console streams to UTF-8 when possible."""
    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def _build_processors(log_format: str) -> list[Any]:
    """Build the structlog processor chain for the configured format."""
    common_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
    ]
    if log_format == "json":
        return [
            *common_processors,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ]
    return [
        *common_processors,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
        structlog.dev.ConsoleRenderer(),
    ]


def _build_handlers(log_file: str | None) -> list[logging.Handler]:
    """Create logging handlers without mutating global logging state."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.insert(0, logging.FileHandler(log_path, encoding="utf-8"))
    return handlers


def _configure_standard_logging(level: int, log_file: str | None) -> None:
    """Install a predictable root logger handler set exactly once per config."""
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()
    for handler in _build_handlers(log_file):
        handler.setFormatter(logging.Formatter("%(message)s"))
        root_logger.addHandler(handler)
    root_logger.setLevel(level)


def configure_structlog() -> None:
    """
    Configure structlog based on application config.

    This function sets up the logging pipeline (processors) that transform
    log entries before output. We configure it once globally and reuse it.
    """
    global _configured, _configured_signature
    config = get_config()
    signature = (
        config.logging.level,
        config.logging.format,
        config.logging.log_file,
        config.application.environment,
    )
    if _configured and _configured_signature == signature:
        return

    _ensure_windows_utf8_streams()
    log_level = getattr(logging, config.logging.level)
    processors = _build_processors(config.logging.format)

    # Configure structlog globally
    structlog.configure(
        processors=processors,
        # Filter logs by level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        # WHY: Prevents debug logs from cluttering production
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        # Use dict for context storage (simple and fast)
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        # Cache logger instances for performance
        # WHY: Avoid recreating loggers for the same name
        cache_logger_on_first_use=True,
    )

    _configure_standard_logging(log_level, config.logging.log_file)

    # Bind global context that appears in EVERY log entry
    # WHY: Automatically adds environment/version to all logs without manual effort
    # EXAMPLE: Makes it easy to filter production vs development logs
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        environment=config.application.environment,
        version="0.1.0",
    )

    _configured = True
    _configured_signature = signature


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """
    Get a structlog logger instance.

    This is the main function you'll use throughout the codebase.
    Call it once per module and use the returned logger.

    Args:
        name: Logger name (use __name__ to get your module name)

    Returns:
        Configured structlog logger with all context bound

    Example:
        logger = get_logger(__name__)
        logger.info("processing_started", user_id=123)
        logger.error("processing_failed", error="timeout", retry_count=3)

    Output (development):
        2025-11-21 04:10:09 [info     ] processing_started    environment=development user_id=123 version=0.1.0

    Output (production JSON):
        {"timestamp": "2025-11-21T04:10:09Z", "level": "info", "event": "processing_started",
         "environment": "production", "user_id": 123, "version": "0.1.0"}
    """
    configure_structlog()
    return structlog.get_logger(name)


# Alias for compatibility with existing code that uses setup_logger
setup_logger = get_logger
