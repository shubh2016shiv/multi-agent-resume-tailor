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

import structlog

from .config import get_config

_configured = False


def configure_structlog() -> None:
    """
    Configure structlog based on application config.

    This function sets up the logging pipeline (processors) that transform
    log entries before output. We configure it once globally and reuse it.
    """
    global _configured
    if _configured:
        return

    config = get_config()

    # PROCESSORS: The pipeline that transforms log entries
    # Each processor adds/modifies data before the next one runs
    # Order matters! They execute in sequence.

    if config.logging.format == "json":
        # PRODUCTION: JSON output for log aggregation tools (ELK, Splunk, etc.)
        # WHY: Machine-readable, queryable, easy to parse and analyze
        processors = [
            # 1. Merge context variables (environment, version, request_id, etc.)
            structlog.contextvars.merge_contextvars,
            # 2. Add log level (INFO, ERROR, etc.) to the output
            structlog.processors.add_log_level,
            # 3. Add ISO timestamp (2025-11-21T04:10:09Z)
            # WHY ISO: Unambiguous, sortable, timezone-aware
            structlog.processors.TimeStamper(fmt="iso"),
            # 4. Render as JSON (final step)
            structlog.processors.JSONRenderer(),
        ]
    else:
        # DEVELOPMENT: Pretty console output for human readability
        # WHY: Easier to read during development, color-coded, formatted
        processors = [
            # Same context merging as production
            structlog.contextvars.merge_contextvars,
            # Add log level
            structlog.processors.add_log_level,
            # Human-readable timestamp
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
            # Pretty console output with colors
            # WHY: Much easier to scan visually during development
            structlog.dev.ConsoleRenderer(),
        ]

    # Configure structlog globally
    structlog.configure(
        processors=processors,
        # Filter logs by level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        # WHY: Prevents debug logs from cluttering production
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, config.logging.level)),
        # Use dict for context storage (simple and fast)
        context_class=dict,
        # Use print-based logger (writes to stdout/stderr)
        # WHY: Simple, works everywhere, integrates with standard logging
        logger_factory=structlog.PrintLoggerFactory(),
        # Cache logger instances for performance
        # WHY: Avoid recreating loggers for the same name
        cache_logger_on_first_use=True,
    )

    # Setup standard library logging for file output
    # WHY: structlog integrates with Python's logging module for file handling
    if config.logging.log_file:
        log_path = Path(config.logging.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=getattr(logging, config.logging.level),
            format="%(message)s",  # structlog already formats, don't double-format
            handlers=[
                logging.FileHandler(log_path, encoding="utf-8"),
                logging.StreamHandler(sys.stdout),
            ],
        )
    else:
        # Console-only logging
        logging.basicConfig(
            level=getattr(logging, config.logging.level),
            format="%(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

    # Bind global context that appears in EVERY log entry
    # WHY: Automatically adds environment/version to all logs without manual effort
    # EXAMPLE: Makes it easy to filter production vs development logs
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        environment=config.application.environment,
        version="0.1.0",
    )

    _configured = True


def get_logger(name: str = None) -> structlog.BoundLogger:
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
