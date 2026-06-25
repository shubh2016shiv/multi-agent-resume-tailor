"""
Centralized Structured Logging (built on structlog)
====================================================

WHAT THIS FILE DOES, IN ONE SENTENCE
------------------------------------
Every module calls `get_logger(__name__)` and logs key=value events
("task_started", task_id="123") instead of English sentences. This file
is the ONE place that decides how those events become actual lines: on a
developer's screen (colored and readable) or in production (JSON, which a
log database can search and chart).

WHY structlog
-------------
structlog is the library most production Python teams converge on for
structured logging. Unlike the plain `logging` module, it does not make
you hand-build formatters and filters just to get output that is readable
AND machine-parseable at the same time. A "processor" (see below) is the
only concept you need to learn to extend it.

THE DESIGN DECISIONS MADE HERE (plain language)
-----------------------------------------------
1. ONE FORMAT FOR HUMANS, ONE FOR MACHINES.
   On a laptop you want logs your eyes can read. In production you want
   JSON, because a JSON line is a row a log database can filter and chart
   -- an English sentence is not. We switch on `logging.format`.

2. EVERY LINE CARRIES service / environment / version / host / pid,
   WITHOUT EACH DEVELOPER REMEMBERING TO ADD IT.
   This "static context" is true for every log line the process emits.

3. STATIC CONTEXT IS BAKED INTO THE PIPELINE, NOT INTO contextvars.
   This is the subtle bug the old version had. If you store
   "environment"/"version" with `bind_contextvars`, then a normal
   per-request `clear_contextvars()` wipes them too. We instead inject
   them through a processor closure (see `_make_static_context_processor`),
   so nothing downstream can erase the service's identity.

4. EVERY LINE CAN CARRY PER-RUN IDENTITY, WHEN A CALLER BINDS IT.
   The pipeline's `merge_contextvars` step picks up anything bound via
   `structlog.contextvars.bind_contextvars(...)`. So to tie every log to
   one run, the orchestrator binds the run_id once at run start and it
   appears on every line -- the difference between "grep millions of lines
   for 'timeout'" and "filter by this one ID." If OpenTelemetry has an
   active trace, its trace_id/span_id are attached automatically too.

5. LOGS DO NOT LEAK SECRETS.
   A field accidentally named `password` or `api_key` is masked before it
   ever reaches a terminal, a file, or a log platform.

6. THE LOG FILE DOES NOT GROW FOREVER.
   The file handler rotates: once it hits a size cap it starts a fresh
   file and keeps only a few old copies.

7. YOU DON'T PAY FOR DETAIL YOU DON'T NEED.
   "Which file and line logged this" is computed only in dev (text mode),
   where it helps debugging, and skipped in production where it costs CPU
   on every call for little benefit.

8. RE-CONFIGURING IS SAFE.
   `configure_structlog()` runs the real work once and only redoes it if
   the relevant settings actually changed (every module calling
   `get_logger` would otherwise re-run setup constantly).

USAGE
-----
    from src.core.logger import get_logger

    logger = get_logger(__name__)
    logger.info("task_started", task_id="123", user_id=456)
    logger.error("task_failed", task_id="123", error="timeout")

Tying every log to one run (done by the orchestrator, not this module):
    import structlog

    structlog.contextvars.bind_contextvars(run_id=run_id)  # at run start
    # every log emitted from here on automatically includes run_id
    structlog.contextvars.clear_contextvars()               # at run end

OUTPUT EXAMPLES
---------------
Development (text, human-readable console):
    2026-06-24T10:15:02Z [info] task_started service=resume-tailor environment=development task_id=123

Production (JSON -- one line, split here for readability):
    {"timestamp": "2026-06-24T10:15:02Z", "level": "info", "event": "task_started",
     "service": "resume-tailor", "environment": "production", "version": "0.1.0",
     "host": "ip-10-0-1-23", "pid": 4821, "task_id": "123"}
"""

import logging
import os
import socket
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import structlog

from src.core.settings import get_config

# --------------------------------------------------------------------------
# OpenTelemetry is an OPTIONAL integration. We try to import it once here,
# at module load, and remember whether it worked. Every place that wants
# it checks this flag instead of re-importing -- that is what makes the
# integration "do nothing silently" when OTel is absent, rather than crash.
# --------------------------------------------------------------------------
try:
    from opentelemetry import trace as _otel_trace

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False


# --------------------------------------------------------------------------
# Static facts attached to EVERY log line. These never change during a
# process's life, so they are constants rather than configuration -- there
# is exactly one correct value for each in a given run.
# --------------------------------------------------------------------------
_SERVICE_NAME = "resume-tailor"
_SERVICE_VERSION = "0.1.0"

# Rotation policy for the log FILE (stdout is never rotated -- that is the
# container platform's job). Constants, not config: tune here if ops needs
# a different cap. 10 MB per file, 5 old copies kept.
DEFAULT_MAX_LOG_BYTES = 10 * 1024 * 1024
DEFAULT_LOG_BACKUP_COUNT = 5

# Field names whose VALUE must never appear in a log. Matching is
# "substring, case-insensitive", so the single entry "password" also
# catches `user_password` and `PASSWORD_HASH`.
_SENSITIVE_KEY_MARKERS: tuple[str, ...] = (
    "password",
    "passwd",
    "secret",
    "api_key",
    "apikey",
    "authorization",
    "access_key",
    "private_key",
)
_REDACTED_VALUE = "***REDACTED***"


# --------------------------------------------------------------------------
# State for the "set up logging once, redo only if settings changed" guard.
#   _logging_configured            -> have we already run structlog setup?
#   _configured_settings_fingerprint -> the snapshot of logging settings we
#       last set up against. configure_structlog() compares the current
#       settings to this; if they match, it returns immediately instead of
#       rebuilding the whole pipeline on every get_logger() call.
# --------------------------------------------------------------------------
_logging_configured = False
_configured_settings_fingerprint: tuple[Any, ...] | None = None


# ==========================================================================
# CUSTOM PROCESSORS
#
# A "processor" is just a function that takes the in-progress log event
# (a plain dict of key/value pairs) and returns a modified version of it.
# structlog runs every event through a CHAIN of these, in order, before
# rendering it as text or JSON. Writing one is the normal way to extend
# structlog -- there is no separate plugin system to learn.
#
# The three-argument signature (logger, method_name, event_dict) is the
# contract structlog calls every processor with; we only ever use the
# third argument here.
# ==========================================================================


def _redact_sensitive_fields(logger: Any, method_name: str, event_dict: dict) -> dict:
    """Mask any field whose NAME looks sensitive, regardless of its value.

    Only the TOP-LEVEL keys of the event are inspected. If you log a whole
    nested object (`logger.info("user", data=user_dict)`), fields inside
    `user_dict` are NOT scanned -- keep secrets out of nested payloads, or
    flatten them before logging.
    """
    for key in event_dict:
        if any(marker in key.lower() for marker in _SENSITIVE_KEY_MARKERS):
            event_dict[key] = _REDACTED_VALUE
    return event_dict


def _add_trace_context(logger: Any, method_name: str, event_dict: dict) -> dict:
    """Attach the active OpenTelemetry trace_id/span_id, if any.

    This lets you jump from a log line straight to the matching distributed
    trace. If OpenTelemetry is not installed, or no span is currently
    active, this returns the event unchanged.
    """
    if not _OTEL_AVAILABLE:
        return event_dict

    span = _otel_trace.get_current_span()
    span_context = span.get_span_context()
    if span_context.is_valid:
        # Trace tooling expects fixed hex widths: 32 chars for a trace id,
        # 16 for a span id. "032x"/"016x" produce exactly that.
        event_dict["trace_id"] = format(span_context.trace_id, "032x")
        event_dict["span_id"] = format(span_context.span_id, "016x")
        event_dict["trace_sampled"] = bool(span_context.trace_flags.sampled)
    return event_dict


def _make_static_context_processor(static_context: dict[str, Any]):
    """Build a processor that injects the fixed service-identity fields.

    WHY a processor and not `bind_contextvars`: contextvars-based values
    can be wiped by any `clear_contextvars()` call elsewhere in the app
    (a normal thing to do between requests). Baking these into a closure
    makes them part of the pipeline itself -- nothing downstream can erase
    "which service emitted this log."
    """

    def _inject_static_context(logger: Any, method_name: str, event_dict: dict) -> dict:
        # Static fields go first so a caller that deliberately logs the
        # same key (rare) can still override it.
        return {**static_context, **event_dict}

    return _inject_static_context


def _build_processors(log_format: str, static_context: dict[str, Any]) -> list[Any]:
    """Build the structlog processor chain for the configured format.

    The list is read top-to-bottom for every log call, so ORDER matters:
    each processor sees the dict as shaped by everything before it.
    Returns the full chain ending in a renderer (JSON or console).
    """

    # STEP 1: ENRICH -- add the context every line needs, in order.
    enrich_processors = [
        # Merges any structlog contextvars bound by the caller into the event.
        # This is the SEAM for per-run identity: when the orchestrator wants
        # the pipeline run_id on every log line, bind it once at run start
        # with `structlog.contextvars.bind_contextvars(run_id=...)` and it
        # appears here automatically. (Do not couple this module to
        # src/core/run_id_binding -- that package is scoped to ingestion
        # tools and fails closed outside an ingestion scope.)
        structlog.contextvars.merge_contextvars,
        # Adds the "level" field (info / warning / error / ...).
        structlog.processors.add_log_level,
        # Adds service/environment/version/host/pid. See decision #3 above
        # for why this is a processor and not contextvars.
        _make_static_context_processor(static_context),
        # Adds trace_id/span_id when an OTel span is active; no-op otherwise.
        _add_trace_context,
        # Masks secrets. Runs AFTER the above so it also catches a secret
        # that arrived via bound contextvars.
        _redact_sensitive_fields,
        # UTC ISO-8601 timestamp -- the sortable format every log platform
        # expects. One timestamper for both formats keeps lines comparable.
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]

    # STEP 2: DEV-ONLY EXTRA -- file/function/line costs CPU per call, so
    # we only add it in text (development) mode, not in JSON (production).
    if log_format != "json":
        enrich_processors.append(
            structlog.processors.CallsiteParameterAdder(
                {
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                    structlog.processors.CallsiteParameter.LINENO,
                }
            )
        )

    # STEP 3: RENDER -- turn the dict into its final output shape.
    if log_format == "json":
        return [
            *enrich_processors,
            # Render an exception as a structured list of stack frames
            # (file/line/function each) instead of one text blob, so a log
            # platform can search inside it.
            #
            # show_locals=False is deliberate: dumping every local variable
            # in every frame is a known way to leak a secret that happened
            # to be in a local when the crash occurred -- and the redaction
            # processor above cannot reach inside a traceback.
            structlog.processors.ExceptionRenderer(
                structlog.tracebacks.ExceptionDictTransformer(show_locals=False)
            ),
            structlog.processors.JSONRenderer(),
        ]

    # Development: colored, aligned, human-readable console output with a
    # normal (non-JSON) exception traceback.
    return [*enrich_processors, structlog.dev.ConsoleRenderer()]


def _build_handlers(log_file: str | None) -> list[logging.Handler]:
    """Create the output handlers (stdout, plus an optional rotating file).

    Always logs to stdout. If `log_file` is set, ALSO writes to that file
    through a size-capped rotating handler so it never grows without limit.
    Does not mutate any global logging state -- the caller installs these.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # RotatingFileHandler instead of a plain FileHandler: once the file
        # hits DEFAULT_MAX_LOG_BYTES it is rotated out and a fresh one
        # starts, keeping at most DEFAULT_LOG_BACKUP_COUNT old copies.
        #
        # NOTE: inside a container, the usual pattern is to log to stdout
        # ONLY (already covered above) and let the platform handle
        # collection/rotation. Keep `log_file` unset there; this path is
        # for VMs, bare metal, and local debugging.
        handlers.insert(
            0,
            RotatingFileHandler(
                log_path,
                maxBytes=DEFAULT_MAX_LOG_BYTES,
                backupCount=DEFAULT_LOG_BACKUP_COUNT,
                encoding="utf-8",
            ),
        )
    return handlers


def _configure_standard_logging(level: int, log_file: str | None) -> None:
    """Install a predictable set of root-logger handlers for this config.

    Removes any handlers a previous configuration installed, then attaches
    fresh ones. The message reaching these handlers is already fully
    formatted (text or JSON) by structlog, so the stdlib formatter's only
    job is to pass it through untouched.
    """
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()
    for handler in _build_handlers(log_file):
        handler.setFormatter(logging.Formatter("%(message)s"))
        root_logger.addHandler(handler)
    root_logger.setLevel(level)


def configure_structlog() -> None:
    """Configure the structlog pipeline from application settings.

    Idempotent: does the real work once and only re-runs it if the
    settings that affect logging actually changed. Safe to call from every
    module via `get_logger`.
    """
    global _logging_configured, _configured_settings_fingerprint

    # STEP 1: read config and skip if nothing relevant changed.
    config = get_config()
    settings_fingerprint = (
        config.logging.level,
        config.logging.format,
        config.logging.log_file,
        config.application.environment,
    )
    if _logging_configured and _configured_settings_fingerprint == settings_fingerprint:
        return

    # STEP 2: build the facts that go on every log line.
    static_context = {
        "service": _SERVICE_NAME,
        "environment": config.application.environment,
        "version": _SERVICE_VERSION,
        "host": socket.gethostname(),
        "pid": os.getpid(),
    }

    # STEP 3: wire up the processor pipeline.
    log_level = getattr(logging, config.logging.level)
    processors = _build_processors(config.logging.format, static_context)

    structlog.configure(
        processors=processors,
        # Filter by level BEFORE the expensive formatting work, so a DEBUG
        # log filtered out in production costs almost nothing.
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        # Cache the logger built for each name, so repeated
        # get_logger(__name__) calls for a module don't rebuild it.
        cache_logger_on_first_use=True,
    )

    # STEP 4: point the underlying stdlib logger at stdout (+ optional file).
    _configure_standard_logging(log_level, config.logging.log_file)

    # STEP 5: clear leftover per-request bindings from before this
    # (re-)configuration. The service identity lives in the processor
    # chain (STEP 2/3), NOT in contextvars, so clearing here is safe -- it
    # only removes request-scoped state, never the global identity.
    structlog.contextvars.clear_contextvars()

    _logging_configured = True
    _configured_settings_fingerprint = settings_fingerprint


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """Return a configured structlog logger. Call once per module.

    Ensures the pipeline is configured, then returns a logger bound to
    `name` (pass `__name__`). The returned logger emits key=value events:
    `logger.info("event_name", key=value, ...)`.
    """
    configure_structlog()
    return structlog.get_logger(name)
