"""LangSmith initialization and lifecycle.

This module is the single place that talks to LangSmith. The rest of the app
only ever imports the facade in ``src/observability/__init__.py``; it never
imports this file directly. That keeps the tracing vendor swappable behind one
seam.

HOW AGENT BEHAVIOR IS CAPTURED (two complementary layers)
---------------------------------------------------------
1. Automatic LLM layer — CrewAI runs every model call through LiteLLM. Setting
   ``litellm.callbacks = ["langsmith"]`` once makes LiteLLM stream each call
   (prompt, completion, token counts, cost, latency) straight to LangSmith. No
   per-agent code is needed for this.
2. Readable workflow layer — ``trace_agent`` / ``trace_tool`` wrap functions
   with ``langsmith.traceable`` so each agent run shows up as a named span in
   the dashboard. The LiteLLM calls from layer 1 nest inside that span
   automatically, giving a per-agent -> per-LLM-call tree with token/cost
   rollups.

DESIGN RULES
------------
- Never raise into the pipeline. If the key/library is missing or tracing is
  disabled, every function degrades to a safe no-op and logs why.
- Always mirror metrics to structlog so behavior is observable even offline.
"""

import os

from src.core.logger import get_logger
from src.core.settings import get_config

logger = get_logger(__name__)

# Set True only after a successful init_observability() call.
_is_initialized = False


def init_observability(project_name: str = "resume-tailor-agents", enabled: bool = True) -> bool:
    """Initialize LangSmith tracing once at application startup.

    Expects: ``LANGSMITH_API_KEY`` is set (read from settings, which loads it
        from .env); ``observability`` settings present in config.
    Returns: True if tracing is now active, False if it was disabled, the key
        was missing, or the libraries are unavailable (pipeline still runs).
    Notes: idempotent — repeat calls are no-ops that return the current state.
    """
    global _is_initialized

    ####################################################
    # STEP 1: EXIT EARLY IF OBSERVABILITY IS ALREADY ACTIVE#
    ####################################################
    if _is_initialized:
        return True

    ####################################################
    # STEP 2: READ OBSERVABILITY SETTINGS FROM CENTRAL APP CONFIG#
    ####################################################
    config = get_config()
    observability_config = config.observability

    ####################################################
    # STEP 3: RESPECT BOTH THE CALLER SWITCH AND THE APP-LEVEL SWITCH#
    ####################################################
    if not enabled or not observability_config.enabled:
        logger.info("langsmith_disabled", reason="observability.enabled is false")
        return False

    ####################################################
    # STEP 4: REQUIRE THE LANGSMITH API KEY BEFORE SETUP#
    ####################################################
    api_key = config.langsmith_api_key
    if not api_key:
        logger.warning(
            "langsmith_api_key_missing",
            hint="Set LANGSMITH_API_KEY in .env to enable tracing.",
        )
        return False

    ####################################################
    # STEP 5: REGISTER THE LANGSMITH CALLBACK WITH LITELLM#
    ####################################################
    try:
        import litellm
    except ImportError:
        logger.warning(
            "litellm_import_failed",
            impact="LLM-level token/cost tracing disabled.",
        )
        return False

    if "langsmith" not in litellm.callbacks:
        litellm.callbacks = [*litellm.callbacks, "langsmith"]

    ####################################################
    # STEP 6: HAND OFF SETTINGS TO THIRD-PARTY LIBRARIES VIA ENV VARS#
    ####################################################
    # WRITE config out to environment variables — this is the ONE place in the
    # codebase that does so, and it is on purpose. The LangSmith SDK and
    # LiteLLM's "langsmith" callback are third-party libraries that read their
    # settings ONLY from these env vars; they give us no Python API to pass the
    # values in directly. So this block is a one-way hand-off: every value comes
    # FROM settings (above), and we copy it OUT to where those libraries look.
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = api_key
    os.environ["LANGSMITH_PROJECT"] = project_name or observability_config.project
    os.environ["LANGSMITH_ENDPOINT"] = observability_config.endpoint

    ####################################################
    # STEP 7: MARK OBSERVABILITY AS ACTIVE AND LOG THE RESULT#
    ####################################################
    _is_initialized = True
    logger.info(
        "langsmith_initialized",
        project=project_name or observability_config.project,
        endpoint=observability_config.endpoint,
        dashboard="https://smith.langchain.com",
    )
    return True


def is_observability_enabled() -> bool:
    """Return True if LangSmith tracing is initialized and active."""
    return _is_initialized
