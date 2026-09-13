"""Turn LangSmith recording on (or leave it off safely).

This file is the only place that configures LangSmith and tells LiteLLM to
send each LLM call to the dashboard. Other app code should import
``src.observability``, not this file.

What "on" means
---------------
Agents talk to models through LiteLLM. Once we add ``"langsmith"`` to
LiteLLM's callback list, LiteLLM notifies LangSmith after every model call
with the prompt, the reply, token counts, cost, and how long it took. You do
not need to change agent code for that.

Named boxes around agents (``@trace_agent`` / ``@trace_tool``) are a separate
feature in ``tracing.py``. Those helpers are available but not wired onto
production functions yet.

Rules we keep
-------------
- Never crash the pipeline because tracing failed.
- If something is missing (key, library, config switch), log why and return
  False / stay off.
"""

import os

from src.core.logger import get_logger
from src.core.settings import get_config

logger = get_logger(__name__)

# True only after init_observability() finishes successfully.
_is_initialized = False


def init_observability(project_name: str = "resume-tailor-agents", enabled: bool = True) -> bool:
    """Start LangSmith recording once when the app starts.

    What it needs:
        - ``observability.enabled`` true in settings (unless you pass
          ``enabled=False`` to force off, e.g. in tests)
        - ``LANGSMITH_API_KEY`` available via settings (from ``.env``)
        - LiteLLM installed

    What it returns:
        - ``True`` — recording is active
        - ``False`` — recording stayed off (missing key, disabled, or import
          failed). The rest of the app still runs normally.

    Safe to call more than once: later calls just report the current state.

    Who calls this today:
        ``src/orchestration/runner.py``, at the top of each public entry point.
    """
    global _is_initialized

    ####################################################
    # STEP 1: EXIT EARLY IF OBSERVABILITY IS ALREADY ACTIVE
    ####################################################
    # Calling init again (or importing runner twice) must not redo setup.
    if _is_initialized:
        return True

    ####################################################
    # STEP 2: READ OBSERVABILITY SETTINGS FROM CENTRAL APP CONFIG
    ####################################################
    config = get_config()
    observability_config = config.observability

    ####################################################
    # STEP 3: RESPECT BOTH THE CALLER SWITCH AND THE APP-LEVEL SWITCH
    ####################################################
    # Tests can pass enabled=False; settings.yaml can also turn tracing off.
    if not enabled or not observability_config.enabled:
        logger.info("langsmith_disabled", reason="observability.enabled is false")
        return False

    ####################################################
    # STEP 4: REQUIRE THE LANGSMITH API KEY BEFORE SETUP
    ####################################################
    # Better to stay off with a clear warning than fail later inside SDKs.
    api_key = config.langsmith_api_key
    if not api_key:
        logger.warning(
            "langsmith_api_key_missing",
            hint="Set LANGSMITH_API_KEY in .env to enable tracing.",
        )
        return False

    ####################################################
    # STEP 5: REGISTER THE LANGSMITH CALLBACK WITH LITELLM
    ####################################################
    # This is the automatic recorder: after this, LiteLLM can report every
    # model call (agents + structured tools) without decorating those calls.
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
    # STEP 6: HAND OFF SETTINGS TO THIRD-PARTY LIBRARIES VIA ENV VARS
    ####################################################
    # LangSmith and LiteLLM's langsmith callback only read config from
    # environment variables. We intentionally copy our typed settings into
    # those vars here — the one place in the app that does this hand-off.
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = api_key
    os.environ["LANGSMITH_PROJECT"] = project_name or observability_config.project
    os.environ["LANGSMITH_ENDPOINT"] = observability_config.endpoint

    ####################################################
    # STEP 7: MARK OBSERVABILITY AS ACTIVE AND LOG THE RESULT
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
    """Return whether LangSmith recording was successfully turned on.

    Prefer calling this function over copying a bool at import time. Startup
    often happens *after* other observability modules are imported, so a
    cached import-time value would stay False forever.
    """
    ####################################################
    # STEP 1: REPORT THE LIVE INITIALIZATION FLAG
    ####################################################
    return _is_initialized
