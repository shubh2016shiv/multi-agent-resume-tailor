"""Configure LiteLLM's on-disk response cache for this process.

When enabled, LiteLLM serves an identical completion from a local disk cache
instead of re-calling the provider. Repeated development runs then cost
nothing after the first. The cache key includes the full request, so different
resumes or job descriptions never collide.

WIRING: `configure_llm_cache()` is not called arbitrarily around the codebase.
It is called at the two real LLM entry points only:
- `src/orchestration/crew_task_execution.py::run_agent_task()` for agent calls
- `src/tools/llm_gateway/structured_output.py::request_structured_output()` for
  tool-side structured calls

That makes this module the single place where process-wide LiteLLM cache state
is turned on, left alone, or turned off.
"""

from typing import Any

import litellm

from src.core.logger import get_logger
from src.core.settings import get_config

logger = get_logger(__name__)

_CACHE_DIR = ".litellm_cache"
_configured_cache_enabled: bool | None = None


def configure_llm_cache() -> None:
    """Synchronize LiteLLM's process-wide cache with the current feature flag.

    Idempotent for repeated calls with the same setting. If the feature flag
    changes between calls, this function updates LiteLLM's cache state to match.
    """
    # This function is intentionally called from the two LLM choke points rather
    # than from random callers. That keeps "should caching be on for this
    # process?" as one decision made once per real LLM entry path.
    ####################################################
    # STEP 1: READ WHETHER CACHING SHOULD BE ENABLED RIGHT NOW#
    ####################################################
    cache_enabled = get_config().feature_flags.enable_cache

    ####################################################
    # STEP 2: LEAVE THE PROCESS-WIDE CACHE ALONE IF NOTHING CHANGED#
    ####################################################
    global _configured_cache_enabled
    if _configured_cache_enabled == cache_enabled:
        return

    ####################################################
    # STEP 3: CLEAR LITELLM'S CACHE WHEN THE FEATURE FLAG IS OFF#
    ####################################################
    if not cache_enabled:
        litellm.cache = None
        _configured_cache_enabled = False
        logger.info("llm_response_cache_disabled")
        return

    ####################################################
    # STEP 4: ATTACH THE DISK CACHE WHEN THE FEATURE FLAG IS ON#
    ####################################################
    # LiteLLM exposes Cache at runtime, but its type stubs do not mark that
    # attribute as public. Keep the exception narrow to this one handoff.
    cache_factory: Any = litellm.Cache  # pyright: ignore[reportPrivateImportUsage]
    litellm.cache = cache_factory(type="disk", disk_cache_dir=_CACHE_DIR)
    _configured_cache_enabled = True
    logger.info("llm_response_cache_enabled", cache_dir=_CACHE_DIR)
