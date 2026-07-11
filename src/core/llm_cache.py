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

CACHE_DIR = ".litellm_cache"


def configure_llm_cache() -> None:
    """Synchronize LiteLLM's process-wide cache with the current feature flag.

    Idempotent for repeated calls with the same setting. If the feature flag
    changes between calls, this function updates LiteLLM's cache state to match.

    Reads `litellm.cache` itself as the source of truth (rather than
    remembering a separately-tracked bool), so this function is always correct
    relative to LiteLLM's actual state -- even if something else cleared it
    between calls -- with no state of its own to go stale.
    """
    # This function is intentionally called from the two LLM choke points rather
    # than from random callers. That keeps "should caching be on for this
    # process?" as one decision made once per real LLM entry path.
    ####################################################
    # STEP 1: READ WHETHER CACHING SHOULD BE ENABLED RIGHT NOW
    ####################################################
    cache_enabled = get_config().feature_flags.enable_cache

    ####################################################
    # STEP 2: LEAVE THE PROCESS-WIDE CACHE ALONE IF NOTHING CHANGED
    ####################################################
    cache_is_active = litellm.cache is not None
    if cache_enabled == cache_is_active:
        return

    ####################################################
    # STEP 3: CLEAR LITELLM'S CACHE WHEN THE FEATURE FLAG IS OFF
    ####################################################
    if not cache_enabled:
        litellm.cache = None
        logger.info("llm_response_cache_disabled")
        return

    ####################################################
    # STEP 4: ATTACH THE DISK CACHE WHEN THE FEATURE FLAG IS ON
    ####################################################
    # LiteLLM exposes Cache at runtime, but its type stubs do not mark that
    # attribute as public. Keep the exception narrow to this one handoff.
    cache_factory: Any = litellm.Cache  # pyright: ignore[reportPrivateImportUsage]
    litellm.cache = cache_factory(type="disk", disk_cache_dir=CACHE_DIR)
    logger.info("llm_response_cache_enabled", cache_dir=CACHE_DIR)


"""
REUSABLE PATTERN — portable to any project, not specific to LiteLLM or this repo

The design choice worth carrying forward from this file: when synchronizing
your own code with a stateful external object (a library's cache, a client
connection, a feature toggle on some SDK), don't keep a second, separately
tracked variable to remember what you last set it to. Read the external
object's own current state directly and compare against that.

Before (the tempting version):

    _configured_cache_enabled: bool | None = None

    def configure_cache() -> None:
        global _configured_cache_enabled
        if _configured_cache_enabled == desired_state:
            return
        ...
        _configured_cache_enabled = desired_state

After (this file's version):

    def configure_cache() -> None:
        current_state = external_object.cache is not None   # ask reality, don't remember
        if desired_state == current_state:
            return
        ...

Why the second form is better, in any language or framework:
- One less piece of state to keep in sync -- nothing to forget to update.
- Self-correcting: if anything else (a test teardown, a library upgrade, a
  second caller) changes the external object between calls, this function
  notices on its next call instead of trusting a now-stale local copy.
- No `global`/mutable-module-state keyword needed, which also removes the
  read-check-write race condition a shared mutable global invites under
  concurrency.
- Tests assert on the externally observable outcome (the real object's state)
  instead of a private implementation-detail variable, so they survive
  refactors that don't change behavior.

When this pattern applies: any time you're about to write "remember whether
I already turned X on" as your own bool/enum, check first whether X exposes
its own current state you can just read instead.
"""
