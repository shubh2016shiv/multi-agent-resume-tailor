"""On-disk cache for LLM responses, toggled by feature_flags.enable_cache.

When enabled, litellm serves an identical (model, messages, params) completion from a
local disk cache instead of re-calling the provider. Repeated development runs then cost
nothing after the first. The cache key includes the full request, so different resumes or
job descriptions never collide. When disabled, every call hits the provider as normal.

configure_llm_cache() is called at the two LLM entry points (run_agent_task for agents,
request_structured_output for tools), so every pipeline LLM call is covered.
"""

import litellm

from src.core.logger import get_logger
from src.core.settings import get_config

logger = get_logger(__name__)

_CACHE_DIR = ".litellm_cache"
_configured = False


def configure_llm_cache() -> None:
    """Enable litellm's on-disk response cache when feature_flags.enable_cache is True.

    Idempotent: runs once per process. Toggling the flag takes effect on the next run.
    """
    global _configured
    if _configured:
        return
    _configured = True

    if not get_config().feature_flags.enable_cache:
        return
    litellm.cache = litellm.Cache(type="disk", disk_cache_dir=_CACHE_DIR)
    logger.info("LLM response cache enabled (repeated identical calls served from disk)", dir=_CACHE_DIR)
