"""Function-level tracing via langsmith.traceable.

``trace_agent`` and ``trace_tool`` wrap functions so LangSmith records them
as named spans. CrewAI agent runs nest inside ``trace_agent`` spans; LLM calls
nest inside ``trace_tool`` spans, building a per-agent -> per-LLM-call tree.

When tracing is off or the langsmith library is missing, these decorators
return the function unchanged — no tracing, no crash.
"""

from __future__ import annotations

from typing import Literal

from src.core.logger import get_logger
from src.observability.langsmith_init import is_observability_enabled

logger = get_logger(__name__)


def _wrap_with_traceable(run_type: Literal["chain", "tool"], func):
    """Wrap a function so LangSmith records it as a span, or leave it alone.

    run_type is "chain" for whole agents or "tool" for helper functions.
    If tracing is off (or the langsmith library is missing), the function is
    returned exactly as it came in — it runs normally with no tracing.
    """
    # Call the function (don't import the flag's value) so we read whether
    # tracing is live RIGHT NOW. init runs after this module imports, so a
    # snapshotted bool would be stale forever.
    if not is_observability_enabled():
        return func

    try:
        from langsmith import traceable
    except ImportError:
        logger.warning("langsmith_import_failed", function=func.__name__)
        return func

    return traceable(run_type=run_type, name=func.__name__)(func)


def trace_agent(func):
    """Make an agent show up as a named "chain" span in LangSmith."""
    return _wrap_with_traceable("chain", func)


def trace_tool(func):
    """Make a helper function show up as a named "tool" span in LangSmith."""
    return _wrap_with_traceable("tool", func)
