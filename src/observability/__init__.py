"""What this package does, in plain words.

When agents talk to an LLM, you usually cannot see what was sent, what came
back, how many tokens it used, or what it cost. This package turns that
recording on (when configured) and sends it to LangSmith — a web dashboard
for LLM apps.

Import from here only::

    from src.observability import init_observability, trace_agent, ...

Do not import the files inside this folder from the rest of the app. That way,
if we ever switch away from LangSmith, only this package changes.

Two kinds of recording
----------------------
1. Automatic (already turned on in production): after startup succeeds, every
   LLM call that goes through LiteLLM can show up in LangSmith with prompt,
   reply, tokens, cost, and timing. You do not decorate each agent for this.
   Startup runs from ``src/orchestration/runner.py``.

2. Named labels (ready, but not used in production yet): put ``@trace_agent``
   or ``@trace_tool`` on a function if you want a clearly named box in the
   dashboard tree. ``log_iteration_metrics`` can attach custom scores to that
   box. These helpers exist and are tested; nothing in production calls them
   today.

Safety rule: if the API key is missing, tracing is disabled in config, or a
library is unavailable, these functions quietly do nothing. The resume
pipeline still runs.
"""

from src.observability.iteration_metrics import log_iteration_metrics
from src.observability.langsmith_backend import init_observability, is_observability_enabled
from src.observability.tracing import trace_agent, trace_tool

__all__ = [
    "init_observability",
    "trace_agent",
    "trace_tool",
    "log_iteration_metrics",
    "is_observability_enabled",
]
