"""Optional labels so dashboard boxes have clear names.

Put ``@trace_agent`` above a function that runs a whole agent, or
``@trace_tool`` above a helper. LangSmith then shows a named box for that
function. LLM calls that happen inside can appear nested under that box
(when the automatic LiteLLM recorder from ``langsmith_backend`` is also on).

Are these used in production today?
-----------------------------------
No. They are public, tested helpers. Token/cost recording can still work from
LiteLLM alone. What you do not get without these decorators is the tidy
named parent box that groups calls by agent.

If tracing is off, or the LangSmith library is missing, these decorators
return your function unchanged — normal behavior, no crash.
"""

from __future__ import annotations

from typing import Literal

from src.core.logger import get_logger
from src.observability.langsmith_backend import is_observability_enabled

logger = get_logger(__name__)


def build_traced_function(run_type: Literal["chain", "tool"], func):
    """Return a LangSmith-wrapped function, or the original if tracing is off.

    Args:
        run_type: ``"chain"`` for a whole agent-style step; ``"tool"`` for a
            smaller helper.
        func: The function to wrap.

    Returns:
        Either ``func`` unchanged, or a wrapper that opens a named dashboard
        box whenever ``func`` runs.
    """
    ####################################################
    # STEP 1: LEAVE THE FUNCTION ALONE WHEN OBSERVABILITY IS OFF
    ####################################################
    # Ask is_observability_enabled() now — do not cache the answer at import,
    # because init usually runs later.
    if not is_observability_enabled():
        return func

    ####################################################
    # STEP 2: IMPORT LANGSMITH ONLY WHEN WE ACTUALLY NEED TO TRACE
    ####################################################
    # Importing only here keeps this package loadable even if langsmith is
    # not installed.
    try:
        from langsmith import traceable
    except ImportError:
        logger.warning("langsmith_import_failed", function=func.__name__)
        return func

    ####################################################
    # STEP 3: WRAP THE FUNCTION AS A NAMED LANGSMITH SPAN
    ####################################################
    # The dashboard box title is the function's name (func.__name__).
    return traceable(run_type=run_type, name=func.__name__)(func)


def trace_agent(func):
    """Decorator: show this function as a named agent-level box in LangSmith.

    Example::

        @trace_agent
        def run_experience_optimizer(...):
            ...
    """
    ####################################################
    # STEP 1: DELEGATE TO THE SHARED WRAPPER AS RUN TYPE "chain"
    ####################################################
    return build_traced_function("chain", func)


def trace_tool(func):
    """Decorator: show this helper as a named tool-level box in LangSmith.

    Example::

        @trace_tool
        def audit_summary(...):
            ...
    """
    ####################################################
    # STEP 1: DELEGATE TO THE SHARED WRAPPER AS RUN TYPE "tool"
    ####################################################
    return build_traced_function("tool", func)
