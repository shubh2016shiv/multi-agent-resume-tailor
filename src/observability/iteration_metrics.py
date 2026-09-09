"""Log custom numbers from retry / improve loops.

LiteLLM + LangSmith already record prompts, replies, tokens, and timing for
model calls. They do not know your app-specific scores (for example quality
score, how much it improved, how many issues remain). Use this helper to log
those numbers.

Behavior
--------
1. Always writes a local log line (structlog), even if LangSmith is off.
2. If LangSmith is on, also tries to attach the same numbers to the current
   dashboard box (only works if you are inside a ``@trace_agent`` /
   ``@trace_tool`` span).
3. Never raises — a metrics failure must not stop a resume run.

Used in production today?
-------------------------
No. Exported and tested; call it from a shared loop when you need charts,
rather than from every agent file.
"""

from typing import Any

from src.core.logger import get_logger
from src.observability.langsmith_backend import is_observability_enabled

logger = get_logger(__name__)


def log_iteration_metrics(agent_name: str, iteration: int, metrics: dict[str, Any]) -> None:
    """Record one iteration's custom metrics.

    Args:
        agent_name: Who produced these numbers (for logs and dashboards).
        iteration: Loop count, usually starting at 1.
        metrics: Any JSON-friendly fields, e.g.
            ``{"quality_score": 78, "improvement_delta": 13}``.

    Always logs locally. Optionally attaches to LangSmith when tracing is on
    and a current run exists. Never raises.
    """
    ####################################################
    # STEP 1: ALWAYS LOG THE METRICS LOCALLY FIRST
    ####################################################
    # Even without an API key, developers still see these numbers in logs.
    logger.info("iteration_metrics", agent=agent_name, iteration=iteration, **metrics)

    ####################################################
    # STEP 2: EXIT EARLY WHEN LANGSMITH TRACING IS OFF
    ####################################################
    # Ask live status each time (same reason as in tracing.py).
    if not is_observability_enabled():
        return

    ####################################################
    # STEP 3: ATTACH THE SAME METRICS TO THE CURRENT LANGSMITH RUN
    ####################################################
    # This only finds a run if we are inside an active @trace_agent /
    # @trace_tool box. If not, we already logged locally — that is fine.
    try:
        from langsmith.run_helpers import get_current_run_tree

        current_run = get_current_run_tree()
        if current_run is not None:
            current_run.metadata.update(
                {"agent_name": agent_name, "iteration": iteration, **metrics}
            )
    except Exception as exc:  # noqa: BLE001 — metrics must never break a run
        logger.warning("langsmith_metric_attach_failed", error=str(exc))
