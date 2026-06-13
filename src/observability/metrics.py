"""Custom metric emission for agent iteration loops.

CrewAI + LangSmith automatically capture agent inputs, outputs, token usage,
and timing. This module adds custom domain metrics (quality scores, improvement
deltas, issues remaining) that LangSmith doesn't auto-capture.

Metrics always hit structlog first for local debuggability. When a LangSmith
run is active, the same metrics are attached to that run's metadata so they
appear on the trace. Never raises — metrics must never break a pipeline run.
"""

from __future__ import annotations

from typing import Any

from src.core.logger import get_logger
from src.observability.langsmith_init import is_observability_enabled

logger = get_logger(__name__)


def log_iteration_metrics(agent_name: str, iteration: int, metrics: dict[str, Any]) -> None:
    """Record per-iteration metrics (scores, tokens, cost, deltas).

    Always logs to structlog. When LangSmith is active, attaches the same
    metrics to the current trace's metadata. Never raises.
    """
    logger.info("iteration_metrics", agent=agent_name, iteration=iteration, **metrics)

    # Read live (function call), not a snapshotted import — see tracing.py.
    if not is_observability_enabled():
        return
    try:
        from langsmith.run_helpers import get_current_run_tree

        run = get_current_run_tree()
        if run is not None:
            run.metadata.update({"agent_name": agent_name, "iteration": iteration, **metrics})
    except Exception as exc:  # noqa: BLE001 — metrics must never break a run
        logger.warning("langsmith_metric_attach_failed", error=str(exc))
