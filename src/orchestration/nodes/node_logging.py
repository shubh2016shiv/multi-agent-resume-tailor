"""Structured start, completion, and duration logging for pipeline nodes."""

import time
from collections.abc import Callable
from functools import wraps
from typing import TypeVar, cast

from src.core.logger import get_logger
from src.orchestration.nodes.node_contract import StateNode, StateUpdate
from src.orchestration.state import ResumeEnhancementPipelineState

logger = get_logger(__name__)

type ResumeTailorNode = StateNode[ResumeEnhancementPipelineState, StateUpdate]
NodeT = TypeVar("NodeT", bound=ResumeTailorNode)


def log_node_execution(node_name: str) -> Callable[[NodeT], NodeT]:
    """Log when a Resume Tailor node starts and finishes, including duration."""

    def decorate(node: NodeT) -> NodeT:
        @wraps(node)
        def logged_node(state: ResumeEnhancementPipelineState) -> StateUpdate:
            started_at = time.monotonic()
            logger.info("pipeline_stage_started", stage=node_name, run_id=state["run_id"])
            update = node(state)
            logger.info(
                "pipeline_stage_completed",
                stage=node_name,
                run_id=state["run_id"],
                duration_ms=round((time.monotonic() - started_at) * 1000),
            )
            return update

        return cast(NodeT, logged_node)

    return decorate


__all__ = ["log_node_execution"]
