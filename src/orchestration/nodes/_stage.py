"""The @pipeline_stage decorator: one node's start, end, and duration in the log.

Applied to every node in this package except await_candidate_clarifications, which
raises GraphInterrupt to pause the run rather than completing normally.
"""

import time
from collections.abc import Callable
from functools import wraps
from typing import Protocol

from src.core.logger import get_logger
from src.orchestration.state import ResumeEnhancementPipelineState

logger = get_logger(__name__)


class NodeFunction(Protocol):
    """One graph node: whole state in, partial state out.

    A Protocol, not Callable[[State], dict], because LangGraph's add_node() calls
    the node with `state` as a keyword; a Callable alias would make it positional-only.
    """

    def __call__(self, state: ResumeEnhancementPipelineState) -> dict: ...


def pipeline_stage(stage: str) -> Callable[[NodeFunction], NodeFunction]:
    """Log one node's start and end, timing it with a monotonic clock.

    Emits pipeline_stage_started (stage, run_id) before the node and
    pipeline_stage_completed (stage, run_id, duration_ms) after it. A node with
    more to record logs its own event; these two stay uniform across all stages.

    A node that raises emits no completion event -- the runner logs the failure
    once for the whole run.

    Two things to know when reading the output:
      * structlog takes filename/func_name from the logging call, so both events
        report this module. Filter on `stage` to identify the node.
      * functools.wraps must stay: LangGraph inspects the node's signature to
        decide what to pass it, and the wrapper would otherwise hide it.
    """

    def decorate(node: NodeFunction) -> NodeFunction:
        @wraps(node)
        def instrumented_node(state: ResumeEnhancementPipelineState) -> dict:
            start_time = time.monotonic()
            logger.info("pipeline_stage_started", stage=stage, run_id=state["run_id"])
            result = node(state)
            logger.info(
                "pipeline_stage_completed",
                stage=stage,
                run_id=state["run_id"],
                duration_ms=round((time.monotonic() - start_time) * 1000),
            )
            return result

        return instrumented_node

    return decorate
