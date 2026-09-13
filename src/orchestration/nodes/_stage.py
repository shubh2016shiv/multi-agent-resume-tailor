"""Uniform stage instrumentation for graph nodes.

Every node opened and closed with the same timer-and-logger boilerplate:
start the monotonic clock, log pipeline_stage_started, do the work, compute
duration_ms, log pipeline_stage_completed. Eleven nodes x ~10 lines, and
rehydrate_pii repeated the completion half three times for its early returns.

@pipeline_stage("name") owns that pattern so a node's body is only its own work.
"""

import time
from collections.abc import Callable
from functools import wraps
from typing import Protocol

from src.core.logger import get_logger
from src.orchestration.state import ResumeEnhancementPipelineState

logger = get_logger(__name__)


class NodeFunction(Protocol):
    """One graph node: state in, partial state out.

    A Protocol rather than Callable[[...], dict] on purpose -- LangGraph's
    add_node() requires `state` to be callable BY NAME, and a Callable alias
    makes its parameters position-only, which add_node rejects.
    """

    def __call__(self, state: ResumeEnhancementPipelineState) -> dict: ...


def pipeline_stage(stage: str) -> Callable[[NodeFunction], NodeFunction]:
    """Log one node's start, completion, and wall-clock duration.

    Emits the same two events with the same fields the nodes emitted inline:
    pipeline_stage_started (stage, run_id) and pipeline_stage_completed
    (stage, run_id, duration_ms). A node that needs to record more than that
    logs its own domain event from inside its body.

    A node that raises logs no completion event, which is what the inline
    version did too (its completion log sat after the call that could raise).
    The run-level failure is logged once by the runner.

    Note on callsite metadata: structlog stamps filename/func_name from the
    logging call, so these two events now point at this module rather than at
    the node's own file. The `stage` field is the node identity to filter on,
    and it is unchanged.

    functools.wraps is required, not cosmetic: LangGraph inspects a node's
    signature to decide what to pass it, so the wrapper must not hide it.
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
