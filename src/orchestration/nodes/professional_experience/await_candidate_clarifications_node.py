"""Graph node that pauses for missing candidate-owned experience facts."""

from langgraph.types import interrupt

from src.core.logger import get_logger
from src.orchestration.nodes.node_contract import StateUpdate
from src.orchestration.state import ResumeEnhancementPipelineState

logger = get_logger(__name__)


def await_candidate_clarifications(
    state: ResumeEnhancementPipelineState,
) -> StateUpdate:
    """Pause when experience questions exist and answers have not arrived."""
    clarifications = state.get("experience_clarifications") or []
    answers = state.get("clarification_answers") or []
    if not clarifications:
        return {}

    # LangGraph re-enters this node after resume. Checking answers before
    # interrupt() prevents the resumed execution from pausing a second time.
    if answers:
        logger.info(
            "candidate_clarifications_received",
            run_id=state["run_id"],
            answered_clarifications=len(answers),
        )
        return {}

    logger.info(
        "candidate_clarifications_requested",
        run_id=state["run_id"],
        clarification_count=len(clarifications),
    )
    interrupt(
        {
            "type": "candidate_clarifications_required",
            "questions": [item.model_dump(mode="json") for item in clarifications],
        }
    )
    return {}


__all__ = ["await_candidate_clarifications"]
