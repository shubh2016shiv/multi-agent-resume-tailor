"""Context manager for logging LLM token usage around agent execution."""

from collections.abc import Iterator
from contextlib import contextmanager

from src.core.llm_token_tracker.llm_token_counter import TokenCounter, get_token_counter
from src.core.logger import get_logger

logger = get_logger(__name__)


@contextmanager
def track_agent_tokens(
    agent_name: str, model: str, task_description: str
) -> Iterator[TokenCounter]:
    """Track estimated input tokens around an agent execution block.

    Args:
        agent_name: Agent being executed.
        model: Model name used by the agent.
        task_description: Prompt/task text sent to the agent.

    Yields:
        Shared token counter for optional usage logging inside the block.
    """
    counter = get_token_counter()
    input_tokens = counter.count_tokens(task_description, model)
    logger.info(
        "agent_execution_started",
        agent=agent_name,
        model=model,
        estimated_input_tokens=input_tokens,
    )
    try:
        yield counter
    finally:
        logger.debug("agent_execution_completed", agent=agent_name, model=model)
