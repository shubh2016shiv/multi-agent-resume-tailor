"""Context manager for token-aware agent execution logging.

Wrap a higher-level agent execution block with this context manager when you
want one small, reusable place to:
- estimate input tokens before the block runs,
- log that the agent execution started,
- hand the shared token counter to the block,
- and always log completion when the block exits.

Example:
    with track_agent_tokens("summary_writer", model, task_text) as counter:
        result = run_agent()
        counter.log_token_usage(...)

This helper is about execution-boundary logging, not about performing the LLM
call itself. The calling code still owns the actual agent or tool work.
"""

from collections.abc import Iterator
from contextlib import contextmanager

from src.core.llm_token_tracker.counter import TokenCounter, get_token_counter
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
        task_description: Prompt or task text whose input tokens should be estimated.

    Yields:
        Shared token counter for optional in-block usage logging.
    """
    ####################################################
    # STEP 1: FETCH THE SHARED TOKEN COUNTER#
    ####################################################
    # We reuse the process-wide counter so every caller goes through the same
    # provider-availability and graceful-degradation behavior.
    counter = get_token_counter()

    ####################################################
    # STEP 2: ESTIMATE THE INPUT TOKENS FOR THIS EXECUTION#
    ####################################################
    # The purpose of this helper is to measure the prompt/task payload at the
    # boundary of the agent run before the wrapped block starts executing.
    input_tokens = counter.count_tokens(task_description, model)

    ####################################################
    # STEP 3: LOG THE START OF THE AGENT EXECUTION#
    ####################################################
    # This gives the caller one standard structured event for "we are about to
    # run this agent with roughly this many input tokens."
    logger.info(
        "agent_execution_started",
        agent=agent_name,
        model=model,
        estimated_input_tokens=input_tokens,
    )
    try:
        ####################################################
        # STEP 4: YIELD CONTROL BACK TO THE CALLER'S EXECUTION BLOCK#
        ####################################################
        # The caller does the real work inside the with-block and can optionally
        # reuse the returned counter to log final token usage later.
        yield counter
    finally:
        ####################################################
        # STEP 5: ALWAYS LOG THAT THE EXECUTION BLOCK FINISHED#
        ####################################################
        # The finally block guarantees a completion log whether the wrapped work
        # succeeds or raises.
        logger.debug("agent_execution_completed", agent=agent_name, model=model)
