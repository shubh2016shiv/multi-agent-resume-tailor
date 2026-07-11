"""Context manager for estimated token logging around one execution boundary.

IMPORTANT: this helper does NOT capture the real provider-reported token usage.
It only estimates input tokens locally from the prompt/task text you pass in.

In this repo, the true observability source of truth for LLM calls is
``src/observability``:
- LiteLLM -> LangSmith callback captures real prompt/completion tokens, cost,
  and latency for actual provider calls.
- ``trace_agent`` / ``trace_tool`` are the right place for readable span
  boundaries in observability.

So where would this helper belong if we use it?
- NOT inside individual files under ``src/agents/``.
  That would spread infrastructure concerns across every agent definition.
- ONLY at one execution choke point, such as:
  - ``src/orchestration/crew_task_execution.py:run_agent_task`` for CrewAI agent runs
  - ``src/tools/llm_gateway/structured_output.py`` for tool-owned direct LLM calls

Why it exists:
- to log an estimated input-token number before a block starts,
- to yield the shared counter into that block,
- and to always emit a completion log on block exit.

Current repo status:
- this helper is not wired into any production ``src/`` call site today.
- it is therefore optional infrastructure, not an active part of the pipeline.
"""

from collections.abc import Iterator
from contextlib import contextmanager

from src.core.llm_token_tracker.counter import TokenCounter, get_token_counter  # shared instance
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

    Notes:
        This is a local logging helper, not the authoritative observability path.
        Use it only at a shared execution boundary if you need estimated token
        context in structlog. Do not sprinkle it through individual agent files.
    """
    ####################################################
    # STEP 1: FETCH THE SHARED TOKEN COUNTER
    ####################################################
    # We reuse the process-wide counter so every caller goes through the same
    # provider-availability and graceful-degradation behavior.
    counter = get_token_counter()

    ####################################################
    # STEP 2: ESTIMATE THE INPUT TOKENS FOR THIS EXECUTION
    ####################################################
    # The purpose of this helper is to measure the prompt/task payload at the
    # boundary of the agent run before the wrapped block starts executing.
    input_tokens = counter.count_tokens(task_description, model)

    ####################################################
    # STEP 3: LOG THE START OF THE AGENT EXECUTION
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
        # STEP 4: YIELD CONTROL BACK TO THE CALLER'S EXECUTION BLOCK
        ####################################################
        # The caller does the real work inside the with-block and can optionally
        # reuse the returned counter to log final token usage later.
        yield counter
    finally:
        ####################################################
        # STEP 5: ALWAYS LOG THAT THE EXECUTION BLOCK FINISHED
        ####################################################
        # The finally block guarantees a completion log whether the wrapped work
        # succeeds or raises — even if the caller's code inside the with-block
        # throws, this line still runs before the exception propagates.
        logger.debug("agent_execution_completed", agent=agent_name, model=model)
