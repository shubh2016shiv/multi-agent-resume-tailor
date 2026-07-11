"""Explicit token budget guards.

This is the package's main production entry point today — it's the pre-flight
check called before an LLM request goes out (see structured_output.py), so a
budget violation is caught before the request is sent, not after it fails.
"""

from src.core.llm_token_tracker.counter import get_token_counter  # shared measurement instance
from src.core.llm_token_tracker.exceptions import TokenBudgetExceeded


def ensure_token_budget(text: str, model: str, max_tokens: int) -> int:
    """Validate that text fits within an explicit token budget.

    Sequence: validate the budget argument itself, measure the text via the
    shared TokenCounter, then compare and raise if the budget is exceeded.

    Args:
        text: Text to measure.
        model: Model name to pass to the token counter.
        max_tokens: Maximum allowed tokens.

    Returns:
        Counted tokens.

    Raises:
        TokenBudgetExceeded: If counted tokens exceed max_tokens.
        ValueError: If max_tokens is negative.
    """
    ####################################################
    # STEP 1: REJECT AN INVALID BUDGET BEFORE DOING ANY WORK
    ####################################################
    if max_tokens < 0:
        raise ValueError("max_tokens must be non-negative")

    ####################################################
    # STEP 2: MEASURE THE TEXT AGAINST THE SHARED COUNTER
    ####################################################
    tokens = get_token_counter().count_tokens(text, model)

    ####################################################
    # STEP 3: RAISE IF THE MEASURED COUNT EXCEEDS THE BUDGET
    ####################################################
    if tokens > max_tokens:
        raise TokenBudgetExceeded(f"token budget exceeded: {tokens} > {max_tokens}")

    return tokens
