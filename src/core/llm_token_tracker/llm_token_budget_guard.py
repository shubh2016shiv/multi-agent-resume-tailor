"""Explicit token budget guards."""

from src.core.llm_token_tracker.exceptions import TokenBudgetExceeded
from src.core.llm_token_tracker.llm_token_counter import get_token_counter


def ensure_token_budget(text: str, model: str, max_tokens: int) -> int:
    """Validate that text fits within an explicit token budget.

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
    if max_tokens < 0:
        raise ValueError("max_tokens must be non-negative")
    tokens = get_token_counter().count_tokens(text, model)
    if tokens > max_tokens:
        raise TokenBudgetExceeded(f"token budget exceeded: {tokens} > {max_tokens}")
    return tokens
