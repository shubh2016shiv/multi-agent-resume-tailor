"""Public facade for LLM token tracking.

Use this package when code needs LLM token counting, budget checks, or usage logs:

    from src.core.llm_token_tracker import get_token_counter, ensure_token_budget

Module map:
- `llm_token_counter.py`: LiteLLM-backed LLM token and cost counter.
- `llm_token_usage.py`: immutable LLM token usage record.
- `exceptions.py`: LLM-token-specific exceptions.
- `llm_token_budget_guard.py`: explicit LLM token budget guard.
- `llm_token_tracking_context.py`: context manager for agent execution logging.
"""

from src.core.llm_token_tracker.exceptions import TokenBudgetExceeded
from src.core.llm_token_tracker.llm_token_budget_guard import ensure_token_budget
from src.core.llm_token_tracker.llm_token_counter import TokenCounter, get_token_counter
from src.core.llm_token_tracker.llm_token_tracking_context import track_agent_tokens
from src.core.llm_token_tracker.llm_token_usage import TokenUsage

__all__ = [
    "TokenBudgetExceeded",
    "TokenCounter",
    "TokenUsage",
    "ensure_token_budget",
    "get_token_counter",
    "track_agent_tokens",
]
