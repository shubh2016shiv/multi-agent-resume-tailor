"""LLM token measurement and guardrail capabilities.

How the pieces fit together (read counter.py first — everything else calls into it):

    counter.py         TokenCounter + get_token_counter() — the measurement core.
                        Everything below calls get_token_counter() rather than
                        constructing TokenCounter() directly.
        |
        +-- budget_guard.py    ensure_token_budget() — pre-flight check called
        |                      before an LLM request goes out. The main
        |                      production entry point today.
        |
        +-- tracking_context.py track_agent_tokens() — context manager for
                                estimated-token logging around one execution
                                block. Not wired into a production call site
                                yet (see that file's module docstring).

usage.py (TokenUsage) and exceptions.py (TokenBudgetExceeded) are the plain
data/error types the above pass around — no logic of their own.

SCOPE BOUNDARY — read this before assuming the package does more than it does:
This package MEASURES INPUT TOKENS ONLY. It never counts a response/completion
string. `output_tokens` (on TokenUsage, build_usage(), log_token_usage()) is a
plain parameter the caller must already know and supply — usually from the
real LLM response's own usage metadata — not something this package computes.
estimate_cost() DOES account for both input and output, but only if the caller
gives it both counts; it cannot estimate the cost of one side alone. The real,
provider-reported source of truth for both input and output tokens/cost is
``src/observability`` (LiteLLM -> LangSmith), not this package — see
tracking_context.py's module docstring for the full reasoning.
"""

from src.core.llm_token_tracker.budget_guard import ensure_token_budget
from src.core.llm_token_tracker.counter import TokenCounter, get_token_counter
from src.core.llm_token_tracker.exceptions import TokenBudgetExceeded
from src.core.llm_token_tracker.tracking_context import track_agent_tokens
from src.core.llm_token_tracker.usage import TokenUsage

__all__ = [
    "TokenBudgetExceeded",
    "TokenCounter",
    "TokenUsage",
    "ensure_token_budget",
    "get_token_counter",
    "track_agent_tokens",
]
