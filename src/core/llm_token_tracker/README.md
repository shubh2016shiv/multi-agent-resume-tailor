# LLM Token Tracker

This package tracks LLM token use: counting prompt/message tokens, estimating
LLM cost, enforcing LLM token budgets, and logging LLM token usage.

## What Belongs Here

- `llm_token_counter.py`: the LiteLLM adapter and `TokenCounter` class.
- `llm_token_usage.py`: immutable records such as `TokenUsage`.
- `exceptions.py`: LLM-token-specific exceptions.
- `llm_token_budget_guard.py`: guard functions such as `ensure_token_budget`.
- `llm_token_tracking_context.py`: lightweight context manager helpers for execution logging.
- `__init__.py`: public facade and import map.

## What Does Not Belong Here

- Prompt text.
- Agent business logic.
- YAML model names or runtime values.
- Retry/circuit-breaker logic.

Those belong in `src/config/`, `src/agents/`, or `src/core/resiliency/`.

## How To Use

```python
from src.core.llm_token_tracker import ensure_token_budget, get_token_counter

counter = get_token_counter()
tokens = counter.count_tokens("hello", "gpt-4o")
ensure_token_budget("hello", "gpt-4o", max_tokens=100)
```

The old import path `src.core.token_counter` is kept as a compatibility facade.
