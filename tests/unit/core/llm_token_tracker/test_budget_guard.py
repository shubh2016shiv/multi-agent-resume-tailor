"""Unit tests for src/core/llm_token_tracker/budget_guard.py."""

import pytest

from src.core.llm_token_tracker import TokenBudgetExceeded, ensure_token_budget
from src.core.llm_token_tracker.counter import get_token_counter


class TestEnsureTokenBudget:
    """Tests for explicit token-budget enforcement."""

    def test_ensure_token_budget_returns_counted_tokens_within_budget(self, monkeypatch):
        """Contract: valid input returns the counted token total."""
        monkeypatch.setattr(
            "src.core.llm_token_tracker.counter.token_counter",
            lambda *, model, messages: 42,
        )
        monkeypatch.setattr(
            "src.core.llm_token_tracker.counter.cost_per_token",
            lambda *, model, prompt_tokens, completion_tokens: (0.0, 0.0),
        )
        get_token_counter.cache_clear()

        result = ensure_token_budget("resume summary", "gpt-4o", max_tokens=100)

        assert result == 42
        get_token_counter.cache_clear()

    def test_ensure_token_budget_raises_for_negative_max_tokens(self):
        """Contract: negative budgets are rejected explicitly."""
        with pytest.raises(ValueError, match="max_tokens must be non-negative"):
            ensure_token_budget("resume summary", "gpt-4o", max_tokens=-1)

    def test_ensure_token_budget_raises_when_count_exceeds_budget(self, monkeypatch):
        """Contract: counted tokens above the limit raise TokenBudgetExceeded."""
        monkeypatch.setattr(
            "src.core.llm_token_tracker.counter.token_counter",
            lambda *, model, messages: 101,
        )
        monkeypatch.setattr(
            "src.core.llm_token_tracker.counter.cost_per_token",
            lambda *, model, prompt_tokens, completion_tokens: (0.0, 0.0),
        )
        get_token_counter.cache_clear()

        with pytest.raises(TokenBudgetExceeded, match="token budget exceeded: 101 > 100"):
            ensure_token_budget("resume summary", "gpt-4o", max_tokens=100)
        get_token_counter.cache_clear()
