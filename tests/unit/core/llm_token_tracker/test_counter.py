"""Unit tests for src/core/llm_token_tracker/counter.py."""

from src.core.llm_token_tracker.counter import TokenCounter, get_token_counter


class TestTokenCounter:
    """Tests for token counting and cost estimation."""

    def test_count_tokens_with_empty_text_returns_zero(self):
        """Contract: empty text has zero tokens."""
        counter = TokenCounter()

        result = counter.count_tokens("", "gpt-4o")

        assert result == 0

    def test_count_message_tokens_returns_zero_when_litellm_is_unavailable(self, monkeypatch):
        """Contract: unavailable counting returns zero instead of raising."""
        monkeypatch.setattr("src.core.llm_token_tracker.counter.token_counter", None)
        monkeypatch.setattr("src.core.llm_token_tracker.counter.cost_per_token", None)
        counter = TokenCounter()

        result = counter.count_message_tokens([{"role": "user", "content": "hello"}], "gpt-4o")

        assert result == 0

    def test_count_message_tokens_returns_litellm_total_when_provider_succeeds(self, monkeypatch):
        """Contract: successful provider counting returns its token total."""
        monkeypatch.setattr(
            "src.core.llm_token_tracker.counter.token_counter",
            lambda *, model, messages: 17,
        )
        monkeypatch.setattr(
            "src.core.llm_token_tracker.counter.cost_per_token",
            lambda *, model, prompt_tokens, completion_tokens: (0.0, 0.0),
        )
        counter = TokenCounter()

        result = counter.count_message_tokens([{"role": "user", "content": "hello"}], "gpt-4o")

        assert result == 17

    def test_count_message_tokens_returns_zero_when_provider_raises(self, monkeypatch):
        """Contract: provider errors degrade to zero counted tokens."""
        def raise_token_error(*, model, messages):
            raise RuntimeError("tokenizer unavailable")

        monkeypatch.setattr("src.core.llm_token_tracker.counter.token_counter", raise_token_error)
        monkeypatch.setattr(
            "src.core.llm_token_tracker.counter.cost_per_token",
            lambda *, model, prompt_tokens, completion_tokens: (0.0, 0.0),
        )
        counter = TokenCounter()

        result = counter.count_message_tokens([{"role": "user", "content": "hello"}], "gpt-4o")

        assert result == 0

    def test_estimate_cost_returns_none_when_cost_estimation_is_unavailable(self, monkeypatch):
        """Contract: unavailable cost estimation returns None."""
        monkeypatch.setattr("src.core.llm_token_tracker.counter.token_counter", None)
        monkeypatch.setattr("src.core.llm_token_tracker.counter.cost_per_token", None)
        counter = TokenCounter()

        result = counter.estimate_cost(prompt_tokens=100, completion_tokens=20, model="gpt-4o")

        assert result is None

    def test_estimate_cost_returns_summed_cost_when_provider_succeeds(self, monkeypatch):
        """Contract: prompt and completion costs are summed into one USD estimate."""
        monkeypatch.setattr(
            "src.core.llm_token_tracker.counter.token_counter",
            lambda *, model, messages: 17,
        )
        monkeypatch.setattr(
            "src.core.llm_token_tracker.counter.cost_per_token",
            lambda *, model, prompt_tokens, completion_tokens: (0.25, 0.5),
        )
        counter = TokenCounter()

        result = counter.estimate_cost(prompt_tokens=100, completion_tokens=20, model="gpt-4o")

        assert result == 0.75

    def test_estimate_cost_returns_none_when_provider_raises(self, monkeypatch):
        """Contract: provider cost failures degrade to None."""
        def raise_cost_error(*, model, prompt_tokens, completion_tokens):
            raise RuntimeError("pricing unavailable")

        monkeypatch.setattr(
            "src.core.llm_token_tracker.counter.token_counter",
            lambda *, model, messages: 17,
        )
        monkeypatch.setattr("src.core.llm_token_tracker.counter.cost_per_token", raise_cost_error)
        counter = TokenCounter()

        result = counter.estimate_cost(prompt_tokens=100, completion_tokens=20, model="gpt-4o")

        assert result is None

    def test_build_usage_returns_token_usage_record_with_total_tokens(self, monkeypatch):
        """Contract: usage records expose the documented fields and derived total."""
        monkeypatch.setattr(
            "src.core.llm_token_tracker.counter.token_counter",
            lambda *, model, messages: 17,
        )
        monkeypatch.setattr(
            "src.core.llm_token_tracker.counter.cost_per_token",
            lambda *, model, prompt_tokens, completion_tokens: (0.25, 0.5),
        )
        counter = TokenCounter()

        result = counter.build_usage(
            agent_name="summary_writer",
            input_tokens=100,
            output_tokens=20,
            model="gpt-4o",
        )

        assert result.agent_name == "summary_writer"
        assert result.input_tokens == 100
        assert result.output_tokens == 20
        assert result.model == "gpt-4o"
        assert result.cost_usd == 0.75
        assert result.total_tokens == 120

    def test_get_token_counter_returns_the_process_wide_shared_instance(self, monkeypatch):
        """Contract: repeated calls return the shared cached counter instance."""
        monkeypatch.setattr(
            "src.core.llm_token_tracker.counter.token_counter",
            lambda *, model, messages: 17,
        )
        monkeypatch.setattr(
            "src.core.llm_token_tracker.counter.cost_per_token",
            lambda *, model, prompt_tokens, completion_tokens: (0.0, 0.0),
        )
        get_token_counter.cache_clear()

        first_counter = get_token_counter()
        second_counter = get_token_counter()

        assert first_counter is second_counter
        get_token_counter.cache_clear()
