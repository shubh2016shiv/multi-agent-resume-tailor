"""Unit tests for src/core/resiliency/resilient_call.py."""

from unittest.mock import MagicMock

import pytest
import structlog

from src.core.resiliency.policy import ResiliencePolicy
from src.core.resiliency.resilient_call import resilient_llm_call


def _make_policy() -> ResiliencePolicy:
    """Build a small fixed policy for wrapper-composition tests."""
    return ResiliencePolicy(
        provider="openai",
        attempt_limit=1,
        retry_initial_delay=0.0,
        retry_max_delay=0.0,
        retry_exponential_multiplier=0.0,
        circuit_breaker_failure_threshold=5,
        circuit_breaker_timeout=60,
        rate_limit_calls_per_minute=60,
        timeout_seconds=30,
    )


class TestResilientLlmCall:
    """Tests for the public resilience decorator."""

    def test_resilient_llm_call_preserves_wrapped_function_metadata(self, monkeypatch):
        """Contract: the outer wrapper keeps the original function name."""
        monkeypatch.setattr(
            "src.core.resiliency.resilient_call.resolve_policy", lambda p, m: _make_policy()
        )

        def sample_call():
            return "ok"

        decorated = resilient_llm_call()(sample_call)

        assert decorated.__name__ == "sample_call"

    def test_resilient_llm_call_binds_one_correlation_id_and_unbinds_afterward(self, monkeypatch):
        """Contract: one logical call gets one correlation id, then leaves no residue."""
        monkeypatch.setattr(
            "src.core.resiliency.resilient_call.resolve_policy", lambda p, m: _make_policy()
        )
        structlog.contextvars.clear_contextvars()

        @resilient_llm_call()
        def inspect_context():
            return structlog.contextvars.get_contextvars().get("correlation_id")

        correlation_id = inspect_context()

        assert correlation_id is not None
        assert "correlation_id" not in structlog.contextvars.get_contextvars()

    def test_resilient_llm_call_logs_success_and_failure_paths(self, monkeypatch):
        """Contract: wrapper logs final success and failure outcomes."""
        monkeypatch.setattr(
            "src.core.resiliency.resilient_call.resolve_policy", lambda p, m: _make_policy()
        )
        mock_logger = MagicMock()
        monkeypatch.setattr("src.core.resiliency.resilient_call.logger", mock_logger)

        @resilient_llm_call()
        def successful_call():
            return "ok"

        @resilient_llm_call()
        def failing_call():
            raise ValueError("boom")

        assert successful_call() == "ok"
        with pytest.raises(ValueError, match="boom"):
            failing_call()

        mock_logger.info.assert_any_call(
            "llm_call_success", provider="openai", fn="successful_call"
        )
        mock_logger.warning.assert_any_call("llm_call_failed", provider="openai", fn="failing_call")

    def test_resilient_llm_call_composes_rate_gate_then_retry_then_breaker(self, monkeypatch):
        """Contract: runtime order is breaker -> retry -> rate gate -> function body."""
        monkeypatch.setattr(
            "src.core.resiliency.resilient_call.resolve_policy", lambda p, m: _make_policy()
        )
        sequence: list[str] = []

        def fake_rate_checker_for(provider: str, calls_per_minute: int):
            def burn_token():
                sequence.append("rate")

            return burn_token

        def fake_create_retry_decorator(policy: ResiliencePolicy):
            def decorator(func):
                def wrapped(*args, **kwargs):
                    sequence.append("retry")
                    return func(*args, **kwargs)

                return wrapped

            return decorator

        class FakeBreaker:
            def call(self, func, *args, **kwargs):
                sequence.append("breaker")
                return func(*args, **kwargs)

        monkeypatch.setattr(
            "src.core.resiliency.resilient_call.rate_checker_for", fake_rate_checker_for
        )
        monkeypatch.setattr(
            "src.core.resiliency.resilient_call.create_retry_decorator",
            fake_create_retry_decorator,
        )
        monkeypatch.setattr(
            "src.core.resiliency.resilient_call.breaker_for",
            lambda provider, policy: FakeBreaker(),
        )
        monkeypatch.setattr("src.core.resiliency.resilient_call.logger", MagicMock())

        @resilient_llm_call()
        def sample_call():
            sequence.append("function")
            return "ok"

        assert sample_call() == "ok"
        assert sequence == ["breaker", "retry", "rate", "function"]
