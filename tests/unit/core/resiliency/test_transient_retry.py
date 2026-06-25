"""Unit tests for src/core/resiliency/transient_retry.py."""

import pytest
from tenacity import wait_fixed

from src.core.resiliency.policy import ResiliencePolicy
from src.core.resiliency.transient_retry import create_retry_decorator


def _make_policy(attempt_limit: int = 3) -> ResiliencePolicy:
    """Build a zero-wait policy fixture for retry tests."""
    return ResiliencePolicy(
        provider="openai",
        attempt_limit=attempt_limit,
        retry_initial_delay=0.0,
        retry_max_delay=0.0,
        retry_exponential_multiplier=0.0,
        circuit_breaker_failure_threshold=5,
        circuit_breaker_timeout=60,
        rate_limit_calls_per_minute=60,
        timeout_seconds=30,
    )


class TestCreateRetryDecorator:
    """Tests for transient retry behavior."""

    def test_create_retry_decorator_retries_transient_failures_until_success(self, monkeypatch):
        """Contract: transient connection errors are retried up to success."""
        monkeypatch.setattr(
            "src.core.resiliency.transient_retry.wait_random",
            lambda _min, _max: wait_fixed(0),
        )
        retry_decorator = create_retry_decorator(_make_policy(attempt_limit=3))
        attempts = {"count": 0}

        @retry_decorator
        def flaky_call():
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise ConnectionError("provider blip")
            return "ok"

        result = flaky_call()

        assert result == "ok"
        assert attempts["count"] == 3

    def test_create_retry_decorator_does_not_retry_non_retryable_failures(self, monkeypatch):
        """Contract: programming errors surface immediately."""
        monkeypatch.setattr(
            "src.core.resiliency.transient_retry.wait_random",
            lambda _min, _max: wait_fixed(0),
        )
        retry_decorator = create_retry_decorator(_make_policy(attempt_limit=3))
        attempts = {"count": 0}

        @retry_decorator
        def broken_call():
            attempts["count"] += 1
            raise ValueError("programming error")

        with pytest.raises(ValueError, match="programming error"):
            broken_call()

        assert attempts["count"] == 1

    def test_create_retry_decorator_respects_the_attempt_limit(self, monkeypatch):
        """Contract: retry stops after the configured total attempts."""
        monkeypatch.setattr(
            "src.core.resiliency.transient_retry.wait_random",
            lambda _min, _max: wait_fixed(0),
        )
        retry_decorator = create_retry_decorator(_make_policy(attempt_limit=2))
        attempts = {"count": 0}

        @retry_decorator
        def always_fails():
            attempts["count"] += 1
            raise ConnectionError("still failing")

        with pytest.raises(ConnectionError, match="still failing"):
            always_fails()

        assert attempts["count"] == 2
