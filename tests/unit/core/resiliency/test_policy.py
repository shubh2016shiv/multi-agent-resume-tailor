"""Unit tests for src/core/resiliency/policy.py."""

from types import SimpleNamespace

from ratelimit import RateLimitException

from src.core.resiliency.policy import RETRYABLE_EXCEPTIONS, resolve_policy


class TestResolvePolicy:
    """Tests for resilience policy resolution."""

    def test_resolve_policy_reads_defaults_from_config(self, monkeypatch):
        """Contract: policy defaults come from central app config."""
        config = SimpleNamespace(
            llm=SimpleNamespace(
                provider="openai",
                resilience=SimpleNamespace(
                    retry_max_attempts=3,
                    retry_initial_delay=1.0,
                    retry_max_delay=60.0,
                    retry_exponential_multiplier=2.0,
                    circuit_breaker_failure_threshold=5,
                    circuit_breaker_timeout=60,
                    rate_limit_calls_per_minute=60,
                    timeout_seconds=30,
                ),
            )
        )
        monkeypatch.setattr("src.core.resiliency.policy.get_config", lambda: config)

        result = resolve_policy()

        assert result.provider == "openai"
        assert result.attempt_limit == 3
        assert result.retry_initial_delay == 1.0
        assert result.retry_max_delay == 60.0
        assert result.retry_exponential_multiplier == 2.0
        assert result.circuit_breaker_failure_threshold == 5
        assert result.circuit_breaker_timeout == 60
        assert result.rate_limit_calls_per_minute == 60
        assert result.timeout_seconds == 30

    def test_resolve_policy_allows_decorator_arguments_to_override_provider_and_attempts(
        self, monkeypatch
    ):
        """Contract: decorator overrides win over config for provider and max attempts."""
        config = SimpleNamespace(
            llm=SimpleNamespace(
                provider="openai",
                resilience=SimpleNamespace(
                    retry_max_attempts=3,
                    retry_initial_delay=1.0,
                    retry_max_delay=60.0,
                    retry_exponential_multiplier=2.0,
                    circuit_breaker_failure_threshold=5,
                    circuit_breaker_timeout=60,
                    rate_limit_calls_per_minute=60,
                    timeout_seconds=30,
                ),
            )
        )
        monkeypatch.setattr("src.core.resiliency.policy.get_config", lambda: config)

        result = resolve_policy(provider="gemini", max_attempts=5)

        assert result.provider == "gemini"
        assert result.attempt_limit == 5


class TestRetryableExceptions:
    """Tests for the retryable exception contract."""

    def test_retryable_exceptions_match_documented_transient_failures(self):
        """Contract: only timeout, connection, and rate-limit failures are retried."""
        assert RETRYABLE_EXCEPTIONS == (TimeoutError, ConnectionError, RateLimitException)
