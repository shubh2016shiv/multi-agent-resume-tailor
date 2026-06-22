"""Unit tests for src/core/resiliency/circuit_breaker.py."""

from ratelimit import RateLimitException

from src.core.resiliency.circuit_breaker import (
    breaker_for,
    get_resilience_stats,
    reset_circuit_breakers,
)
from src.core.resiliency.policy import ResiliencePolicy


def _make_policy(provider: str = "openai") -> ResiliencePolicy:
    """Build a small policy fixture for breaker tests."""
    return ResiliencePolicy(
        provider=provider,
        attempt_limit=3,
        retry_initial_delay=0.0,
        retry_max_delay=0.0,
        retry_exponential_multiplier=0.0,
        circuit_breaker_failure_threshold=5,
        circuit_breaker_timeout=60,
        rate_limit_calls_per_minute=60,
        timeout_seconds=30,
    )


class TestCircuitBreakerRegistry:
    """Tests for provider-scoped circuit-breaker state."""

    def test_breaker_for_reuses_the_same_breaker_for_one_provider(self):
        """Contract: one provider shares one breaker instance."""
        policy = _make_policy("openai")

        first_breaker = breaker_for("openai", policy)
        second_breaker = breaker_for("openai", policy)

        assert first_breaker is second_breaker

    def test_breaker_for_creates_distinct_breakers_for_different_providers(self):
        """Contract: different providers do not share breaker state."""
        openai_breaker = breaker_for("openai", _make_policy("openai"))
        gemini_breaker = breaker_for("gemini", _make_policy("gemini"))

        assert openai_breaker is not gemini_breaker

    def test_breaker_for_excludes_rate_limit_failures_from_breaker_counting(self):
        """Contract: rate-limit exhaustion is not counted as a provider outage."""
        breaker = breaker_for("openai", _make_policy("openai"))

        try:
            breaker.call(lambda: (_ for _ in ()).throw(RateLimitException("limit", 12)))
        except RateLimitException:
            pass

        assert breaker.fail_counter == 0
        assert breaker.current_state == "closed"

    def test_reset_circuit_breakers_clears_the_shared_registry(self):
        """Contract: resetting breakers gives the next lookup a fresh instance."""
        original_breaker = breaker_for("openai", _make_policy("openai"))

        reset_circuit_breakers()
        reset_breaker = breaker_for("openai", _make_policy("openai"))

        assert original_breaker is not reset_breaker

    def test_get_resilience_stats_returns_a_stable_documented_shape(self):
        """Contract: stats expose stable keys for the requested provider."""
        breaker_for("openai", _make_policy("openai"))

        result = get_resilience_stats("openai")

        assert result["provider"] == "openai"
        assert result["circuit_breaker_state"] == "closed"
        assert result["failure_count"] == 0
        assert result["failure_threshold"] == 5
        assert result["reset_timeout_seconds"] == 60
        assert "opened_at" in result
