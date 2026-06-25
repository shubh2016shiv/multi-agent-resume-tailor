"""Unit tests for src/core/resiliency/rate_limit.py."""

from src.core.resiliency.rate_limit import rate_checker_for, reset_rate_limiters


class TestRateCheckerRegistry:
    """Tests for provider-scoped rate-limit checker reuse."""

    def test_rate_checker_for_reuses_the_same_checker_for_one_provider(self):
        """Contract: one provider shares one rate-limit checker."""
        first_checker = rate_checker_for("openai", 60)
        second_checker = rate_checker_for("openai", 60)

        assert first_checker is second_checker

    def test_rate_checker_for_creates_distinct_checkers_for_different_providers(self):
        """Contract: different providers do not share rate-limit counters."""
        openai_checker = rate_checker_for("openai", 60)
        gemini_checker = rate_checker_for("gemini", 60)

        assert openai_checker is not gemini_checker

    def test_reset_rate_limiters_clears_the_shared_registry(self):
        """Contract: reset gives the next lookup a fresh checker."""
        original_checker = rate_checker_for("openai", 60)

        reset_rate_limiters()
        reset_checker = rate_checker_for("openai", 60)

        assert original_checker is not reset_checker
