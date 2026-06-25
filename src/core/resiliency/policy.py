"""Policy resolution for direct LLM call resilience."""

from __future__ import annotations

from dataclasses import dataclass

from ratelimit import RateLimitException

from src.core.settings import get_config

RETRYABLE_EXCEPTIONS = (TimeoutError, ConnectionError, RateLimitException)


@dataclass(frozen=True)
class ResiliencePolicy:
    """Resolved configuration for one decorated provider-call function."""

    provider: str
    attempt_limit: int
    retry_initial_delay: float
    retry_max_delay: float
    retry_exponential_multiplier: float
    circuit_breaker_failure_threshold: int
    circuit_breaker_timeout: int
    rate_limit_calls_per_minute: int
    timeout_seconds: int


def resolve_policy(
    provider: str | None = None, max_attempts: int | None = None
) -> ResiliencePolicy:
    """Resolve one function's resilience policy from config plus decorator overrides."""
    ####################################################
    # STEP 1: READ THE CENTRAL LLM RESILIENCE SETTINGS#
    ####################################################
    config = get_config()
    resilience = config.llm.resilience

    ####################################################
    # STEP 2: RESOLVE THE PROVIDER NAME FOR THIS DECORATED FUNCTION#
    ####################################################
    provider_name = provider if provider is not None else config.llm.provider

    ####################################################
    # STEP 3: RESOLVE THE TOTAL ATTEMPT LIMIT#
    ####################################################
    # Decorator arguments override config, but we still clamp the result to at
    # least one attempt so the first call always happens.
    attempt_limit = max(
        1,
        max_attempts if max_attempts is not None else resilience.retry_max_attempts,
    )

    ####################################################
    # STEP 4: FREEZE THE RESOLVED SETTINGS INTO ONE VALUE OBJECT#
    ####################################################
    return ResiliencePolicy(
        provider=provider_name,
        attempt_limit=attempt_limit,
        retry_initial_delay=resilience.retry_initial_delay,
        retry_max_delay=resilience.retry_max_delay,
        retry_exponential_multiplier=resilience.retry_exponential_multiplier,
        circuit_breaker_failure_threshold=resilience.circuit_breaker_failure_threshold,
        circuit_breaker_timeout=resilience.circuit_breaker_timeout,
        rate_limit_calls_per_minute=resilience.rate_limit_calls_per_minute,
        timeout_seconds=resilience.timeout_seconds,
    )
