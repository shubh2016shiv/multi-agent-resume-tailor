"""Policy resolution for direct LLM call resilience.

Turns the central config (plus any per-decorator overrides) into one frozen
ResiliencePolicy value object. Everything else in the package reads its limits
from that object, so config lookups happen here and nowhere else.
"""

from __future__ import annotations

from dataclasses import dataclass

from ratelimit import RateLimitException

from src.core.settings import get_config

# The exception types the retry layer treats as transient (worth retrying).
# TimeoutError/ConnectionError are transient network faults; RateLimitException
# means "slow down", which a retry after backoff can satisfy. Defined here
# because it is policy — what counts as retryable — consumed by transient_retry.py.
RETRYABLE_EXCEPTIONS = (TimeoutError, ConnectionError, RateLimitException)


@dataclass(frozen=True)
class ResiliencePolicy:
    """Resolved, immutable limits for one decorated provider-call function.

    Frozen so a function's policy can't drift after it's decorated — the whole
    point is that each wrapped function has one stable ruleset for its lifetime.
    """

    provider: str
    attempt_limit: int
    retry_initial_delay: float
    retry_max_delay: float
    retry_exponential_multiplier: float
    circuit_breaker_failure_threshold: int
    circuit_breaker_timeout: int
    rate_limit_calls_per_minute: int
    # NOTE: resolved and carried, but NOT enforced anywhere in this package yet
    # (no code cancels a hung call on this deadline). Config-only for now — see
    # README.md "Important limitation". Kept here so the wiring is ready if/when
    # real timeout cancellation is added.
    timeout_seconds: int


def resolve_policy(
    provider: str | None = None, max_attempts: int | None = None
) -> ResiliencePolicy:
    """Resolve one function's resilience policy from config plus decorator overrides.

    Precedence rule throughout: an explicit decorator argument (provider,
    max_attempts) wins; when it's None, fall back to the central config value.
    """
    ####################################################
    # STEP 1: READ THE CENTRAL LLM RESILIENCE SETTINGS
    ####################################################
    config = get_config()
    resilience = config.llm.resilience

    ####################################################
    # STEP 2: RESOLVE THE PROVIDER NAME FOR THIS DECORATED FUNCTION
    ####################################################
    # The provider name is the key every registry is scoped by (one breaker and
    # one rate checker per provider), so it must resolve to something concrete.
    provider_name = provider if provider is not None else config.llm.provider

    ####################################################
    # STEP 3: RESOLVE THE TOTAL ATTEMPT LIMIT
    ####################################################
    # Decorator argument overrides config, but we still clamp to at least 1 so
    # the first call always happens even if a caller passes 0 or a negative.
    attempt_limit = max(
        1,
        max_attempts if max_attempts is not None else resilience.retry_max_attempts,
    )

    ####################################################
    # STEP 4: FREEZE THE RESOLVED SETTINGS INTO ONE VALUE OBJECT
    ####################################################
    # The remaining settings are passed straight through from config — only
    # provider and attempt_limit above accept per-decorator overrides.
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
