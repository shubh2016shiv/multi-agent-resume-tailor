"""Resilience for LLM provider calls — retry, circuit breaker, rate limiting.

Usage::

    from src.core.resiliency import resilient_llm_call

    @resilient_llm_call(provider="gemini")
    def call_gemini(...):
        ...

Every decorated call gets:
- Exponential-backoff retry with jitter — transient errors only.
- Per-provider circuit breaker — fails fast after N consecutive failures.
  Rate-limit exhaustion does NOT count as a failure (it is not a provider outage).
- Per-provider rate limiting — all functions on the same provider share one bucket.
- One correlation ID per logical call, stable across all retry attempts.

Decorator arguments override ``settings.yaml → llm.resilience``.
"""

from __future__ import annotations

import functools
import time
import uuid
from collections.abc import Callable

import structlog
from pybreaker import CircuitBreaker, CircuitBreakerError
from ratelimit import RateLimitException, limits
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

from src.core.logger import get_logger
from src.core.settings import get_config

logger = get_logger(__name__)

# Module-level registries — one breaker and one rate-limit bucket per provider,
# shared across every decorated function that names the same provider.
_breakers: dict[str, CircuitBreaker] = {}
_rate_checkers: dict[str, Callable[[], None]] = {}  # provider -> @limits-decorated callable


# ── Provider-scoped singletons ────────────────────────────────────────────────


def breaker_for(provider: str) -> CircuitBreaker:
    if provider not in _breakers:
        resilience_config = get_config().llm.resilience
        _breakers[provider] = CircuitBreaker(
            fail_max=resilience_config.circuit_breaker_failure_threshold,
            reset_timeout=resilience_config.circuit_breaker_timeout,
            exclude=[RateLimitException],
        )
        logger.info("circuit_breaker_created", provider=provider)
    return _breakers[provider]


def rate_checker_for(provider: str, rpm: int):
    """Return a no-arg callable that burns one token from the provider's rate bucket.

    Separating the *check* from the *call* means we never need to pass functions
    as arguments — no ``fn(*args, **kwargs)`` noise inside a rate-limiter.
    """
    if provider in _rate_checkers:
        return _rate_checkers[provider]

    @limits(calls=rpm, period=60)
    def burn_token():
        pass  # existence is enough; the decorator does the counting

    _rate_checkers[provider] = burn_token
    return burn_token


# ── Public decorator ──────────────────────────────────────────────────────────


def resilient_llm_call(provider: str | None = None, max_attempts: int | None = None):
    """Wrap an LLM-calling function with retry, circuit breaker, and rate limiting.

    Args:
        provider:     Provider name — controls which breaker / rate bucket is used.
                      Defaults to ``settings.yaml → llm.provider``.
        max_attempts: Total attempts including the first call (minimum 1).
                      Defaults to ``settings.yaml → llm.resilience.retry_max_attempts``.
    """
    config = get_config()
    resilience = config.llm.resilience

    provider_name = provider if provider is not None else config.llm.provider
    attempt_limit = max(
        1,
        max_attempts if max_attempts is not None else resilience.retry_max_attempts,
    )
    calls_per_minute = resilience.rate_limit_calls_per_minute

    breaker = breaker_for(provider_name)
    burn_token = rate_checker_for(provider_name, calls_per_minute)

    def decorator(func):
        # Layer 1 — innermost: enforce rate limit, then call the real function.
        @functools.wraps(func)
        def rate_limited(*args, **kwargs):
            try:
                burn_token()
            except RateLimitException as exc:
                sleep_for = getattr(exc, "period_remaining", 60.0 / calls_per_minute)
                logger.info(
                    "rate_limit_hit",
                    provider=provider_name,
                    sleep_seconds=round(sleep_for, 1),
                )
                time.sleep(sleep_for)
                burn_token()
            return func(*args, **kwargs)

        # Layer 2 — retry transient provider/rate-limit errors with backoff + jitter.
        retried = retry(
            retry=retry_if_exception_type((TimeoutError, ConnectionError, RateLimitException)),
            stop=stop_after_attempt(attempt_limit),
            wait=wait_exponential(
                multiplier=resilience.retry_exponential_multiplier,
                min=resilience.retry_initial_delay,
                max=resilience.retry_max_delay,
            )
            + wait_random(0, 1),
            reraise=True,
        )(rate_limited)

        # Layer 3 — count one circuit-breaker result per logical call.
        def guarded(*args, **kwargs):
            try:
                result = breaker.call(retried, *args, **kwargs)
                logger.info("llm_call_success", provider=provider_name, fn=func.__name__)
                return result
            except CircuitBreakerError:
                logger.warning("circuit_open", provider=provider_name, fn=func.__name__)
                raise
            except Exception:
                logger.warning("llm_call_failed", provider=provider_name, fn=func.__name__)
                raise

        # Layer 4 — outermost: bind one correlation ID for the whole logical call.
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            structlog.contextvars.bind_contextvars(correlation_id=str(uuid.uuid4())[:8])
            try:
                return guarded(*args, **kwargs)
            finally:
                structlog.contextvars.unbind_contextvars("correlation_id")

        return wrapper

    return decorator
