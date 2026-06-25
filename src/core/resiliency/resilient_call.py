"""Composed resilience wrapper for direct LLM provider calls."""

from __future__ import annotations

import functools
import time
import uuid
from collections.abc import Callable

import structlog
from pybreaker import CircuitBreakerError
from ratelimit import RateLimitException

from src.core.logger import get_logger
from src.core.resiliency.circuit_breaker import breaker_for
from src.core.resiliency.policy import ResiliencePolicy, resolve_policy
from src.core.resiliency.rate_limit import rate_checker_for
from src.core.resiliency.transient_retry import create_retry_decorator

logger = get_logger(__name__)


def _compose_resilience[**P, T](func: Callable[P, T], policy: ResiliencePolicy) -> Callable[P, T]:
    """Compose rate limiting, retry, and circuit breaking around one function."""

    ####################################################
    # STEP 1: BUILD THE INNERMOST RATE-LIMIT GATE#
    ####################################################
    @functools.wraps(func)
    def rate_limited(*args: P.args, **kwargs: P.kwargs) -> T:
        burn_token = rate_checker_for(policy.provider, policy.rate_limit_calls_per_minute)
        try:
            burn_token()
        except RateLimitException as exc:
            ####################################################
            # STEP 2: SLEEP ONE WINDOW SLOT WHEN THE RATE LIMIT IS HIT#
            ####################################################
            sleep_for = getattr(
                exc,
                "period_remaining",
                60.0 / policy.rate_limit_calls_per_minute,
            )
            logger.info(
                "rate_limit_hit",
                provider=policy.provider,
                sleep_seconds=round(sleep_for, 1),
            )
            time.sleep(sleep_for)
            burn_token()
        return func(*args, **kwargs)

    ####################################################
    # STEP 3: WRAP THE RATE GATE WITH TRANSIENT RETRY#
    ####################################################
    retried = create_retry_decorator(policy)(rate_limited)

    ####################################################
    # STEP 4: WRAP THE RETRIED CALL WITH THE PROVIDER BREAKER#
    ####################################################
    @functools.wraps(func)
    def guarded(*args: P.args, **kwargs: P.kwargs) -> T:
        breaker = breaker_for(policy.provider, policy)
        try:
            result = breaker.call(retried, *args, **kwargs)
            logger.info("llm_call_success", provider=policy.provider, fn=func.__name__)
            return result
        except CircuitBreakerError:
            logger.warning("circuit_open", provider=policy.provider, fn=func.__name__)
            raise
        except Exception:
            logger.warning("llm_call_failed", provider=policy.provider, fn=func.__name__)
            raise

    return guarded


def resilient_llm_call[**P, T](
    provider: str | None = None,
    max_attempts: int | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Wrap a direct LLM provider-call function with resilience layers."""
    ####################################################
    # STEP 1: RESOLVE THE FUNCTION'S FIXED RESILIENCE POLICY#
    ####################################################
    # Decorator arguments override config here, at decoration time, so the
    # wrapped function has one stable policy for its lifetime.
    policy = resolve_policy(provider, max_attempts)

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        ####################################################
        # STEP 2: COMPOSE THE INNER RESILIENCE STACK#
        ####################################################
        guarded = _compose_resilience(func, policy)

        ####################################################
        # STEP 3: BIND ONE CORRELATION ID FOR THE LOGICAL CALL#
        ####################################################
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            structlog.contextvars.bind_contextvars(correlation_id=str(uuid.uuid4())[:8])
            try:
                return guarded(*args, **kwargs)
            finally:
                ####################################################
                # STEP 4: ALWAYS UNBIND THE CORRELATION ID AFTERWARD#
                ####################################################
                structlog.contextvars.unbind_contextvars("correlation_id")

        return wrapper

    return decorator
