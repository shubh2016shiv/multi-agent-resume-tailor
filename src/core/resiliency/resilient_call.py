"""Composed resilience wrapper for direct LLM provider calls.

This file is where the package's four building blocks come together into one
decorator. Each block lives in its own file and is combined here:
    policy.py           -> what the limits ARE (resolved config)
    rate_limit.py       -> ratelimit-backed per-provider budget gate
    transient_retry.py  -> tenacity-backed retry + backoff
    circuit_breaker.py  -> pybreaker-backed per-provider breaker
See _compose_resilience() below for how they nest, and README.md for the
bounded scope of this package.
"""

from __future__ import annotations

import functools
import time
import uuid
from collections.abc import Callable

import structlog  # for the per-call correlation_id bound into every log line
from pybreaker import CircuitBreakerError  # raised by the breaker when it is open
from ratelimit import RateLimitException  # raised by the rate gate when the budget is spent

from src.core.logger import get_logger
from src.core.resiliency.circuit_breaker import breaker_for  # LAYER 3 factory
from src.core.resiliency.policy import ResiliencePolicy, resolve_policy  # the resolved limits
from src.core.resiliency.rate_limit import rate_checker_for  # LAYER 1 factory
from src.core.resiliency.transient_retry import create_retry_decorator  # LAYER 2 factory

logger = get_logger(__name__)


def _compose_resilience[**P, T](func: Callable[P, T], policy: ResiliencePolicy) -> Callable[P, T]:
    """Wrap `func` in three nested resilience layers and return the outer wrapper.

    This is the coordination core of the package. The three protections are
    built as an ONION — each layer wraps the one below it, so the runtime order
    is determined by the *nesting*, not by the order of the STEP comments:

        guarded          (circuit breaker)   <- outermost, runs first
          └─ retried     (retry + backoff)
               └─ rate_limited (rate gate)
                    └─ func  (the real provider call)   <- innermost, runs last

    Read the layers below inside-out (rate gate first) because each one is
    built before the layer that wraps it — but at call time they execute
    outside-in (breaker first). The nesting order is deliberate:

    - Rate gate is INNERMOST, so every single attempt — including each retry —
      spends from the same per-minute budget. If it were outside retry, a burst
      of retries could blow past the provider's rate limit.
    - Retry is in the MIDDLE, so only the rate-passed call is retried, and each
      retry re-enters the rate gate above.
    - Breaker is OUTERMOST, so it observes ONE final outcome per logical call
      (success, or failure after all retries are exhausted) — NOT one event per
      retry attempt. This is what stops a single flaky call from tripping the
      breaker on its own, and lets an already-open breaker fast-fail before any
      retry or rate-limit work happens.
    """

    ####################################################
    # LAYER 1 (innermost): RATE-LIMIT GATE
    # Runs on every attempt. Spends one token from the provider's per-minute
    # budget before letting the real call through.
    ####################################################
    @functools.wraps(func)
    def rate_limited(*args: P.args, **kwargs: P.kwargs) -> T:
        burn_token = rate_checker_for(policy.provider, policy.rate_limit_calls_per_minute)
        try:
            burn_token()
        except RateLimitException as exc:
            # Budget exhausted for this window. Self-heal by sleeping until the
            # window rolls over, then spend one token and proceed — cheaper than
            # bouncing all the way out to the retry layer for an expected,
            # non-error condition. (RateLimitException is also excluded from the
            # breaker in circuit_breaker.py, so a rate hit never trips it.)
            sleep_for = getattr(
                exc,
                "period_remaining",  # ratelimit sets this to the seconds left in the window
                60.0 / policy.rate_limit_calls_per_minute,  # fallback: one even slot
            )
            logger.info(
                "rate_limit_hit",
                provider=policy.provider,
                sleep_seconds=round(sleep_for, 1),
            )
            time.sleep(sleep_for)
            burn_token()  # if this still raises, it propagates up to the retry layer
        return func(*args, **kwargs)

    ####################################################
    # LAYER 2 (middle): TRANSIENT RETRY
    # Wraps the rate gate. Retries only the exceptions policy deems transient
    # (see RETRYABLE_EXCEPTIONS), with exponential backoff + jitter.
    ####################################################
    retried = create_retry_decorator(policy)(rate_limited)

    ####################################################
    # LAYER 3 (outermost): CIRCUIT BREAKER
    # Wraps the retried call. breaker.call() records ONE outcome per logical
    # call — so N retries count as a single failure, not N — and short-circuits
    # (raises CircuitBreakerError) once the provider looks broken.
    ####################################################
    @functools.wraps(func)
    def guarded(*args: P.args, **kwargs: P.kwargs) -> T:
        breaker = breaker_for(policy.provider, policy)
        try:
            result = breaker.call(retried, *args, **kwargs)
            logger.info("llm_call_success", provider=policy.provider, fn=func.__name__)
            return result
        except CircuitBreakerError:
            # Breaker is open: we failed fast without even attempting the call.
            logger.warning("circuit_open", provider=policy.provider, fn=func.__name__)
            raise
        except Exception:
            # Real call failure after retries were exhausted; the breaker has
            # already counted it. Re-raise so the caller sees the true error.
            logger.warning("llm_call_failed", provider=policy.provider, fn=func.__name__)
            raise

    return guarded


def resilient_llm_call[**P, T](
    provider: str | None = None,
    max_attempts: int | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator: wrap a direct LLM provider-call function with resilience layers.

    Usage:
        @resilient_llm_call(provider="gemini", max_attempts=5)
        def call_provider(...): ...

    Two distinct times matter here:
    - DECORATION TIME (once, when Python applies the decorator): the policy is
      resolved and the resilience onion is composed. `provider`/`max_attempts`
      override config now and are then fixed for the wrapped function's life —
      later config changes do NOT re-resolve it.
    - CALL TIME (every invocation): `wrapper` runs, binding a fresh correlation
      ID so every log line from this one logical call shares one traceable id.
    """
    ####################################################
    # STEP 1: RESOLVE THE FUNCTION'S FIXED RESILIENCE POLICY (decoration time)
    ####################################################
    # Decorator arguments override config here, once, so the wrapped function
    # has one stable, immutable policy for its lifetime.
    policy = resolve_policy(provider, max_attempts)

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        ####################################################
        # STEP 2: COMPOSE THE INNER RESILIENCE STACK (decoration time)
        ####################################################
        # Build the breaker/retry/rate-limit onion once; `guarded` is reused on
        # every call rather than rebuilt each time.
        guarded = _compose_resilience(func, policy)

        ####################################################
        # STEP 3: BIND ONE CORRELATION ID PER LOGICAL CALL (call time)
        ####################################################
        # Bound at the OUTERMOST layer so it's present for every log emitted by
        # the breaker, retry, and rate-limit layers underneath during this call.
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            structlog.contextvars.bind_contextvars(correlation_id=str(uuid.uuid4())[:8])
            try:
                return guarded(*args, **kwargs)
            finally:
                ####################################################
                # STEP 4: ALWAYS UNBIND THE CORRELATION ID AFTERWARD
                ####################################################
                # finally guarantees cleanup even if the call raised, so the id
                # never leaks into an unrelated call on this same thread.
                structlog.contextvars.unbind_contextvars("correlation_id")

        return wrapper

    return decorator
