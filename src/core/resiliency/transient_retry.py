"""Transient retry decorator factory for direct LLM calls.

Builds LAYER 2 of the resilience onion (see resilient_call.py). Tenacity's
retry() is configured from three independent rules — WHICH errors to retry,
WHEN to stop, and HOW LONG to wait between tries — assembled here from the
resolved policy so the rest of the package never touches tenacity directly.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import ParamSpec, TypeVar, cast

from tenacity import (
    retry,
    retry_if_exception_type,  # WHICH: only retry these exception types
    stop_after_attempt,  # WHEN to stop: after N total tries
    wait_exponential,  # HOW LONG: exponential backoff
    wait_random,  # HOW LONG: added jitter, see STEP 3
)

from src.core.resiliency.policy import RETRYABLE_EXCEPTIONS, ResiliencePolicy

# ParamSpec/TypeVar preserve the wrapped function's exact signature through the
# decorator, so a decorated function keeps its real arg/return types.
P = ParamSpec("P")
T = TypeVar("T")


def create_retry_decorator(policy: ResiliencePolicy) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Build the Tenacity retry decorator for one resolved policy."""
    ####################################################
    # STEP 1: DEFINE WHICH FAILURES COUNT AS TRANSIENT
    ####################################################
    # Only these are worth retrying (timeouts, dropped connections, rate hits);
    # anything else — a bad request, an auth error — fails immediately instead
    # of wasting attempts on an error that won't fix itself.
    retry_rule = retry_if_exception_type(RETRYABLE_EXCEPTIONS)

    ####################################################
    # STEP 2: DEFINE HOW MANY TOTAL ATTEMPTS ARE ALLOWED
    ####################################################
    # attempt_limit counts the FIRST try plus retries (already clamped to >= 1
    # in policy.resolve_policy), so a limit of 3 means 1 call + 2 retries.
    stop_rule = stop_after_attempt(policy.attempt_limit)

    ####################################################
    # STEP 3: DEFINE THE BACKOFF STRATEGY WITH JITTER
    ####################################################
    # Exponential backoff: each wait grows (initial_delay, x2, x4, ...) up to
    # max_delay, so a struggling provider gets progressively more breathing room.
    # The extra wait_random(0, 1) is JITTER — without it, many callers that
    # failed at the same instant would all retry at the same instant (a
    # "thundering herd") and hammer the provider in sync. The random 0-1s
    # spreads those retries out.
    wait_rule = wait_exponential(
        multiplier=policy.retry_exponential_multiplier,
        min=policy.retry_initial_delay,
        max=policy.retry_max_delay,
    ) + wait_random(0, 1)

    ####################################################
    # STEP 4: RETURN THE COMPOSED TENACITY DECORATOR
    ####################################################
    # reraise=True: after the last attempt fails, raise the ORIGINAL provider
    # exception, not tenacity's RetryError wrapper — so the circuit breaker and
    # caller above see the true error type.
    return cast(
        Callable[[Callable[P, T]], Callable[P, T]],
        retry(
            retry=retry_rule,
            stop=stop_rule,
            wait=wait_rule,
            reraise=True,
        ),
    )
