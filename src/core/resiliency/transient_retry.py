"""Transient retry decorator factory for direct LLM calls."""

from __future__ import annotations

from collections.abc import Callable
from typing import ParamSpec, TypeVar, cast

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

from src.core.resiliency.policy import RETRYABLE_EXCEPTIONS, ResiliencePolicy

P = ParamSpec("P")
T = TypeVar("T")


def create_retry_decorator(policy: ResiliencePolicy) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Build the Tenacity retry decorator for one resolved policy."""
    ####################################################
    # STEP 1: DEFINE WHICH FAILURES COUNT AS TRANSIENT#
    ####################################################
    retry_rule = retry_if_exception_type(RETRYABLE_EXCEPTIONS)

    ####################################################
    # STEP 2: DEFINE HOW MANY TOTAL ATTEMPTS ARE ALLOWED#
    ####################################################
    stop_rule = stop_after_attempt(policy.attempt_limit)

    ####################################################
    # STEP 3: DEFINE THE BACKOFF STRATEGY WITH JITTER#
    ####################################################
    wait_rule = wait_exponential(
        multiplier=policy.retry_exponential_multiplier,
        min=policy.retry_initial_delay,
        max=policy.retry_max_delay,
    ) + wait_random(0, 1)

    ####################################################
    # STEP 4: RETURN THE COMPOSED TENACITY DECORATOR#
    ####################################################
    return cast(
        Callable[[Callable[P, T]], Callable[P, T]],
        retry(
            retry=retry_rule,
            stop=stop_rule,
            wait=wait_rule,
            reraise=True,
        ),
    )
