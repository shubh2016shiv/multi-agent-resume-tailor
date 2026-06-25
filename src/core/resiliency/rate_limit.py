"""Provider-scoped rate-limit checkers for direct LLM calls."""

from __future__ import annotations

from collections.abc import Callable

from ratelimit import limits

_rate_checkers: dict[str, Callable[[], None]] = {}


def rate_checker_for(provider: str, calls_per_minute: int) -> Callable[[], None]:
    """Return the shared per-provider rate-limit checker."""
    ####################################################
    # STEP 1: REUSE THE EXISTING CHECKER WHEN IT ALREADY EXISTS#
    ####################################################
    if provider in _rate_checkers:
        return _rate_checkers[provider]

    ####################################################
    # STEP 2: CREATE A DECORATED TOKEN-BURNER FOR THIS PROVIDER#
    ####################################################
    # The callable's body does nothing; the ratelimit decorator is the
    # mechanism that tracks and enforces the call budget.
    @limits(calls=calls_per_minute, period=60)
    def burn_token() -> None:
        return None

    ####################################################
    # STEP 3: STORE AND REUSE THE CHECKER#
    ####################################################
    _rate_checkers[provider] = burn_token
    return burn_token


def reset_rate_limiters() -> None:
    """Forget every provider rate-limit checker so tests start clean."""
    ####################################################
    # STEP 1: CLEAR THE SHARED RATE-CHECKER REGISTRY#
    ####################################################
    _rate_checkers.clear()
