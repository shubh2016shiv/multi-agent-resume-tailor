"""Provider-scoped rate-limit checkers for direct LLM calls.

Like the circuit breaker, a rate limiter must remember how many calls happened
in the current window ACROSS calls — so there is one shared checker per
provider, kept in a module-level registry and handed out by rate_checker_for().
`_rate_checkers` is underscored internal state: use the accessor, don't import
the dict.
"""

from __future__ import annotations

from collections.abc import Callable

from ratelimit import limits  # decorator that raises RateLimitException once the budget is spent

# One rate-limit checker per provider name, shared across all calls.
_rate_checkers: dict[str, Callable[[], None]] = {}


def rate_checker_for(provider: str, calls_per_minute: int) -> Callable[[], None]:
    """Return the shared per-provider rate-limit checker, creating it once.

    The returned callable is a "token burner": calling it is how you spend one
    unit of the budget. It does no real work — the whole point is the side
    effect inside `ratelimit`, which raises RateLimitException on the call that
    would exceed `calls_per_minute`. The caller (resilient_call.py) treats that
    exception as the signal to wait.
    """
    ####################################################
    # STEP 1: REUSE THE EXISTING CHECKER WHEN IT ALREADY EXISTS
    ####################################################
    # Recreating it would reset the window counter and defeat rate limiting, so
    # the first checker for a provider is the one we keep.
    if provider in _rate_checkers:
        return _rate_checkers[provider]

    ####################################################
    # STEP 2: CREATE A DECORATED TOKEN-BURNER FOR THIS PROVIDER
    ####################################################
    # Empty body on purpose: @limits does the counting/enforcement. Each call to
    # burn_token() = one unit spent from a rolling 60-second budget.
    @limits(calls=calls_per_minute, period=60)
    def burn_token() -> None:
        return None

    ####################################################
    # STEP 3: STORE AND REUSE THE CHECKER
    ####################################################
    _rate_checkers[provider] = burn_token
    return burn_token


def reset_rate_limiters() -> None:
    """Forget every provider rate-limit checker so tests start clean.

    Not exported from the package (see __init__.py) — it is a test-support
    helper. The paired breaker reset, reset_circuit_breakers(), IS public
    because production code may want to clear breaker state; rate windows are
    only ever reset in tests.
    """
    ####################################################
    # STEP 1: CLEAR THE SHARED RATE-CHECKER REGISTRY
    ####################################################
    _rate_checkers.clear()
