"""Circuit-breaker registry and inspection helpers for provider calls.

A circuit breaker must remember failures ACROSS calls to do its job — so the
breaker for a provider can't be rebuilt on each call. This file keeps one
shared breaker per provider in a module-level registry, and hands it out
through breaker_for(). The `_breakers` name is underscored because it is
internal mutable state: reach it only through the functions in this file, never
by importing the dict directly.
"""

from __future__ import annotations

from pybreaker import CircuitBreaker
from ratelimit import RateLimitException  # excluded from breaker counting, see breaker_for()

from src.core.logger import get_logger
from src.core.resiliency.policy import ResiliencePolicy

logger = get_logger(__name__)

# One breaker per provider name, shared across all calls for that provider.
_breakers: dict[str, CircuitBreaker] = {}


def breaker_for(provider: str, policy: ResiliencePolicy) -> CircuitBreaker:
    """Return the shared circuit breaker for one provider, creating it once."""
    ####################################################
    # STEP 1: CREATE THE BREAKER ON FIRST USE
    ####################################################
    if provider not in _breakers:
        _breakers[provider] = CircuitBreaker(
            fail_max=policy.circuit_breaker_failure_threshold,  # trips open after this many fails
            reset_timeout=policy.circuit_breaker_timeout,  # seconds open before a trial call
            # Rate-limit hits are expected throttling, not provider failure, so
            # they must NOT count toward tripping the breaker open.
            exclude=[RateLimitException],
        )
        logger.info("circuit_breaker_created", provider=provider)

    ####################################################
    # STEP 2: REUSE THE SHARED BREAKER THEREAFTER
    ####################################################
    # Returning the same instance is what lets failure counts accumulate across
    # calls until the threshold trips it.
    return _breakers[provider]


def get_resilience_stats(provider: str) -> dict[str, object]:
    """Return one provider's current circuit-breaker status (read-only inspection)."""
    ####################################################
    # STEP 1: RETURN A STABLE SHAPE EVEN BEFORE FIRST USE
    ####################################################
    # A provider that hasn't been called yet has no breaker. Return the same
    # dict keys anyway so callers/dashboards never have to special-case a
    # missing provider.
    breaker = _breakers.get(provider)
    if breaker is None:
        return {
            "provider": provider,
            "circuit_breaker_state": "not_initialized",
            "failure_count": 0,
            "failure_threshold": None,
            "reset_timeout_seconds": None,
            "opened_at": None,
        }

    ####################################################
    # STEP 2: EXPOSE ONLY THE OBSERVABLE BREAKER STATE
    ####################################################
    return {
        "provider": provider,
        "circuit_breaker_state": breaker.current_state,  # "closed" | "open" | "half-open"
        "failure_count": breaker.fail_counter,
        "failure_threshold": breaker.fail_max,
        "reset_timeout_seconds": breaker.reset_timeout,
        # pybreaker exposes no public getter for the open timestamp, so read the
        # storage attribute directly; it's None while the breaker is closed.
        "opened_at": breaker._state_storage.opened_at,
    }


def reset_circuit_breakers() -> None:
    """Forget every provider breaker so the next call starts fresh."""
    ####################################################
    # STEP 1: CLEAR THE REGISTRY OF SHARED BREAKERS
    ####################################################
    # breaker_for() lazily recreates a breaker on next use, so emptying the
    # registry is enough to fully reset breaker state — mainly for tests that
    # need an isolated starting point.
    _breakers.clear()
