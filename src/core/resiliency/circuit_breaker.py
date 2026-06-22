"""Circuit-breaker registry and inspection helpers for provider calls."""

from __future__ import annotations

from pybreaker import CircuitBreaker
from ratelimit import RateLimitException

from src.core.logger import get_logger
from src.core.resiliency.policy import ResiliencePolicy

logger = get_logger(__name__)

_breakers: dict[str, CircuitBreaker] = {}


def breaker_for(provider: str, policy: ResiliencePolicy) -> CircuitBreaker:
    """Return the shared circuit breaker for one provider."""
    ####################################################
    # STEP 1: CREATE THE BREAKER ON FIRST USE#
    ####################################################
    if provider not in _breakers:
        _breakers[provider] = CircuitBreaker(
            fail_max=policy.circuit_breaker_failure_threshold,
            reset_timeout=policy.circuit_breaker_timeout,
            exclude=[RateLimitException],
        )
        logger.info("circuit_breaker_created", provider=provider)

    ####################################################
    # STEP 2: REUSE THE SHARED BREAKER THEREAFTER#
    ####################################################
    return _breakers[provider]


def get_resilience_stats(provider: str) -> dict[str, object]:
    """Return one provider's current circuit-breaker status."""
    ####################################################
    # STEP 1: RETURN A STABLE SHAPE EVEN BEFORE FIRST USE#
    ####################################################
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
    # STEP 2: EXPOSE ONLY THE OBSERVABLE BREAKER STATE#
    ####################################################
    return {
        "provider": provider,
        "circuit_breaker_state": breaker.current_state,
        "failure_count": breaker.fail_counter,
        "failure_threshold": breaker.fail_max,
        "reset_timeout_seconds": breaker.reset_timeout,
        "opened_at": breaker._state_storage.opened_at,
    }


def reset_circuit_breakers() -> None:
    """Forget every provider breaker so the next call starts fresh."""
    ####################################################
    # STEP 1: CLEAR THE REGISTRY OF SHARED BREAKERS#
    ####################################################
    # Decorated functions resolve breakers through this registry on each
    # logical call, so clearing it is enough to give tests a clean slate.
    _breakers.clear()
