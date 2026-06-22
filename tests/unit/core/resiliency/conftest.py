"""Shared fixtures for resiliency unit tests."""

import pytest

from src.core.resiliency.circuit_breaker import reset_circuit_breakers
from src.core.resiliency.rate_limit import reset_rate_limiters


@pytest.fixture(autouse=True)
def reset_resiliency_state():
    """Reset shared provider state before and after each test."""
    reset_circuit_breakers()
    reset_rate_limiters()
    yield
    reset_circuit_breakers()
    reset_rate_limiters()
