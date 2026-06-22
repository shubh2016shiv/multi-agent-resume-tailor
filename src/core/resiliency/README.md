# `src/core/resiliency` — Resilience wrappers for direct LLM calls

This package owns one bounded context: protecting a **specific Python function
that makes a direct provider call** with retry, circuit breaking, rate limiting,
and correlation-aware logging.

It does **not** own:
- CrewAI task-level retries or agent-wide RPM caps
- observability storage or tracing backends
- business logic for parsing or validating LLM output
- real timeout cancellation (`timeout_seconds` is config only for now)

## Public API

Import only from the package root:

```python
from src.core.resiliency import (
    get_resilience_stats,
    resilient_llm_call,
    reset_circuit_breakers,
)
```

- `resilient_llm_call(...)`
  wraps one direct provider-call function
- `reset_circuit_breakers()`
  clears shared breaker state, mainly for tests
- `get_resilience_stats(provider)`
  returns one provider's current breaker status

## Current production integration

The live integration point today is:

- `src/tools/llm_gateway/structured_output.py`

`_request_structured_output()` is decorated with `@resilient_llm_call()`, so
the tools-layer direct LLM gateway gets this package's protections.

## File map

| File | Owns |
|---|---|
| `__init__.py` | Public package surface |
| `resilient_call.py` | Wrapper composition order and correlation-id binding |
| `policy.py` | `ResiliencePolicy`, retryable exception set, config resolution |
| `circuit_breaker.py` | Provider-scoped breaker registry and stats/reset helpers |
| `rate_limit.py` | Provider-scoped rate-limit checker registry |
| `transient_retry.py` | Tenacity retry decorator factory |

## Runtime sequence

For one logical decorated call, the runtime order is:

1. bind correlation ID
2. circuit breaker wrapper
3. retry wrapper
4. rate-limit gate
5. original function body
6. success/failure log
7. unbind correlation ID

Retry is outside the rate gate, so each retry attempt still respects rate
limits. The circuit breaker is outside retry, so an open breaker fast-fails
without re-driving known-bad calls.

## Configuration

Settings live in `src/config/settings.yaml` under `llm.resilience`:

- `retry_max_attempts`
- `retry_initial_delay`
- `retry_max_delay`
- `retry_exponential_multiplier`
- `circuit_breaker_failure_threshold`
- `circuit_breaker_timeout`
- `rate_limit_calls_per_minute`
- `timeout_seconds`

Decorator arguments override config for that one wrapped function:

```python
@resilient_llm_call(provider="gemini", max_attempts=5)
```

## Important limitation

`timeout_seconds` is **not enforced** in this package yet. It is config drift,
not active behavior. Do not assume a hung provider call will be cancelled here.
