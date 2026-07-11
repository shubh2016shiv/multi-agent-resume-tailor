"""Resilience wrappers for direct LLM provider calls.

WHAT THIS PACKAGE DOES
Decorate one function that makes a direct provider call, and every invocation
is protected by three nested layers plus correlation-aware logging. The layers
are an ONION — at call time they execute outside-in:

    correlation id                 (tag every log line of this call)
      └─ circuit breaker           (fast-fail if the provider looks broken)
           └─ retry + backoff      (retry transient errors, with jitter)
                └─ rate-limit gate  (stay within the per-minute budget)
                     └─ your function (the real provider call)

The ordering is the whole design — see _compose_resilience() in
resilient_call.py for WHY each layer wraps the next. Each layer is built from a
battle-tested library (pybreaker / tenacity / ratelimit), one per file.

SCOPE BOUNDARY — read before assuming this does more than it does:
- It protects ONE direct-provider Python function. It does NOT govern CrewAI
  task-level retries or agent-wide RPM caps — that's a different layer.
- `timeout_seconds` is resolved into the policy but NOT enforced here: nothing
  cancels a hung call on that deadline yet. It is config-only. See README.md.
- Breaker and rate-limit state is shared per-provider in module-level
  registries (in circuit_breaker.py / rate_limit.py) so it survives across
  calls — that persistence is required for either protection to work at all.

See README.md for the file map, the live integration point, and config keys.
"""

from src.core.resiliency.circuit_breaker import get_resilience_stats, reset_circuit_breakers
from src.core.resiliency.resilient_call import resilient_llm_call

__all__ = [
    "get_resilience_stats",
    "resilient_llm_call",
    "reset_circuit_breakers",
]
