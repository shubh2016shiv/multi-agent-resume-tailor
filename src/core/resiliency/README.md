# `src/core/resiliency` — Resilience for LLM provider calls

New to this package? Read this top to bottom. It explains **why resilience
exists here**, **how it connects to the rest of the project**, **how control
flows through it**, and **where every decision is made in the code**.

---

## 1. Why this exists

This project is a multi-agent resume tailoring pipeline. Every agent (job
analysis, gap analysis, skills optimization, summary writing, etc.) does its
real work by calling an external LLM provider API (Gemini in production).

External API calls fail in ways your own code never does:

- **Transient failures** — the network blips, the provider times out. The same
  call would succeed a moment later.
- **Rate limits** — the provider caps calls per minute. Exceed it and every
  call is rejected.
- **Sustained outages** — the provider is down. Hammering it with retries
  wastes time, burns quota, and delays the failure report.

This package is the single place that makes those calls **survivable**. Agents
do not each reinvent retry/backoff/limiting logic.

---

## 2. Two layers of resilience in this project — and why both exist

Before touching this package, understand that the project has **two independent
resilience layers**. They protect different things.

### Layer 1: CrewAI agent defaults (`src/config/settings.yaml`, `agent_defaults`)

```yaml
agent_defaults:
  max_retry_limit: 3        # retries when an agent errors mid-task
  max_rpm: 60               # requests per minute across the whole agent
  max_iter: 15              # max reasoning iterations before forcing a result
  max_execution_time: 300   # hard wall: 5 minutes per task
```

CrewAI applies these to **every agent automatically** — no decorator needed.
They protect the **agent task execution level**: CrewAI retries the whole agent
task when it crashes, caps the agent's overall request rate, and kills a
runaway task after 5 minutes. You do not control this in Python code; it is
wired by the CrewAI framework based on these config values.

### Layer 2: This package (`src/core/resiliency`)

This protects a **specific function that makes a direct provider API call**.
It is applied by explicitly decorating that function with
`@resilient_llm_call(...)`. It is not automatic — if a function is not
decorated, it is not protected by this layer.

It adds: retry with exponential backoff, per-provider circuit breaking, and
per-provider rate limiting — all at the individual call level.

### Which layer covers what

| | Layer 1: CrewAI | Layer 2: this package |
|---|---|---|
| Applied by | Framework, automatically | You, explicitly, with `@resilient_llm_call` |
| Protects | Whole agent task execution | One specific provider-call function |
| Retry on | Agent-level task error | Transient network / timeout / rate-limit |
| Rate limit | Across whole agent | Per decorated function's provider |
| Circuit breaking | None | Per provider (Gemini, etc.) |
| Configured in | `settings.yaml` `agent_defaults` | `settings.yaml` `llm.resilience` |

They are complementary. Layer 1 is the outer safety net — it catches an agent
that completely fell over. Layer 2 is the inner guard — it catches a flaky
individual provider call before the agent ever sees it as an error. If Layer 2
retries succeed, Layer 1 never triggers. If they exhaust, Layer 1 may retry
the entire task.

---

## 3. The one touch point between an agent and this package

The integration surface is **one decorator applied to the function that makes
the provider call**:

```python
from src.core.resiliency import resilient_llm_call


@resilient_llm_call(provider="gemini")
def call_the_model(prompt: str, agent: Agent) -> SomeResult:
    # This body does the LLM work. It knows nothing about retry or
    # circuit breaking. The decorator adds that from the outside.
    response = agent.llm.call(prompt)
    return parse_result(response)
```

Any agent, present or future, adds resilience the **same way**: decorate the
function that talks to the provider. No changes inside this package are needed.

---

## 4. How control flows at runtime

When the decorated function is called, three wrappers execute in sequence
before the real function body runs.

**The ordering is determined by the application order in `_compose_resilience`
in `resilient_call.py` lines 48-63.** Python decorators compose inside-out: the
first wrapper applied is the innermost at runtime; the last applied is the
outermost. Rate limiter is applied first (line 48), circuit breaker second
(line 52), retry last (line 56).

```
caller calls the decorated function
   |
   v
[ bind correlation_id ]
     structlog.contextvars.bind_contextvars(...)     <- resilient_call.py wrapper line 98
     every log line for this call now carries one id
   |
   v
[ RETRY ]                                            <- resilient_call.py line 56 (outermost)
     transient_retry.py: create_retry_decorator(...)
     catches: TimeoutError, ConnectionError, RateLimitException
     waits with exponential backoff, up to max_attempts
     re-drives the ENTIRE stack below on each attempt
   |
   v
[ CIRCUIT BREAKER ]                                  <- resilient_call.py line 52
     circuit_breaker.py: protect_with_circuit_breaker(...)
     one breaker per provider, shared across all calls to that provider
     if failure_threshold consecutive failures: opens, raises CircuitBreakerError
     CircuitBreakerError is NOT retryable -- it surfaces immediately
   |
   v
[ RATE LIMITER ]                                     <- resilient_call.py line 48 (innermost)
     rate_limit.py: create_rate_limiter(...)
     caps calls_per_minute over a 60s window
     on limit hit: logs, sleeps one slot, retries once
   |
   v
[ the original function body ]                       <- your code
     the real LLM provider API call runs here
   |
   v
[ log success or failure ]                           <- resilient_call.py wrapper
     logger.info("llm_call_success") or
     logger.info("llm_call_failed", error=..., retryable=...)
   |
   v
[ unbind correlation_id ]                            <- resilient_call.py finally block
     structlog.contextvars.unbind_contextvars("correlation_id")
     the next call starts clean
```

**Why this order and not another?**

- Rate limiter innermost: it gates the actual call. Retry and circuit breaker
  should observe a rate-limited call, not bypass the limit on each attempt.
- Circuit breaker in the middle: it watches the rate-limited call. If the
  provider is broadly failing it opens, and the outermost retry sees
  `CircuitBreakerError` — which is not in `RETRYABLE_EXCEPTIONS`, so retry
  does not re-drive a known-open circuit.
- Retry outermost: a transient failure re-drives the whole protected stack,
  so each retry still respects the rate limit and circuit breaker.

---

## 5. A worked example: what happens on a flaky call

Call 1: success.

```
bind correlation_id=a1b2c3d4
retry [attempt 1]
  circuit breaker: closed, allow
  rate limiter: under quota, allow
  provider call: OK
log llm_call_success
unbind correlation_id
```

Call 2: provider returns a transient timeout, then succeeds on retry.

```
bind correlation_id=e5f6g7h8
retry [attempt 1]
  circuit breaker: closed, allow
  rate limiter: under quota, allow
  provider call: TimeoutError  <-- transient
log retry_attempt attempt=1 next_wait=1.0s
retry [attempt 2, after 1s backoff]
  circuit breaker: closed, allow (one failure, threshold not reached)
  rate limiter: under quota, allow
  provider call: OK
log llm_call_success
unbind correlation_id
```

Call 3: provider is down, all retries exhaust, circuit opens.

```
bind correlation_id=i9j0k1l2
retry [attempt 1]  -> TimeoutError -> backoff 1s
retry [attempt 2]  -> TimeoutError -> backoff 2s
retry [attempt 3]  -> TimeoutError -> exhausted, re-raise TimeoutError
  circuit breaker: 3 consecutive failures now counted
log llm_call_failed retryable=True
unbind correlation_id
-- caller receives TimeoutError --
```

After `circuit_breaker_failure_threshold` (default 5) failures, the circuit
opens. Call N then looks like:

```
bind correlation_id=m3n4o5p6
retry [attempt 1]
  circuit breaker: OPEN -> CircuitBreakerError immediately (no provider call)
log circuit_breaker_open
  retry sees CircuitBreakerError: NOT in RETRYABLE_EXCEPTIONS, does not retry
log llm_call_failed retryable=False
unbind correlation_id
-- caller receives CircuitBreakerError immediately --
```

---

## 6. Configuration

This package does not store configuration. The knobs live in:

**`src/config/settings.yaml`** under `llm.resilience`:

```yaml
llm:
  resilience:
    retry_max_attempts: 3
    retry_initial_delay: 1.0
    retry_max_delay: 60.0
    retry_exponential_multiplier: 2.0
    circuit_breaker_failure_threshold: 5
    circuit_breaker_timeout: 60
    rate_limit_calls_per_minute: 60
    timeout_seconds: 30         # NOTE: see Known Gaps below
```

Flow: `settings.yaml` -> loaded by `src/core/settings.get_config()` ->
resolved by `policy.py:resolve_policy()` into a frozen `ResiliencePolicy` ->
read by `resilient_call.py` at decoration time.

Decorator arguments override config for one function only:

```python
@resilient_llm_call(provider="gemini", max_attempts=5)
```

---

## 7. Known gaps

**`timeout_seconds` is declared but not enforced.** The value is read from
config and stored in `ResiliencePolicy.timeout_seconds`, but no call is
actually cancelled after 30 seconds. A hung provider call hangs forever.
A real timeout requires a `threading.Timer`, `signal.alarm`, or an async
`asyncio.wait_for` wrapper. That does not exist in this package yet.

Do not rely on this config value to limit hung calls. Rely on Layer 1's
`max_execution_time` (5 minutes per task) as the outer wall.

---

## 8. Observability

Every decorated call gets a short correlation ID bound to the log context via
`structlog.contextvars`. Every event during that call carries the same ID:
`llm_call_start`, `retry_attempt`, `circuit_breaker_open`, `rate_limit_hit`,
`llm_call_success`, `llm_call_failed`. One flaky call can be traced across
all its retry attempts in the logs without mixing with concurrent calls.

Circuit breaker state can be inspected at runtime:

```python
from src.core.resiliency import get_resilience_stats, reset_circuit_breakers

get_resilience_stats("gemini")   # state, failure_count, last_failure_time
reset_circuit_breakers()         # force all breakers closed (use in tests)
```

---

## 9. File map

| File | Owns |
|---|---|
| `__init__.py` | Public facade. Always import from here. |
| `resilient_call.py` | `resilient_llm_call` and `_compose_resilience`. The ordering of layers is here. |
| `policy.py` | `ResiliencePolicy` dataclass + `RETRYABLE_EXCEPTIONS` + `resolve_policy`. |
| `circuit_breaker.py` | PyBreaker integration. One breaker per provider. |
| `rate_limit.py` | `ratelimit` integration. Per-provider call-rate enforcement. |
| `transient_retry.py` | Tenacity retry. Only retries transient errors, never programming errors. |

---

## 10. Extending this package

- **Adding resilience to a new agent:** do not touch this package. Decorate the
  agent's provider-call function with `@resilient_llm_call(provider=...)`.
- **Changing knobs:** edit `src/config/settings.yaml` under `llm.resilience`.
- **Adding a new mechanism (e.g. timeout enforcer, bulkhead):** add a new
  single-concern file next to the existing mechanisms and wire it into
  `_compose_resilience` in `resilient_call.py`. Follow the inside-out ordering
  rationale — decide where in the stack it belongs before writing code.
- **Testing with a clean circuit breaker state:** call `reset_circuit_breakers()`
  in your test setup. The breakers are module-level singletons in
  `circuit_breaker.py`; they persist across calls within one process.
