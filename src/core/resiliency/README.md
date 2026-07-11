# Resilience for LLM Calls — From Zero to Understanding

This is a guided walk through _why_ LLM calls need protection, _what_ each
protection does, and _how_ they fit together. Read it top to bottom — each
section builds on the one before it.

---

## 1. The Problem: LLM Calls Fail, and They Fail Differently

Imagine you write a function that calls an LLM provider:

```python
def ask_llm(prompt: str) -> str:
    response = openai.chat.completions.create(model="gpt-4", messages=[...])
    return response.choices[0].message.content
```

This function can fail in **three fundamentally different ways**, and each one
needs a different response:

| Failure type | Example | Should you retry? |
|---|---|---|
| **Transient** | Network timeout, connection reset, provider says "429 too many requests" | **Yes** — the problem fixes itself if you wait |
| **Permanent** | Bad API key, malformed request, model not found | **No** — retrying won't help, it'll just waste time and money |
| **Systemic** | The provider has been returning errors for the last 30 calls | **No** — the provider itself is broken; stop calling it for a while |

A naive `try/except` doesn't distinguish between these. If you retry a
permanent error, you waste attempts. If you don't retry a transient error,
you fail unnecessarily. If you keep hammering a broken provider, you make
things worse.

We need **three separate protections**, each designed for one of these
failure modes.

---

## 2. Protection 1: Retry (handles transient failures)

The simplest protection. When a call fails with a transient error, wait a
moment and try again.

```text
  call ──► fails (timeout)
               │
               ▼
           wait 1 second
               │
               ▼
           call ──► succeeds ✓
```

### The two questions every retry strategy must answer

**Q1: WHICH errors are worth retrying?**

Only retry errors that can fix themselves with time:
- `TimeoutError` — the network was slow, not broken
- `ConnectionError` — a momentary drop
- `RateLimitException` — the provider said "slow down"

Do NOT retry:
- Bad API keys, invalid requests, model-not-found — these will never succeed

**Q2: HOW LONG should you wait between retries?**

Three wrong answers, and why:

| Strategy | Problem |
|---|---|
| Retry **immediately** | The problem hasn't had time to clear; you fail again instantly |
| Retry after a **fixed delay** (e.g., always 2 seconds) | If many callers failed together, they all retry together — a "thundering herd" |
| Retry with **exponential backoff only** (1s → 2s → 4s → ...) | Close, but identical callers still retry at identical times |

The correct answer: **exponential backoff + random jitter**.

```text
attempt 1: fails
  │
  ▼
wait 1.0s + random(0, 1)s = ~1.5s
  │
  ▼
attempt 2: fails
  │
  ▼
wait 2.0s + random(0, 1)s = ~2.3s
  │
  ▼
attempt 3: succeeds ✓
```

The exponential part (1s → 2s → 4s → ...) gives the provider progressively
more breathing room. The random jitter (0-1 extra second) scatters retries
so they don't land in lockstep. Without jitter, every caller that failed at
the same instant retries at the same instant — hammering the provider in
sync and causing _another_ wave of failures.

This is the retry layer. It lives in `transient_retry.py` and is built with
[tenacity](https://tenacity.readthedocs.io/).

**What you need to remember:** retry transient, not permanent; back off
exponentially; always add jitter.

---

## 3. Protection 2: Rate Limiting (stays within the provider's budget)

Retry solves _transient_ failures, but it creates a new problem: what if the
failure WAS a rate-limit hit?

Most LLM providers enforce a calls-per-minute (or tokens-per-minute) cap.
Exceed it, and they reject your calls with a 429. If every rejected call
triggers a retry, and every retry also gets rejected, you're in a loop:

```text
call ──► 429 (rate limited)
           │
           ▼
       retry after backoff
           │
           ▼
       call ──► 429 (still rate limited!)
           │
           ▼
       retry again ──► 429 ──► retry ──► ... (wastes attempts, never succeeds)
```

We need a **gate** that counts how many calls we've made in the current
minute and says "stop" _before_ we hit the provider's limit:

```text
call arrives
  │
  ▼
rate gate: "have we made fewer than N calls this minute?"
  │
  ├── YES ──► spend 1 token from the budget ──► make the real call
  │
  └── NO  ──► budget exhausted ──► sleep until the window resets ──► then call
```

The gate uses a **sliding 60-second window**. Once the budget is spent, any
new call sleeps until the oldest call in the window ages out. This is
self-healing: you never need to catch up or reset anything.

Note the subtle point: the rate gate **sleeps and retries internally** when
the budget is exhausted. It doesn't bounce the failure up to the retry layer
above it — because a rate hit isn't a "failure," it's expected throttling.
Sleeping inside the gate is cheaper (one `time.sleep`) than bouncing all the
way out to the retry layer, going through backoff, and coming back down.

This layer lives in `rate_limit.py`, built with
[ratelimit](https://pypi.org/project/ratelimit/).

**What you need to remember:** rate limiting prevents you from exceeding the
provider's budget. It must be the innermost protection so _every_ retry
attempt also respects the budget.

---

## 4. Protection 3: Circuit Breaker (detects broken providers)

Retry handles transient failures. Rate limiting handles budget. What about
the third failure mode — when the provider itself is broken?

Imagine the provider starts returning errors on every call. Not rate limits —
actual errors. Without a circuit breaker:

```text
call 1 ──► error
  retry ──► error
  retry ──► error     ← 3 failures, 3 provider calls wasted

call 2 ──► error
  retry ──► error
  retry ──► error     ← 6 failures now

call 3 ──► error      ← and on, and on, and on...
```

Every call wastes retries on a provider that is clearly down. We need
something that **notices the pattern** and says "stop calling this provider,
it's broken" — and then **tests the waters** later to see if it's recovered.

This is a **circuit breaker**. It has three states:

```text
                    ┌──────────────┐
          ┌────────►│   CLOSED     │◄────────┐
          │         │ (normal)     │         │
          │         └──────┬───────┘         │
          │                │                 │
          │     N consecutive failures       │
          │                │                 │
          │                ▼                 │
          │         ┌──────────────┐         │
          │         │    OPEN      │         │
          │         │ (fast-fail)  │         │
          │         └──────┬───────┘         │
          │                │                 │
          │       reset timeout expires      │
          │       (e.g., 60 seconds)         │
          │                │                 │
          │                ▼                 │
          │         ┌──────────────┐         │
          └─────────│  HALF-OPEN   │─────────┘
                    │ (testing)    │
                    └──────────────┘
                      one trial call
                   success → CLOSED
                   failure → OPEN again
```

The three states:

- **CLOSED** — everything is normal. Calls go through. The breaker counts
  failures.

- **OPEN** — the breaker has seen too many consecutive failures. It
  **short-circuits**: calls fail immediately without even attempting the
  provider. This is called "fast-failing" and it saves time, money, and
  provider load.

- **HALF-OPEN** — after a cooldown period, the breaker allows exactly one
  trial call. If it succeeds, the breaker resets to CLOSED. If it fails, it
  goes straight back to OPEN.

### The critical decision: what does the breaker count?

The circuit breaker sits **outside** the retry layer. This means it sees
only the **final outcome** of a logical call — success, or failure after all
retries are exhausted. It does NOT count individual retry attempts.

```text
                    breaker sees THIS ──────────────┐
                                                    │
    ┌──────────┐    ┌──────────┐    ┌────────────┐  │
    │ BREAKER  │───►│  RETRY   │───►│ RATE GATE  │──┼──► provider
    └──────────┘    └──────────┘    └────────────┘  │
         ▲                                          │
         │       breaker counts ONE failure,        │
         │       not one per retry attempt          │
         └──────────────────────────────────────────┘
```

This is important: without this ordering, a single flaky call that retried
3 times would count as 3 failures and might trip the breaker open
prematurely. By wrapping the retry, the breaker sees "did the logical call
succeed or not?" — not "how many tries did it take?"

### Another critical decision: rate-limit hits must NOT count as failures

Being rate-limited means the provider is working fine — it's just busy.
Tripping the breaker because you hit a rate limit would be wrong. So the
breaker explicitly **excludes** `RateLimitException` from its failure count.

This layer lives in `circuit_breaker.py`, built with
[pybreaker](https://pypi.org/project/pybreaker/).

**What you need to remember:** the breaker detects systemic provider failure
and stops calling. It must wrap retry, not sit inside it. Rate-limit hits are
not failures.

---

## 5. Putting It Together: The Onion

Now we have three protections. But the order matters enormously. Here's the
structure, read from the outside in:

```text
                    ┌─────────────────────────────────────┐
                    │         CORRELATION ID              │
                    │  (every log line tagged with one    │
                    │   traceable id for this call)       │
                    │                                     │
                    │  ┌───────────────────────────────┐  │
                    │  │      CIRCUIT BREAKER          │  │
                    │  │  (fast-fail if provider looks │  │
                    │  │   broken; sees final outcome  │  │
                    │  │   of the logical call, not    │  │
                    │  │   individual retry attempts)  │  │
                    │  │                               │  │
                    │  │  ┌─────────────────────────┐  │  │
                    │  │  │    RETRY + BACKOFF      │  │  │
                    │  │  │  (retry transient       │  │  │
                    │  │  │   errors — timeouts,    │  │  │
                    │  │  │   connection drops,     │  │  │
                    │  │  │   rate-limit hits —     │  │  │
                    │  │  │   with exponential      │  │  │
                    │  │  │   backoff + jitter)     │  │  │
                    │  │  │                         │  │  │
                    │  │  │  ┌───────────────────┐  │  │  │
                    │  │  │  │   RATE-LIMIT GATE │  │  │  │
                    │  │  │  │  (stay within the │  │  │  │
                    │  │  │  │   per-minute      │  │  │  │
                    │  │  │  │   provider budget;│  │  │  │
                    │  │  │  │   self-heal by    │  │  │  │
                    │  │  │  │   sleeping)       │  │  │  │
                    │  │  │  └────────┬──────────┘  │  │  │
                    │  │  │           │              │  │  │
                    │  │  │           ▼              │  │  │
                    │  │  │     YOUR FUNCTION        │  │  │
                    │  │  │  (the real provider call)│  │  │
                    │  │  └─────────────────────────┘  │  │
                    │  └───────────────────────────────┘  │
                    └─────────────────────────────────────┘
```

### Why this order (and not any other)

Read this thinking about ONE call that retries 3 times:

**Rate gate is innermost.** Every attempt — first call plus all retries —
must pass through the rate gate. If the rate gate were _outside_ retry, a
burst of retries could blow past the provider's rate limit. One logical call
making 3 retries would count as 1 call at the gate (wrong — it's actually
4 provider calls).

**Retry is in the middle.** It wraps the rate gate, so each retry attempt
re-enters the rate gate and spends from the same budget. It retries only
transient errors; permanent errors propagate straight up to the breaker.

**Circuit breaker is outermost.** It wraps retry, so N retries of one
logical call count as ONE failure — not N. An already-open breaker
fast-fails before any retry or rate-limit work happens, saving time.

**Correlation ID is the skin.** It wraps everything so that every log line
— from the breaker, retry, rate gate, and your function — shares one
traceable ID. When you see `circuit_open provider=gemini correlation_id=abc123`
in your logs, you can find the matching `llm_call_started correlation_id=abc123`
and trace the whole lifecycle.

---

## 6. The Policy: One Frozen Rulebook Per Function

All three protections need numbers:

- How many retry attempts? (e.g., 3)
- What's the backoff start delay? (e.g., 1 second)
- How many failures before the breaker trips? (e.g., 5)
- How many calls per minute? (e.g., 15)

These numbers come from **configuration** (`src/config/settings.yaml`), not
from code. They are resolved once into a `ResiliencePolicy` — a frozen
(immutable) dataclass — at the moment the decorator is applied. After that,
the policy cannot change for the lifetime of that decorated function.

Why frozen? Because if the policy could drift mid-life, a function decorated
with `max_attempts=3` might suddenly behave as if it had `max_attempts=1`
after a config reload. That's a debugging nightmare. Freeze it once, and the
function's contract is stable.

```text
settings.yaml          decorator args           resolved policy
─────────────          ──────────────           ───────────────
retry_max_attempts: 3  max_attempts=5    ──►    attempt_limit: 5
                                              (decorator overrides config)
                                              (frozen — never changes)
```

---

## 7. Shared State: Why Breakers and Rate Limiters Are Singletons

A circuit breaker that forgets its failure count between calls is useless.
If call #1 fails and call #2 starts with a fresh breaker that says "0
failures," the breaker can never trip.

Similarly, a rate limiter that forgets how many calls it allowed this minute
can never say "stop, the budget is spent."

So both the circuit breaker and the rate-limit checker live in **module-level
registries** — one instance per provider name, shared across every decorated
function that uses that provider:

```text
                            _breakers dict
                            ─────────────
provider="openai"    ──►    CircuitBreaker instance (shared)
provider="gemini"    ──►    CircuitBreaker instance (shared)

                            _rate_checkers dict
                            ──────────────────
provider="openai"    ──►    burn_token() function (shared)
provider="gemini"    ──►    burn_token() function (shared)
```

This is why `provider` is the key argument to `@resilient_llm_call()` — it
determines which breaker and rate checker protect this function.

---

## 8. How to Use It (One Decorator)

All of the above — the onion, the policy, the shared state — is hidden
behind a single decorator:

```python
from src.core.resiliency import resilient_llm_call

@resilient_llm_call()                          # use all config defaults
def call_openai(prompt: str) -> str: ...

@resilient_llm_call(provider="gemini")         # scoped to a specific provider
def call_gemini(prompt: str) -> str: ...

@resilient_llm_call(provider="openai", max_attempts=5)  # override retry count
def call_openai_with_more_retries(prompt: str) -> str: ...
```

That's it. Every invocation of `call_openai(...)` is now protected by:

1. A fresh correlation ID for traceability
2. A circuit breaker that fast-fails if OpenAI looks broken
3. Retry with exponential backoff + jitter for transient errors
4. A rate-limit gate that keeps you within OpenAI's budget

The caller writes business logic. The decorator handles resilience.

---

## 9. Portable Decision Rules

These are the concepts worth carrying to any project, in any language.
The library names will change; the rules won't.

| # | Rule | Why |
|---|---|---|
| 1 | **Retry transient errors only** — timeouts, connection drops, rate-limit responses | Permanent errors (auth, validation, not-found) will never succeed on retry |
| 2 | **Always add jitter to backoff** | Without jitter, callers that failed together retry together — "thundering herd" |
| 3 | **Rate gate goes inside retry** | Every retry attempt must spend from the same per-minute budget |
| 4 | **Circuit breaker goes outside retry** | The breaker must see final outcomes (success or exhaustion), not per-attempt noise |
| 5 | **Exclude rate-limit hits from breaker counting** | Throttling is expected behavior, not a provider failure |
| 6 | **Per-provider shared state for breakers and rate gates** | A breaker that forgets between calls can never trip; a rate gate that forgets can never enforce |
| 7 | **Freeze the policy at decoration time** | A decorated function's resilience contract must not drift mid-life |
| 8 | **Bind a unique ID to every logical call** | When something fails, you need to trace every log line back to one originating call |
| 9 | **Timeout is separate from these three protections** | A hung call needs cancellation, not retry — it's a different mechanism entirely |

---

## 10. What This Module Does NOT Do

Knowing the boundaries is as important as knowing the contents:

- **Does NOT govern CrewAI task-level retries.** CrewAI has its own
  `max_retry_limit`. This module protects raw provider-call functions, not
  agent task execution.

- **Does NOT enforce timeouts yet.** `timeout_seconds` exists in config and in
  the policy, but no code cancels a hung call on that deadline. It's wired
  for the future, not active today.

- **Does NOT store traces or metrics.** This module emits structured logs
  (`logger.info("llm_call_success", ...)`) but does not persist them.
  Observability storage is handled by `src/observability/`.

- **Does NOT parse or validate LLM output.** That's the caller's job. This
  module only protects the transport.

---

## 11. File Map

| File | What it owns |
|---|---|
| `policy.py` | `ResiliencePolicy` frozen dataclass, retryable-exception set, config resolution |
| `transient_retry.py` | Tenacity-based retry decorator factory (which errors, how many attempts, what backoff) |
| `rate_limit.py` | Per-provider rate-limit gate registry using the `ratelimit` library |
| `circuit_breaker.py` | Per-provider circuit-breaker registry using `pybreaker`, plus stats/reset helpers |
| `resilient_call.py` | Composes all four layers into one `@resilient_llm_call` decorator |
| `__init__.py` | Public surface: exports `resilient_llm_call`, `get_resilience_stats`, `reset_circuit_breakers` |

---

## 12. Configuration Reference

All defaults live in `src/config/settings.yaml` under `llm.resilience`:

| Key | Purpose | Example |
|---|---|---|
| `retry_max_attempts` | Total attempts (1 call + N retries) | `3` |
| `retry_initial_delay` | Seconds before first retry | `1.0` |
| `retry_max_delay` | Maximum backoff ceiling (seconds) | `30.0` |
| `retry_exponential_multiplier` | Multiplier for exponential growth | `2.0` |
| `circuit_breaker_failure_threshold` | Consecutive failures to trip OPEN | `5` |
| `circuit_breaker_timeout` | Seconds before trying HALF-OPEN | `60` |
| `rate_limit_calls_per_minute` | Max calls per 60-second window | `15` |
| `timeout_seconds` | Per-call deadline (not enforced yet) | `180` |

Override per-decorator:

```python
@resilient_llm_call(provider="gemini", max_attempts=5)
```

---

## 13. Inspection and Debugging

```python
from src.core.resiliency import get_resilience_stats, reset_circuit_breakers

# Check if a provider's breaker is open
stats = get_resilience_stats("openai")
print(stats["circuit_breaker_state"])   # "closed" | "open" | "half-open"
print(stats["failure_count"])           # current failure tally

# Reset all breakers (mainly for tests)
reset_circuit_breakers()
```
