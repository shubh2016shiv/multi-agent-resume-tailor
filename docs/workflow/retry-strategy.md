# Retry Strategy
## Four Layers of Retry, Each Scoped to a Different Kind of Failure

> **Scope:** `src/core/resiliency/` (the direct-provider resilience stack),
> CrewAI's native `agent_defaults`, the malformed-output retry inside
> `request_structured_output`, and the single bounded content-repair passes
> documented in [Orchestration Graph](orchestration-graph.md).
> **Audience:** Contributors adding a new LLM call path; anyone asking "why did
> this fail immediately instead of retrying" or "why did this retry silently
> instead of surfacing the error."

---

## Table of Contents

1. [What Problem the Retry Strategy Solves](#1-what-problem-the-retry-strategy-solves)
2. [The Core Design Principle — Match the Retry to the Failure's Nature](#2-the-core-design-principle--match-the-retry-to-the-failures-nature)
3. [The Four Layers, at a Glance](#3-the-four-layers-at-a-glance)
4. [Layer 1 — CrewAI's Native Agent Resilience](#4-layer-1--crewais-native-agent-resilience)
5. [Layer 2 — The Direct-Provider Resiliency Stack](#5-layer-2--the-direct-provider-resiliency-stack)
6. [Layer 3 — The Malformed-Output Retry](#6-layer-3--the-malformed-output-retry)
7. [Layer 4 — The One Bounded Content-Repair Pass](#7-layer-4--the-one-bounded-content-repair-pass)
8. [Case Study: request_structured_output's Three Stacked Layers](#8-case-study-request_structured_outputs-three-stacked-layers)
9. [Where CrewAI's Retries and the Resiliency Stack Never Meet](#9-where-crewais-retries-and-the-resiliency-stack-never-meet)
10. [The "No Unbounded Retry" Rule, Recap](#10-the-no-unbounded-retry-rule-recap)
11. [Design Rule — Classify Before You Retry](#11-design-rule--classify-before-you-retry)
12. [Future Considerations](#12-future-considerations)

---

## 1. What Problem the Retry Strategy Solves

Not every failure deserves the same response, and treating them as if they do
is how systems become either fragile or wasteful. A dropped network
connection to an LLM provider should be retried — it has nothing to do with
the resume being tailored. A model that returns text a candidate's own
history cannot support should never be retried into compliance — that isn't a
glitch, it's the model doing exactly what an unconstrained retry loop invites
it to do: keep guessing until something looks acceptable. A bullet that's
truthful but missing a number only the candidate knows should never be
retried at all — no amount of re-prompting produces a fact the model was
never given.

This codebase draws that line in four distinct places, each scoped to exactly
one class of failure, and none of them overlapping in what they're allowed to
retry.

---

## 2. The Core Design Principle — Match the Retry to the Failure's Nature

```text
   failure is INFRASTRUCTURE            failure is a SCHEMA/PARSING problem
   (timeout, connection drop,           (the model returned prose instead of
    provider rate limit)                 JSON, or JSON that didn't validate)
        |                                       |
        v                                       v
   retry transparently,                 retry once, narrowly, then fail
   with backoff + a circuit             loudly -- this is not a network
   breaker (Section 5)                  problem, do not treat it like one
                                         (Section 6)

   failure is a CONTENT/QUALITY         failure is a MISSING FACT only
   judgment (a rewrite reads as         the candidate has (a bullet that's
   generic, or fails a truth check)     truthful but too thin)
        |                                       |
        v                                       v
   ONE disclosed repair attempt,        do not retry the model at all --
   fed the exact findings, then         pause for a human (documented in
   ship the best safe version           Human In The Loop)
   (Section 7)
```

Every layer below exists because collapsing these four into one generic
"retry on failure" policy would either retry things that can never succeed
(asking a model again for a fact it was never given) or fail to retry things
that obviously should (a one-off connection drop that has nothing to do with
the resume at all).

---

## 3. The Four Layers, at a Glance

```text
Layer 1  CrewAI's native agent resilience     "Layer 1 defense" per the
         (max_retry_limit, max_iter,           project's own config comment --
         max_execution_time, max_rpm)          applies inside a CrewAI Task run,
                                                identical across all 8 agents

Layer 2  The direct-provider resiliency        rate limit -> transient retry ->
         stack (src/core/resiliency/)          circuit breaker, composed around
                                                ONE function: request_structured_
                                                output's inner provider call

Layer 3  The malformed-output retry            a hand-written "try twice, then
         (structured_output.py)                raise" loop for schema failures --
                                                deliberately NOT part of Layer 2

Layer 4  The one bounded content-repair         orchestration-node-owned; not
         pass (experience.py, skills.py)       infrastructure at all -- a single
                                                disclosed re-prompt with the exact
                                                findings, or a safe fallback
```

Layers 1-3 are infrastructure: they exist to make an LLM call reliable despite
network and provider flakiness, or despite a model occasionally returning
malformed text. Layer 4 is different in kind — it is quality control, living
entirely in orchestration node code, and it is the layer this project is most
disciplined about *not* letting run away.

---

## 4. Layer 1 — CrewAI's Native Agent Resilience

Every one of the eight CrewAI agents documented in
[Agent Roles §3](agent-roles.md#3-anatomy-of-an-agent-factory--the-common-recipe)
is constructed with the identical `agent_defaults` block from
`settings.yaml`, which the config file itself labels, in a comment, as
**"Layer 1 defense"**:

```yaml
agent_defaults:
  max_retry_limit: 3          # Retries when agent encounters error
  max_rpm: 60                 # Max requests per minute (rate limiting)
  max_iter: 15                # Max iterations before forcing best answer
  max_execution_time: 300     # Max seconds per task (5 minutes)
  respect_context_window: true # Auto-summarize to prevent context overflow
```

This is CrewAI's own, framework-native resilience surface, applied uniformly
regardless of which of the eight roles is running. `max_retry_limit` governs
retries CrewAI performs internally when an agent's own tool-call loop hits an
error; `max_iter` caps how many reasoning iterations CrewAI allows before
forcing a best-effort answer rather than looping indefinitely; `max_rpm` and
`max_execution_time` are per-agent throttles independent of anything this
project wrote itself. This layer is entirely CrewAI's implementation — this
codebase only supplies the numbers, uniformly, through the same
`defaults.max_retry_limit` / `defaults.max_iter` fields every agent factory
reads (see [Agent Roles §3](agent-roles.md#3-anatomy-of-an-agent-factory--the-common-recipe)).

---

## 5. Layer 2 — The Direct-Provider Resiliency Stack

`src/core/resiliency/` is a small, purpose-built resilience library this
project wrote for **direct** LLM provider calls — calls made outside CrewAI's
own agent loop, specifically the tool-layer `request_structured_output`
gateway documented in [Tool Contracts §7](tool-contracts.md#7-the-llm-gateway--where-judgment-engines-actually-call-a-model).
`resilient_llm_call()` is a decorator that composes three independent
concerns around one function, in a specific, deliberate order.

```text
   resilient_llm_call()(func)
             |
             v
   OUTERMOST:  circuit breaker   -- fails FAST if the provider is known-down;
                                    does not even attempt a call
             |
             v
   MIDDLE:     transient retry    -- Tenacity-driven; retries ONLY a narrow,
               (exponential        named set of exceptions, with exponential
                backoff + jitter)  backoff + random jitter between attempts
             |
             v
   INNERMOST:  rate limiter       -- a per-provider token-bucket gate; every
                                    individual attempt (including retries)
                                    passes through this before the real call
             |
             v
          func(*args, **kwargs)   -- the actual provider call
```

### 5.1 Rate Limiting — The Innermost Gate

`rate_checker_for(provider, calls_per_minute)` returns a shared, per-provider
token-bucket checker built on the `ratelimit` library. Calling it either
succeeds silently or raises `RateLimitException`; on that exception, the
wrapper sleeps for the window's remaining time (`period_remaining`, falling
back to a computed `60 / calls_per_minute` if that attribute is absent) and
burns the token exactly once more before proceeding. This is the innermost
layer because it governs *every individual attempt*, including ones Tenacity
retries — a retried call still has to clear the same rate gate as the first
attempt.

### 5.2 Transient Retry — Tenacity, and a Narrow Exception Allowlist

The retry layer is deliberately narrow about what it will retry:

```python
RETRYABLE_EXCEPTIONS = (TimeoutError, ConnectionError, RateLimitException)
```

Nothing else is retried at this layer — not a validation error, not a
malformed-JSON error, not a business-logic exception. This is the load-bearing
design decision of the whole layer: it exists exclusively for failures that
are about the *network path to the provider*, never about what the provider
said. The backoff itself is exponential with random jitter
(`wait_exponential(...) + wait_random(0, 1)`), and the decorator is configured
`reraise=True` — meaning that once the attempt budget (`resilience.retry_max_attempts`,
default `3`) is exhausted, the **original** exception type propagates, not an
opaque Tenacity wrapper. That matters downstream: whatever catches this
failure — ultimately the three-way [failure taxonomy](orchestration-graph.md#9-failure-taxonomy--three-kinds-of-the-run-stopped)
— sees a genuine `TimeoutError` or `ConnectionError`, not a framework artifact
it has to unwrap first.

### 5.3 Circuit Breaking — Stop Hammering a Provider That's Down

`breaker_for(provider, policy)` returns one shared `pybreaker.CircuitBreaker`
per provider name, created on first use and reused for the life of the
process. After `circuit_breaker_failure_threshold` (default `5`) consecutive
failures, the breaker opens and every subsequent call fails immediately with
`CircuitBreakerError` — no network attempt is even made — until
`circuit_breaker_timeout` seconds (default `60`) pass and the breaker allows a
trial call through. One detail is easy to miss and important: the breaker is
constructed with `exclude=[RateLimitException]`, meaning a self-imposed rate
limit never counts as a circuit-breaker failure. The breaker exists to detect
that the *provider* is unhealthy; throttling ourselves is an expected,
self-inflicted pause, not evidence of that.

### 5.4 Composition Order and the Correlation ID

The three layers are composed circuit-breaker-outermost specifically so that
a known-down provider is detected *before* spending a retry budget on it —
there is no value in Tenacity's exponential backoff retrying into a breaker
that's already going to reject the call. `resilient_llm_call` also binds a
short correlation ID via `structlog.contextvars` for the duration of the
entire decorated call — every log line emitted by any of the three inner
layers, across every attempt, carries the same ID, then it's unbound in a
`finally` block so it never leaks into unrelated log lines afterward.

---

## 6. Layer 3 — The Malformed-Output Retry

Layer 2 protects the network path; it says nothing about what happens when
the network path succeeds perfectly and the provider simply returns text that
doesn't parse as the requested schema. That is a **separate** failure mode,
handled by a separate, narrower mechanism inside `_request_structured_output`
itself (documented in full in
[Tool Contracts §7](tool-contracts.md#7-the-llm-gateway--where-judgment-engines-actually-call-a-model)):

```python
for attempt in range(2):
    raw_output = structured_llm.call(messages)
    try:
        return parse_structured_output(raw_output, output_model)
    except ValueError as parse_error:
        logger.warning(...)
raise RuntimeError(f"{output_model.__name__}: model returned malformed output twice; aborting")
```

This loop lives *inside* the same function Layer 2 wraps, which means the two
layers are stacked but never confused with each other: Layer 2's Tenacity
retry only fires on `TimeoutError` / `ConnectionError` / `RateLimitException`,
so a malformed-JSON `RuntimeError` from this inner loop is not something Layer
2 will retry again — it propagates straight out. Two attempts, hand-written,
scoped to exactly one failure class (an LLM returning something a strict
parser can't validate), and nothing more.

---

## 7. Layer 4 — The One Bounded Content-Repair Pass

The most disciplined layer in the system is the one with no shared
infrastructure at all: the single repair pass documented in full in
[Orchestration Graph §7](orchestration-graph.md#7-the-no-retry-loops-philosophy)
and [§11](orchestration-graph.md#11-case-study-how-experience-optimization-actually-runs).
When the Experience Optimizer's rewrite fails a truth or quality floor, or the
Skills Optimizer's evidence audit confidently flags an unsupported skill, the
system spends **exactly one** additional model call — fed the precise
findings that need fixing — and then commits to whichever of the two attempts
is safe to ship, falling back to the pre-rewrite source rather than looping
again. There is no exception type driving this repair, no exponential
backoff, no circuit breaker. It is a deliberate, disclosed, single re-prompt
written directly into the orchestration node, chosen specifically because
this is a *content quality* judgment, not an infrastructure failure — and
content-quality judgments are exactly the place an unbounded retry loop would
be most tempting, and most dangerous.

---

## 8. Case Study: request_structured_output's Three Stacked Layers

Tracing one real call through `request_structured_output` shows Layers 2 and
3 stacked correctly, each declining to retry what the other layer owns:

```text
   request_structured_output(ReviewResult, rubric, input)
             |
             v
   ensure_token_budget(...)            -- fails immediately if oversized;
                                           not a retry surface at all
             |
             v
   _request_structured_output(...)     -- decorated with @resilient_llm_call()
             |
        [Layer 2: circuit breaker check]
             |
        [Layer 2: Tenacity retry loop, up to 3 attempts]
             |    each attempt:
             |      [Layer 2: rate-limit gate]
             |            |
             |            v
             |      structured_llm.call(messages)   -- the real provider call
             |            |
             |     TimeoutError/ConnectionError/RateLimitException?
             |            |
             |       YES -> Layer 2 retries (backoff + jitter)
             |       NO, but malformed JSON? -> Layer 3's inner loop retries
             |                                   ONCE more, INSIDE this same
             |                                   Layer-2-wrapped function call
             |       NO, valid JSON -> return validated output_model instance
             v
   caller receives either a validated instance, or a raised exception whose
   TYPE tells the caller which layer gave up (TimeoutError/ConnectionError
   from Layer 2 exhausting its attempts; RuntimeError from Layer 3 exhausting
   its two tries)
```

A caller inspecting the exception type after a failure can tell, without any
special-casing, which layer failed and why — an infrastructure failure and a
schema failure never look the same, because the two retry mechanisms that
handle them were never merged into one.

---

## 9. Where CrewAI's Retries and the Resiliency Stack Never Meet

It's worth being explicit that Layer 1 (CrewAI's native agent resilience) and
Layer 2 (`src/core/resiliency/`) are **completely independent implementations**
protecting two different call paths that never overlap. Every one of the
eight production agents makes its LLM calls through CrewAI's own `Crew(...).kickoff()`,
governed entirely by `agent_defaults` — CrewAI's own retry/timeout machinery,
configured once, uniformly, from outside CrewAI's code. `resilient_llm_call`
is never applied to an agent's `kickoff()` call anywhere in the active
codebase; its only production use is around the tool-layer
`request_structured_output` gateway. (A legacy, non-production file —
`skills_optimizer_agent_OBSELETE.py` — did apply `resilient_llm_call` directly
to hand-rolled LLM calls before that agent was rebuilt into the current
CrewAI-based `skill_optimizer` package; it is not part of the active pipeline.)

```text
   AGENT PATH (8 active agents)          TOOL / JUDGMENT-ENGINE PATH
   -------------------------------        -------------------------------
   Crew(...).kickoff()                    request_structured_output(...)
   governed by CrewAI's OWN internals,     governed by src/core/resiliency,
   configured via agent_defaults           a stack THIS project wrote
   (Layer 1)                               (Layer 2 + Layer 3)

   the two never call into each other -- they are separate resilience
   implementations for separate call paths, not one shared mechanism
```

Neither path is "more correct" than the other — CrewAI's own framework already
solves retry/timeout/iteration limits reasonably well for its own execution
loop, so this project did not reimplement that; the tool-layer gateway, by
contrast, makes raw provider calls with no framework underneath it at all, so
this project built the resilience that call path needed.

---

## 10. The "No Unbounded Retry" Rule, Recap

Every layer above has a hard ceiling, stated as a number, never as "keep
trying until it works":

```text
Layer 1  max_retry_limit: 3, max_iter: 15          -- CrewAI's own bounds
Layer 2  retry_max_attempts: 3 (Tenacity)          -- exhausted -> reraise
Layer 3  range(2)                                   -- exhausted -> RuntimeError
Layer 4  exactly one repair attempt                 -- exhausted -> safe fallback
```

This is the same discipline documented from the orchestration side in
[Orchestration Graph §7](orchestration-graph.md#7-the-no-retry-loops-philosophy):
nowhere in this system does a failure get retried indefinitely in the hope
that persistence alone will eventually produce a passing result. Every layer
either succeeds within its bound, or hands a clearly-typed failure to
whatever's watching next.

---

## 11. Design Rule — Classify Before You Retry

```text
Is the failure about REACHING the provider (timeout, dropped connection,
provider-side rate limit)?
  YES -> Layer 2 already handles this if the call goes through
         request_structured_output. If you're adding a NEW direct provider
         call path, wrap it with @resilient_llm_call() rather than writing
         a new retry loop.

Is the failure about the SHAPE of what came back (not valid JSON, doesn't
match the expected schema)?
  YES -> a narrow, small-fixed-count retry (Layer 3's pattern: try N times,
         then raise a typed error) -- never route this through Layer 2's
         exception allowlist, and never make N unbounded.

Is the failure a CONTENT judgment (a draft reads as generic, a rewrite
fails a truth or quality floor)?
  YES -> at most ONE disclosed repair attempt, fed the exact findings,
         with a known-safe fallback if the repair also fails (Layer 4's
         pattern). Never loop an LLM against its own quality gate.

Is the failure a MISSING FACT only the candidate can supply?
  YES -> this is not a retry at all. Pause for a human
         (see Human In The Loop).
```

---

## 12. Future Considerations

**Whether Layer 2's resiliency stack should be extended to the agent path.**
Today, CrewAI's own `agent_defaults` is the only resilience layer protecting
the eight production agents' LLM calls, and it is a black box this project
does not control the internals of. Whether the direct-provider stack's
circuit breaker specifically — which CrewAI's own retry settings have no
equivalent for — would add value on the agent path (so a known-down provider
fails every agent fast, not just tool-layer calls) is an open question worth
deciding deliberately rather than leaving as an accident of which path
happened to need custom resilience first.

**Whether the circuit breaker's per-provider state should be visible at the
orchestration level.** `get_resilience_stats(provider)` already exposes each
provider's breaker state, failure count, and threshold, but nothing in
`src/orchestration/` currently reads it. Whether a run that hits an open
circuit breaker should surface that as a distinct, named condition in the
[failure taxonomy](orchestration-graph.md#9-failure-taxonomy--three-kinds-of-the-run-stopped)
— rather than being indistinguishable from any other `CircuitBreakerError`
bubbling up as an unhandled exception — is worth resolving as the system's
provider-failure handling matures.

**Whether Layer 4's single-repair pattern should become a named, reusable
primitive.** The Experience Optimizer and Skills Optimizer both implement the
same shape independently — one attempt, one disclosed repair, one safe
fallback — as hand-written node logic rather than a shared abstraction. Given
that exactly two roles already need this pattern, whether a third would
justify extracting it into one shared helper (without losing the
role-specific fallback semantics each currently has) is worth watching for
rather than deciding preemptively.
