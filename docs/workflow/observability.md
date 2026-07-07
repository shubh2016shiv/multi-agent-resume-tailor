# Observability
## What This System Actually Records About Itself, and What It Only Says It Does

> **Scope:** `src/core/logger.py` (structlog, the backbone every module uses),
> `src/observability/` (the LangSmith tracing facade), and the honest gap
> between what this subsystem's own documentation promises and what is
> actually wired into the current, post-refactor architecture.
> **Audience:** Contributors debugging a run after the fact; anyone about to
> reach for `trace_agent` or `log_iteration_metrics` expecting it to already
> be active.

---

## Table of Contents

1. [What Problem Observability Solves](#1-what-problem-observability-solves)
2. [The Core Design Principle — Two Independent Systems, Not One](#2-the-core-design-principle--two-independent-systems-not-one)
3. [structlog — The Backbone Every Module Actually Uses](#3-structlog--the-backbone-every-module-actually-uses)
4. [LangSmith — The Two Layers the Docstring Promises](#4-langsmith--the-two-layers-the-docstring-promises)
5. [log_iteration_metrics — A Third Unwired Mechanism](#5-log_iteration_metrics--a-third-unwired-mechanism)
6. [Case Study: What You'd Actually See in LangSmith Today](#6-case-study-what-youd-actually-see-in-langsmith-today)
7. [The Recurring Pattern — Aspirational Instrumentation From Before the Refactor](#7-the-recurring-pattern--aspirational-instrumentation-from-before-the-refactor)
8. [Debug Checkpoints — Instrumentation Actually Built for This Architecture](#8-debug-checkpoints--instrumentation-actually-built-for-this-architecture)
9. [What Observability Is FOR Here — Semantic Failures, Not Just Crashes](#9-what-observability-is-for-here--semantic-failures-not-just-crashes)
10. [Design Rule — Adding New Observability](#10-design-rule--adding-new-observability)
11. [Future Considerations](#11-future-considerations)

---

## 1. What Problem Observability Solves

A crash is the easy case. A traceback names a file, a line, an exception —
the system tells you exactly where it broke. The failures this pipeline
actually needs to survive are rarely that generous. A run can complete
successfully, produce a syntactically perfect `ProfessionalSummary`, pass
every schema check — and still be wrong in a way no exception was ever going
to surface, because the failure is semantic: a generic summary, a rewrite
that quietly lost a candidate's strongest evidence, a quality score that
degrades without an error anywhere near it. Debugging that kind of failure
requires being able to answer a very specific question after the fact: what
exact context did this one model call actually receive, and what did it
actually return? Observability, in this system, is built to answer that
question — not to tell you a program crashed, but to reconstruct a workflow's
reasoning after it has already finished.

---

## 2. The Core Design Principle — Two Independent Systems, Not One

This codebase runs two entirely separate observability mechanisms side by
side, and understanding them requires keeping them apart:

```text
   structlog (src/core/logger.py)          LangSmith (src/observability/)
   ---------------------------------        ---------------------------------
   ALWAYS ON, no external dependency        OPTIONAL, needs LANGSMITH_API_KEY
   every module calls get_logger(__name__)  a small, dedicated facade package
   writes structured key=value events       sends traces to a hosted dashboard
   to stdout (+ optional rotating file)     (smith.langchain.com)
   THE thing that's actually used           PARTIALLY wired -- see Section 7
   everywhere in this codebase              for what's documented vs. active
```

Neither one depends on the other. structlog runs whether or not
`LANGSMITH_API_KEY` is ever set, and every design decision in it (Section 3)
assumes it is the only thing anyone might be looking at. LangSmith is a
genuinely optional enrichment layer that, when active, adds a hosted,
queryable trace dashboard on top — but as Sections 4 through 7 document in
careful detail, "when active" covers less of this system's actual behavior
today than its own module documentation describes.

---

## 3. structlog — The Backbone Every Module Actually Uses

`src/core/logger.py` is the one file this entire codebase logs through —
every module across every doc in this series calls `get_logger(__name__)`
and emits key-value events (`logger.info("pipeline_stage_started", stage=...,
run_id=...)`), never a hand-built English sentence.

### 3.1 The Processor Chain, Read in Order

A structlog "processor" is a function that takes the in-progress log event
(a plain dict) and returns a modified version; `_build_processors` chains
several together, and the order is meaningful:

```text
   1. merge_contextvars        -- pulls in anything bound via
                                   structlog.contextvars.bind_contextvars(...)
   2. add_log_level            -- adds "level": info/warning/error/...
   3. inject static context    -- service/environment/version/host/pid,
                                   via a CLOSURE, not contextvars (Section 3.2)
   4. add OTel trace context   -- trace_id/span_id, if a span is active
   5. redact sensitive fields  -- masks anything matching a secret-like
                                   field NAME (Section 3.3) -- runs LAST among
                                   the enrichers so it also catches a secret
                                   that arrived via bound contextvars
   6. add ISO-8601 timestamp
   [dev only] 7. add file/function/line -- skipped in JSON/production mode,
                                            because it costs CPU per call for
                                            a benefit only a human reading a
                                            terminal gets
   8. render -- JSON (production) or a colored console (development)
```

### 3.2 Static Context vs. Per-Run Context — Why the Split Exists

The module docstring names a specific bug class it was written to avoid:
service-identity fields (`service`, `environment`, `version`, `host`, `pid`)
are injected through a **processor closure**, not through
`structlog.contextvars.bind_contextvars`. The reason is that anything bound
via contextvars is also erasable by any `clear_contextvars()` call anywhere
else in the app — a perfectly normal thing to do between requests or between
runs. If service identity lived there, a routine per-run context clear would
silently wipe "which service emitted this log" along with it. Baking it into
the processor chain itself means nothing downstream can ever erase it,
because it isn't state that can be cleared — it's part of the pipeline.

### 3.3 Secret Redaction — A Real Safety Net With a Real Gap

`_redact_sensitive_fields` masks any **top-level** event field whose *name*
contains a marker like `password`, `secret`, `api_key`, or `private_key`
(substring, case-insensitive, so `user_password` and `PASSWORD_HASH` are both
caught). This is a real, load-bearing safety net — but its own scope
limitation is stated directly in the code: *"If you log a whole nested
object... fields inside `user_dict` are NOT scanned."* A secret buried inside
a nested payload passed as a single log field survives this redaction
entirely. The discipline this places on every caller is implicit rather than
enforced: flatten sensitive data to top-level keys before logging it, or it
will not be masked.

### 3.4 The run_id Discipline — Manual, Not Automatic (and Why That Matters)

This is worth stating precisely, because the module's own docstring describes
an intended pattern that is not, in fact, how this codebase achieves its
actual result. The docstring's usage example reads:

```python
structlog.contextvars.bind_contextvars(run_id=run_id)  # at run start
# every log emitted from here on automatically includes run_id
structlog.contextvars.clear_contextvars()               # at run end
```

Searching the active codebase for where this is actually called turns up
exactly **one** use of `bind_contextvars` anywhere — and it is not this one.
[Retry Strategy §5.4](retry-strategy.md#54-composition-order-and-the-correlation-id)
documents the only live call: `resilient_llm_call` binds a short-lived
`correlation_id` (an 8-character UUID fragment) for the duration of one
resilience-wrapped call, unrelated to the pipeline's own `run_id`. Nothing in
`src/orchestration/runner.py` — or anywhere else — ever binds `run_id` via
`structlog.contextvars` the way the logger's own documented pattern
describes.

What actually gets `run_id` onto log lines instead is manual, disciplined,
explicit repetition: every node in every stage passes `run_id=state["run_id"]`
as an ordinary keyword argument on every `logger.info(...)` call, by hand,
consistently, across all eleven pipeline nodes. It works — you can grep any
log stream for one `run_id` value and reconstruct a run's whole story — but
it works because of consistent developer habit at every call site, not
because of the automatic mechanism the logging module's own docstring
describes as its design. This is the first of several instances of the same
pattern this document surfaces explicitly in
[Section 7](#7-the-recurring-pattern--aspirational-instrumentation-from-before-the-refactor).

---

## 4. LangSmith — The Two Layers the Docstring Promises

`src/observability/__init__.py` is explicit that agent behavior is meant to
be captured on two complementary layers. It is worth checking both against
what the active codebase actually does.

### 4.1 Layer 1: The Automatic LLM Layer (Actually Wired)

`init_observability()` (`langsmith_backend.py`), called exactly once, at
process startup, from `src/orchestration/runner.py`'s module scope, registers
LiteLLM's built-in LangSmith callback:

```python
if "langsmith" not in litellm.callbacks:
    litellm.callbacks = [*litellm.callbacks, "langsmith"]
```

Because every LLM call in this system — both the CrewAI agent path and the
tool-layer `request_structured_output` path documented in
[Tool Contracts §7](tool-contracts.md#7-the-llm-gateway--where-judgment-engines-actually-call-a-model)
— eventually routes through LiteLLM, this one registration is genuinely
sufficient to make **every** model call in the system stream its prompt,
completion, token counts, cost, and latency to LangSmith automatically, with
zero per-agent code required. This layer is real and active whenever
`LANGSMITH_API_KEY` is set.

### 4.2 Layer 2: The Readable Workflow Layer (Documented, Not Wired)

The second layer — `trace_agent` / `trace_tool` decorators
(`tracing.py`) wrapping functions with `langsmith.traceable` so an agent run
shows up as one named span, with its LLM calls nesting inside it, "giving a
per-agent -> per-LLM-call tree" — is not applied anywhere in the active
pipeline. A search across the codebase for `trace_agent` or `trace_tool`
usage turns up only the retired `_OBSELETE` agent files — none of the eight
active agent factories documented in
[Agent Roles §4](agent-roles.md#4-the-eight-active-roles--a-field-guide), and
none of the eleven orchestration nodes documented in
[Orchestration Graph](orchestration-graph.md), apply either decorator.

The practical consequence: today, LangSmith receives every individual LLM
call as its own trace, ungrouped by which agent or which pipeline stage made
it. The "per-agent -> per-LLM-call tree" the module's own documentation
describes as the dashboard experience does not exist yet for this
architecture's active code paths — see [Section 6](#6-case-study-what-youd-actually-see-in-langsmith-today)
for what you would actually see instead.

### 4.3 Fail-Open Design — Tracing Can Never Break a Run

Whatever else is true about which layers are wired, the fail-open discipline
across this whole package is real and consistent: `init_observability`
returns `False` (never raises) if tracing is disabled, the API key is
missing, or `litellm` fails to import; `build_traced_function` returns the
original, undecorated function if tracing is off or `langsmith` isn't
installed; `log_iteration_metrics` (Section 5) catches any exception from the
LangSmith SDK and logs a warning rather than propagating it. The design rule
stated directly in `langsmith_backend.py`'s docstring — *"Never raise into
the pipeline"* — holds everywhere this package is actually exercised, even in
the places where its intended behavior isn't fully wired in.

---

## 5. log_iteration_metrics — A Third Unwired Mechanism

`log_iteration_metrics` (`iteration_metrics.py`) is built to record custom,
domain-specific metrics — a quality score, an improvement delta, an issue
count — that CrewAI and LangSmith's automatic capture doesn't produce on its
own, always logging to structlog first and, when a LangSmith run is active,
attaching the same metrics to that run's metadata. Searching for callers of
this function across the active codebase turns up, again, only the retired
`_OBSELETE` agent files. No node in the current pipeline — including
`resume_quality.py`, which computes exactly the kind of scores this function
was built to surface (`overall_quality_score`, `accuracy_score`,
`relevance_score`, all documented in [Evaluation](evaluation.md)) — calls it.
Those scores are logged today through ordinary `logger.info("quality_scores_computed",
...)` calls (structlog only), which is a perfectly sound way to record them —
it simply means this particular piece of built, tested, fail-open
infrastructure is not the mechanism actually doing that recording.

---

## 6. Case Study: What You'd Actually See in LangSmith Today

Putting Sections 4 and 5 together, here is the honest, concrete difference
between the dashboard experience the package's documentation describes and
what a real run produces:

```text
   WHAT THE DOCS DESCRIBE:                  WHAT ACTUALLY HAPPENS TODAY:
   ------------------------                  ------------------------------
   One "chain" span per agent run            No agent-level span at all
   (trace_agent), with every LLM call          (trace_agent is never applied)
   nested cleanly underneath it
                                              Every LLM call from every stage
   Custom quality/iteration metrics            appears as its own independent
   attached to the current run's                trace via the automatic LiteLLM
   metadata (log_iteration_metrics)             callback -- ungrouped by agent
                                                 or pipeline stage

                                              Quality scores and iteration
                                                 detail live in structlog JSON
                                                 lines, not LangSmith run
                                                 metadata
```

None of this means LangSmith is not useful today — the automatic LLM-call
tracing (Section 4.1) genuinely captures prompt, completion, token, cost, and
latency data for every call in the system, which is real, valuable signal.
It means the specific "readable workflow" experience — grouped by agent,
enriched with custom domain metrics — described in this package's own
documentation is aspirational for the current architecture rather than
already delivered, and a contributor reading only the module docstring would
reasonably expect more than what a real trace currently shows.

---

## 7. The Recurring Pattern — Aspirational Instrumentation From Before the Refactor

This is the third and fourth time this documentation series has found the
same shape of gap, and it is worth naming as one pattern rather than three or
four unrelated surprises:

```text
   MECHANISM                          DOCUMENTED PURPOSE              STATUS
   ---------                          -------------------              ------
   track_agent_tokens                 estimate input tokens around      built, never
   (Memory Boundaries §11)            an agent execution boundary       called in
                                                                          production

   trace_agent / trace_tool           per-agent LangSmith spans,        built, only
   (Section 4.2, this doc)            "readable workflow" tree          _OBSELETE
                                                                          files use it

   log_iteration_metrics              custom quality/iteration          built, only
   (Section 5, this doc)              metrics on the LangSmith run      _OBSELETE
                                                                          files use it

   structlog run_id auto-binding      "bind once, appears on every      documented,
   (Section 3.4, this doc)            log line automatically"           never called;
                                                                          achieved
                                                                          manually instead
```

Each of these was almost certainly built for, or alongside, the earlier
monolithic-agent architecture documented in
[Agent Roles](agent-roles.md) — a design where a handful of large agent files
called these instrumentation helpers directly. The refactor that produced the
current eight-role, node-per-stage architecture replaced *how* agents are
built and *how* they're called, but did not carry every piece of
instrumentation across that boundary with it. None of these four are broken —
they still work exactly as designed if you call them — they are simply not
called by anything in the active pipeline today. Naming this pattern once,
here, is meant to save a future contributor from independently rediscovering
each instance and wondering whether they're looking at a bug.

---

## 8. Debug Checkpoints — Instrumentation Actually Built for This Architecture

It's worth contrasting the previous section with the one piece of
instrumentation that *is* fully wired into the current pipeline:
`src/checkpointing.py`'s debug input/output capture, documented in full in
[State Management §11](state-management.md#11-two-unrelated-things-both-called-checkpoint--a-naming-trap).
Unlike the four mechanisms above, this one is called from exactly the two
real LLM entry points that exist in the *current* architecture —
`run_agent_task` and, via that same call, every one of the eight active agent
factories — and its own docstring's usage example matches what the code
actually does today. When `DEBUG_CHECKPOINTS=1` is set, every agent call in
every node writes a paired `INPUT`/`OUTPUT` text file, synchronously,
regardless of which of the eight roles made the call. This is the
instrumentation layer to reach for when the question is "what exact context
did the model receive and what did it return" — not the LangSmith workflow
tree, which cannot currently answer that question at the per-agent level for
the reasons in Section 6.

---

## 9. What Observability Is FOR Here — Semantic Failures, Not Just Crashes

Given everything above, the practical guidance for debugging this pipeline
follows directly from which mechanism actually answers which question:

```text
"Did the run crash, and where?"
  -> the three-way failure taxonomy (Orchestration Graph §9) plus ordinary
     structlog output; a real traceback for a genuine bug, a typed
     PipelineQualityGateError/AgentOutputError for the other two categories.

"What exact context did agent X receive on this run, and what came back?"
  -> DEBUG_CHECKPOINTS=1 (Section 8) -- this is the ONE mechanism built
     specifically to answer this question for the current architecture.

"What were this run's quality scores, routing decisions, and stage timings?"
  -> structlog JSON lines, filtered by run_id (Section 3.4) -- every node
     documented in Orchestration Graph logs pipeline_stage_started /
     pipeline_stage_completed, routing decisions, and quality scores this way.

"What did this LLM call cost, and how long did it take?"
  -> LangSmith, if LANGSMITH_API_KEY is set (Section 4.1) -- real per-call
     data, just not yet grouped by agent (Section 6).
```

The first question a semantic failure should prompt — per this package's own
guiding philosophy — is never "why did the model do that?" It is "what exact
context did the model receive?" Debug checkpoints and structlog's run_id-filtered
JSON lines are, today, where that question actually gets answered in this
codebase.

---

## 10. Design Rule — Adding New Observability

```text
Does this need to be captured for EVERY run, with zero external dependency?
  YES -> log it through structlog (get_logger(__name__).info(...)), with
         run_id passed explicitly as a kwarg (Section 3.4) -- don't assume
         it will be bound automatically, because nothing binds it
         automatically today.

Is this genuinely about an LLM call's cost, latency, or token usage?
  YES -> you likely don't need to add anything -- the automatic LiteLLM ->
         LangSmith callback (Section 4.1) already captures this for every
         call in the system, agent-path and tool-path alike.

Are you about to reach for trace_agent, trace_tool, track_agent_tokens, or
log_iteration_metrics because a docstring says it does what you need?
  STOP -> verify it's actually called from the current pipeline first
          (Section 7). If it isn't, decide explicitly whether you are
          wiring it in for real, or building the narrower thing you
          actually need with structlog instead.

Is this instrumentation meant to answer "what did the model literally see
and return" for offline debugging?
  YES -> this is what DEBUG_CHECKPOINTS (Section 8) already does, uniformly,
         for every agent call. Check whether it already answers your
         question before adding a new mechanism.
```

---

## 11. Future Considerations

**Whether the four unwired mechanisms documented in Section 7 should be wired
in, or removed.** `trace_agent`/`trace_tool`, `log_iteration_metrics`, and
`track_agent_tokens` are all well-built, fail-open, and ready to use — they
are simply not connected to anything in the current eight-role architecture.
Each is a small, deliberate decision away from either being wired into the
active node/agent call sites (closing the gap between documented and actual
behavior) or being retired alongside the `_OBSELETE` files that are their only
remaining callers. Leaving them in an indefinite third state — built,
documented as active, actually dormant — is the option most likely to mislead
the next contributor who reads their docstrings at face value.

**Whether `run_id` should, in fact, be bound via `structlog.contextvars` at
the top of `tailor_resume()` and `resume_paused_run()`.** Section 3.4
documents that the logger module's own intended pattern for this is real and
would remove a meaningful amount of repetition across eleven node files, each
of which currently passes `run_id=` by hand on every log call. Given that
`run_id_binding`'s module-global mechanism ([State Management §10](state-management.md#10-run_id_binding--why-a-contextvar-would-have-been-wrong-here))
was deliberately *not* built on `ContextVar` because of CrewAI's worker-thread
behavior, whether the same threading concern would affect a `structlog.contextvars`
binding at the top of the runner is worth checking explicitly before wiring
it in, rather than assuming the two cases are equivalent.

**Whether LangSmith's readable-workflow layer should be revived now, given
how much per-role structure the current architecture actually has.** The
current eight-role, node-per-stage design (documented across
[Agent Roles](agent-roles.md) and [Orchestration Graph](orchestration-graph.md))
arguably has *cleaner* natural span boundaries than the monolithic
architecture `trace_agent`/`trace_tool` were likely built for — each of the
eight `create_*_agent()` factories and each of the eleven orchestration nodes
is already a well-defined unit. Wiring `trace_agent` onto each factory (or
`trace_tool` onto each node function) today would likely be a smaller, more
mechanical change than it would have been against the old architecture, and
would close the exact gap documented in Section 6.
