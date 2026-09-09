# `observability`, explained from zero

You should be able to read this top to bottom, never feel lost, and end up
knowing exactly what this folder does, what it doesn't do, where code
actually starts running, and what the "algorithm" is. No prior knowledge of
this codebase assumed.

---

## Part 0 — The problem, before any code

When this app "tailors" a resume, many **agents** (specialized AI workers)
talk to an **LLM** (a model like GPT-4o or DeepSeek) — sometimes many times
each. Without observability, that is a black box:

```
you start a run
      │
      ▼
agents call the LLM… LLM replies… agents call again…
      │
      ▼
you get a resume back
```

You cannot easily answer:

- What prompt did the experience agent actually send?
- What did the model reply?
- How many **tokens** (billing units) did that cost?
- Which step was slow?

**Observability** here means: attach a passive recorder so those questions
have answers on a dashboard (and in local logs), without changing what the
agents *do*.

The dashboard vendor this package talks to is **LangSmith**
(https://smith.langchain.com). A **trace** is one recorded run of work,
shown as a nested tree of **spans** (boxes): agent-level boxes, tool boxes,
and LLM-call boxes.

---

## Part 1 — One running example, used the whole way through

> You call `tailor_resume(resume.pdf, job.txt)`.  
> Importing the orchestration runner calls `init_observability(...)`.  
> Later, the experience stage’s CrewAI agent asks the LLM to rewrite bullets.  
> You want that LLM call’s prompt, reply, tokens, and cost in LangSmith.

We will follow that one call through the package.

---

## Part 2 — What this package is / isn't

- **What it is:** the app’s single import seam for LangSmith setup, optional
  span decorators, and optional iteration metrics.
- **What it is not:** it does not create agents, does not count tokens itself
  (LiteLLM + LangSmith do), and does not replace `structlog` application logs.

| You might assume... | Actually... |
|---------------------|-------------|
| "`@trace_agent` is on every agent today." | **No.** Decorators exist and are tested, but **no production `src/` file applies them right now.** |
| "Without `@trace_agent`, nothing is recorded." | **No.** After a successful `init_observability`, LiteLLM’s `"langsmith"` callback can still record **each LLM call** (Layer 1). |
| "This package measures tokens like `llm_token_tracker`." | **No.** `llm_token_tracker` is a local **input budget gate**. This package is the **real meter** path via LiteLLM → LangSmith. |
| "If LangSmith is down / key missing, the pipeline dies." | **No.** Everything here fails soft and the pipeline keeps running. |
| "Old docs mentioning `agent_orchestrator.py` are still true." | **No.** Startup lives in `src/orchestration/runner.py` now. |

Keep: **Layer 1 wired; Layer 2 + metrics helpers ready but mostly dormant.**

---

## Part 3 — Files in reading order (not alphabetical)

```
1. langsmith_backend.py   → turn tracing on/off; arm LiteLLM
2. tracing.py             → optional @trace_agent / @trace_tool
3. iteration_metrics.py   → optional custom numbers on a span
4. __init__.py            → public facade (what outsiders import)
```

### 1. `langsmith_backend.py` — the ignition switch

Owns `_is_initialized`, `init_observability()`, and
`is_observability_enabled()`. This is where Layer 1 is armed:
`litellm.callbacks` gets `"langsmith"`, and LangSmith settings are copied
into environment variables (the only place in the app that does that
hand-off, because the SDKs only read env vars).

### 2. `tracing.py` — named boxes (Layer 2)

`build_traced_function` / `trace_agent` / `trace_tool` wrap a function with
`langsmith.traceable`, or return it unchanged if tracing is off / SDK
missing. Use these when you want a named parent span around an agent or
helper. **Not applied in production today.**

### 3. `iteration_metrics.py` — custom scores on a span

`log_iteration_metrics` always writes to structlog; if tracing is on, it
also tries to attach metadata to the current LangSmith run tree. Useful
inside critique loops. **Not called from production today.**

### 4. `__init__.py` — the front door

Re-exports the five public names. Callers should import from
`src.observability`, never from backend modules, so the vendor stays
swappable.

---

## Part 4 — The entry point (what actually runs today)

### Wired in production

**`init_observability("resume-tailor-agents")`** runs at **module import**
time in `src/orchestration/runner.py`:

```python
from src.observability import init_observability

init_observability("resume-tailor-agents")
```

So the first time anything imports the runner (CLI, web app, tests that
import it), startup tries to enable LangSmith.

After a successful init, **every LiteLLM-backed LLM call** in the process
can be reported to LangSmith automatically — including CrewAI agents and
the structured-output gateway — **without decorating those functions**.

### Exists but not wired to production call sites

| API | Status |
|-----|--------|
| `@trace_agent` / `@trace_tool` | Exported + unit-tested; **no production decorator usage** |
| `log_iteration_metrics(...)` | Exported + unit-tested; **no production callers** |

Config that controls init:

- `src/config/settings.yaml` → `observability.enabled`, `project`, `endpoint`
- `.env` → `LANGSMITH_API_KEY` (secret; never commit)

---

## Part 5 — The algorithm(s)

### `init_observability(project_name, enabled=True)`

```
init_observability(...)
       │
       ├─ already initialized? ──────────────► return True
       │
       ▼
  read get_config().observability + langsmith_api_key
       │
       ├─ enabled flag false (caller or YAML)? ► log, return False
       │
       ├─ no API key? ───────────────────────► warn, return False
       │
       ├─ cannot import litellm? ────────────► warn, return False
       │
       ▼
  ensure "langsmith" ∈ litellm.callbacks     ← Layer 1 armed
       │
       ▼
  copy settings → env:
      LANGSMITH_TRACING, LANGSMITH_API_KEY,
      LANGSMITH_PROJECT, LANGSMITH_ENDPOINT
       │
       ▼
  _is_initialized = True
  log success → return True
```

### `build_traced_function(run_type, func)` (used by `@trace_agent` / `@trace_tool`)

```
build_traced_function(...)
       │
       ├─ not is_observability_enabled()? ───► return func unchanged
       │
       ├─ cannot import langsmith.traceable? ► warn, return func
       │
       ▼
  return traceable(run_type=..., name=func.__name__)(func)
```

### `log_iteration_metrics(agent, iteration, metrics)`

```
log_iteration_metrics(...)
       │
       ▼
  structlog.info("iteration_metrics", ...)   ← always
       │
       ├─ not is_observability_enabled()? ───► return
       │
       ▼
  try get_current_run_tree()
       ├─ run exists? update metadata
       └─ error? warn (never raise)
```

---

## Part 6 — Who depends on whom / data flow

```
  CLI / web / tests
        │
        ▼
  import src.orchestration.runner
        │
        │  init_observability(...)          ← ONLY production entry today
        ▼
  ┌─────────────────────────────────────┐
  │ langsmith_backend.py                │
  │  litellm.callbacks += "langsmith"   │
  │  env hand-off to LangSmith SDK      │
  │  _is_initialized = True/False       │
  └──────────────┬──────────────────────┘
                 │ is_observability_enabled()
        ┌────────┴────────┐
        ▼                 ▼
  tracing.py         iteration_metrics.py
  (@trace_* —        (log_iteration_metrics —
   ready, unused)     ready, unused)
        │
        │  (if decorated)
        ▼
  named LangSmith spans


  Meanwhile, during a real agent LLM call:

  agent / structured_output
        │
        ▼
  CrewAI → LiteLLM → provider
              │
              └── "langsmith" callback ──► LangSmith (prompt, tokens, cost)
```

---

## Part 7 — Biggest misconception

### Wrong mental model (what the old README implied)

```
  Every agent method has @trace_agent
        │
        ▼
  That is how tokens get recorded
```

### Actual mental model today

```
  runner import
        │
        ▼
  init_observability ──► LiteLLM "langsmith" callback   ← tokens/cost HERE
        │
        ▼
  agents run without @trace_agent
        │
        ▼
  LangSmith may show LLM runs (Layer 1)
  but fewer named per-agent parent boxes (Layer 2 unused)

  Optional future:
  decorate run_agent_task / node helpers with @trace_agent
  to restore the nice tree grouping
```

Contrast with `llm_token_tracker`: that package **estimates input size and can
block oversized prompts**. Observability **records what the provider actually
did** after the call.

---

## Part 8 — Cheat sheet

**Startup (already done in runner):**

```python
from src.observability import init_observability

init_observability("resume-tailor-agents")
```

**Optional named agent span (not production-wired yet):**

```python
from src.observability import trace_agent

@trace_agent
def run_experience_optimizer(...):
    ...
```

**Optional iteration metrics:**

```python
from src.observability import log_iteration_metrics

log_iteration_metrics(
    "experience_optimizer",
    iteration=2,
    metrics={"quality_score": 78, "improvement_delta": 13},
)
```

**Check live status:**

```python
from src.observability import is_observability_enabled

print(is_observability_enabled())
```

---

## Part 9 — FAQ

**Q: Where is the entry point?**  
`init_observability` at import time in `src/orchestration/runner.py`.

**Q: Where do tokens and cost come from?**  
LiteLLM’s built-in `"langsmith"` callback after init — not from
`llm_token_tracker`, and not from `@trace_agent`.

**Q: Are `@trace_agent` / `@trace_tool` used?**  
Not in production today. They are ready helpers. Best future home: one shared
choke point such as `run_agent_task`, not every agent file.

**Q: Is `log_iteration_metrics` used?**  
Not in production today. Still always safe to call (structlog first).

**Q: How do I turn it off?**  
`observability.enabled: false` in settings, or omit `LANGSMITH_API_KEY`.
Pipeline behavior unchanged.

**Q: Why write env vars inside `init_observability`?**  
LangSmith / LiteLLM callbacks only read configuration from the environment;
this module is the deliberate one-way hand-off from typed settings → env.

**Q: Why call `is_observability_enabled()` instead of reading a bool at import?**  
Init often runs *after* `tracing.py` / `iteration_metrics.py` are imported.
A snapshotted import-time bool would stay `False` forever.

---

## One sentence to keep

**Observability arms LiteLLM→LangSmith once at runner import so real LLM
usage is recorded; named `@trace_*` spans and iteration metrics are optional
helpers that are ready but not production-wired yet.**
