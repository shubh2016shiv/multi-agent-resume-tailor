# `run_id_binding`, explained from zero

You should be able to read this top to bottom, never feel lost, and end up
knowing exactly what this folder does, what it doesn't do, where code
actually starts running, and what the "algorithm" is. No prior knowledge of
this codebase assumed.

---

## Part 0: The problem, before any code

This app runs one big pipeline: turn a resume + a job description into a
tailored resume. Every time that pipeline starts, the app makes up a unique
ID for that one attempt. Call it the **`run_id`**. Think of it as a claim
ticket: everything that happens during this run should be traceable back to
that one ticket number.

Almost every step of the pipeline can just read this ticket directly, because
the pipeline passes a shared bag of data (called **state**) from step to
step, and `run_id` lives in that bag.

```
step 1 ──state["run_id"]──► step 2 ──state["run_id"]──► step 3 ...
```

Easy, except for one step. The **resume ingestion** step (convert the file,
then redact personal info, then extract structured data) doesn't run as an
ordinary Python function call inside the pipeline. It runs as a **CrewAI
agent**, which calls **tools** on its own, often on a different worker
thread, and those tool functions are never handed the pipeline's `state`
object. They have no built-in way to see `run_id`.

That matters because one of those tools has to **redact PII** (names,
emails, phone numbers) and store a mapping of "placeholder to real value" so
it can be put back later. That mapping has to be saved under the current
run's ticket number, or nothing later can find it, and cleanup can't delete
the right thing.

So the real question this package answers is:

> "While the ingestion agent is working, how does a tool, running on a
> thread that was never handed pipeline state, find out what the current
> `run_id` is?"

**That single question is this entire package's job.**

---

## Part 1: One running example, used the whole way through

We'll follow one concrete run through every section below:

> You call `tailor_resume(resume.pdf, job.txt)`.
> The pipeline creates `run_id = "abc123..."`.
> It reaches the ingestion step, which kicks off a CrewAI resume-parser
> agent.
> Mid-run, that agent calls the tool **"Redact PII from Resume Markdown."**
> That tool needs to save its redaction mapping under `"abc123..."` in
> Redis.

Watch `"abc123..."` travel: pipeline, then this package, then the tool, then
Redis.

---

## Part 2: What this package is, in one sentence each

- **What it is:** a small, temporary hand-off spot. While ingestion is
  running, it holds the current `run_id` so ingestion tools can ask for it.
- **What it is not:** it doesn't create the `run_id`, doesn't manage
  pipeline state, doesn't store PII, and isn't used by most of the
  pipeline's other steps.

| You might assume...                                        | Actually...                                                                                          |
|--------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| "It creates the `run_id`."                                   | No. The pipeline's entry point (`tailor_resume`, elsewhere) creates it and puts it in state.            |
| "Every pipeline step uses this to get `run_id`."              | No. Almost every step just reads `state["run_id"]` directly. Only ingestion needs this package, because only ingestion's tools can't see state. |
| "This is the same as a logging `correlation_id`."             | No. Logging has its own, separate correlation mechanism. This is *pipeline run identity for PII scoping*, a different concern, on purpose. |
| "It uses `ContextVar` like a web framework would."            | No, deliberately not. CrewAI can run tools on worker threads that don't inherit the calling thread's `ContextVar`s, so a `ContextVar` would silently look empty inside a tool. This package uses a plain shared variable instead, precisely because that's visible across threads. |
| "It stores or redacts the PII itself."                        | No. It only hands tools the *key* (`run_id`) they need to talk to the actual PII storage, which lives elsewhere. |
| "It's safe if two resumes are being tailored at the same time in one process." | Not today. There's only one shared slot for the current `run_id`; the code explicitly documents this as a known limitation. |

Keep one phrase in your head: **a narrow bridge for ingestion tools, not a
general "current run" system for the whole app.**

---

## Part 3: The files, read in the order that builds understanding

Only two real code files, plus an index. Read in this order:

```
1. exceptions.py        > what "failure" looks like
2. current_run_id.py    > the entire engine (bind + get)
3. __init__.py           > the public index
```

### 1. `exceptions.py`: "what does failure look like?"

```python
class MissingRunIdError(RuntimeError):
    """Raised when code asks for the current run_id outside a binding scope."""
```

In our example: if the redaction tool asked for the current `run_id` and
*nobody had bound one yet*, this error fires instead of quietly returning
`None` or an empty string. That's deliberate: better a loud, obvious
failure than PII silently saved under a missing or wrong key.

### 2. `current_run_id.py`: the entire engine

Everything real happens in this one file. It keeps:

- one module-level "slot" (`_bound_run_id`) that holds the current `run_id`
  (or nothing)
- a `threading.Lock` so two threads can never read/write that slot at the
  exact same instant and see a half-updated value
- `bind_run_id(run_id)`, which fills the slot for the duration of one `with`
  block, and always puts back whatever was there before, even on error
- `get_current_run_id()`, which reads whatever is currently in the slot

**Why not just use `ContextVar`** (Python's normal per-task "current
context" tool)? Because of how CrewAI actually runs tools:

```
  main thread (running the ingestion node)      CrewAI tool worker thread
  ─────────────────────────────────────         ──────────────────────────
  with bind_run_id("abc123..."):
      kick off the agent  ───────────────────►  the tool actually executes
                                                  HERE, on a different thread
                                                       │
                                          a ContextVar set on the main
                                          thread would NOT be visible here:
                                          it isn't inherited
                                                       │
                                          but a plain module-level variable
                                          IS shared across every thread in
                                          the same process, so it works
```

That one fact is the entire reason this package looks "old-fashioned"
(module global plus lock) instead of using the more modern `ContextVar`
pattern you might expect.

### 3. `__init__.py`: the index

No new behavior; it just decides what outsiders are allowed to import, and
restates the scope boundary (ingestion tools only, nothing broader) in its
docstring.

```python
from src.core.run_id_binding import (
    MissingRunIdError,
    bind_run_id,
    get_current_run_id,
)
```

---

## Part 4: The entry point (where execution actually starts)

There are **two** sides, and both are live in production, but they live in
*different files outside this package*.

**Writer side, where `run_id` gets bound (one call site):**
in the orchestration ingestion node, around the CrewAI kickoff:

```python
with bind_run_id(state["run_id"]):
    resume = run_agent_task(... extract_resume_content_task ...)
```

**Reader side, where tools ask for it (two call sites, same tools file):**

```python
save_pii_mapping(get_current_run_id(), placeholder_mapping)
assert_extraction_input_redacted(get_current_run_id(), redacted_markdown)
```

Full path for our running example:

```
tailor_resume(resume.pdf, job.txt)
      creates run_id = "abc123..."
      puts it in pipeline state
              │
              ▼
   ingestion node reaches extract_resume(state)
              │
              │  with bind_run_id("abc123..."):   ← WRITE (this package)
              ▼
        CrewAI agent kicks off
              │
              ▼
   tool: "Redact PII from Resume Markdown"
              │
              │  run_id = get_current_run_id()    ← READ (this package)
              ▼
   save_pii_mapping("abc123...", mapping)          ← Redis, a different package
```

**Bind once, around the kickoff. Read inside whichever tools need the
Redis-scoped PII key.** That's the whole entry-point story.

---

## Part 5: The algorithm(s)

### `bind_run_id(run_id)`

```
bind_run_id(run_id)
       │
       ├─ run_id is empty? ──────────────► raise ValueError
       │
       ▼
  lock:
      remember previously_bound_run_id (whatever was there before)
      set the slot to run_id
  unlock
       │
       ▼
  yield   ← caller's code runs here (CrewAI kickoff; tools may read the slot)
       │
       ▼
  finally (runs no matter what, success or exception):
      lock:
          put the slot back to previously_bound_run_id
      unlock
```

Because it always restores the previous value, nested scopes behave the way
you'd hope:

```
with bind_run_id("outer"):
    get_current_run_id() → "outer"
    with bind_run_id("inner"):
        get_current_run_id() → "inner"
    get_current_run_id() → "outer"     # restored automatically
```

### `get_current_run_id()`

```
get_current_run_id()
       │
       ▼
  lock:
      read the slot
  unlock
       │
       ├─ slot was empty (None)? ─────► raise MissingRunIdError
       │
       ▼
  return the value
```

There's no cleverness here: no reaching into pipeline state, no talking to
Redis. It's just: "what's currently sitting in the shared slot?"

---

## Part 6: Who depends on whom

```
        tailor_resume()  (elsewhere, creates run_id)
                │
                ▼
        pipeline state["run_id"]
                │
                ▼
   ingestion node: extract_resume(state)
                │
                │  with bind_run_id(...)
                ▼
      ┌───────────────────────────┐
      │      run_id_binding       │
      │   (this package's slot)   │
      └─────────────┬─────────────┘
                     │ get_current_run_id()
                     ▼
      ingestion tools (redact / extract)
                     │
                     ▼
      Redis PII mapping store  (a different package)

  exceptions.py  ← raised by get_current_run_id() when nothing is bound
  __init__.py    ← re-exports bind_run_id / get_current_run_id / MissingRunIdError
```

**Rule of thumb:** this package is a short hallway connecting one
orchestration node to a couple of tool functions. Everything interesting
about what the `run_id` actually *unlocks*, the PII mapping itself, lives
outside this folder.

---

## Part 7: The biggest misconception

**Wrong mental model:**

```
  "run_id_binding is how the whole multi-agent pipeline tracks its run."

  Stage 1 ──uses this package──┐
  Stage 2 ──uses this package──┼──► one global run-tracking system
  Stage 3 ──uses this package──┘
```

**Actual mental model:**

```
  Almost the entire pipeline:
      state["run_id"]  ────────────────► every node, logs, cleanup, etc.
      (this package is never touched)

  The one narrow gap: CrewAI ingestion tools can't see state:
      state["run_id"]
            │
            ▼
      bind_run_id(...)          ← this package, only around ingestion
            │
            ▼
      get_current_run_id() inside redact / extract tools
            │
            ▼
      Redis PII mapping, keyed by that run_id
```

If you deleted this package, the pipeline would still know its `run_id`
almost everywhere, via state, as normal. What would actually break is much
narrower: the redaction and extraction tools would have no way to name the
Redis key for their PII mapping during the CrewAI kickoff.

It's also easy to confuse this with two *other* "current run" ideas that
already exist elsewhere and are deliberately kept separate:

- **Logging's `correlation_id`**, a different id, for tying log lines
  together, unrelated to PII scoping.
- **LangSmith trace/span identity**, observability tracing, not pipeline
  run identity.

This package touches none of those. It has exactly one job.

---

## Part 8: Cheat sheet

**Bind around an ingestion kickoff (writer side):**

```python
from src.core.run_id_binding import bind_run_id

with bind_run_id(state["run_id"]):
    # any CrewAI tool called inside here can read the run_id
    resume = run_agent_task(...)
```

**Read inside a tool (reader side):**

```python
from src.core.run_id_binding import get_current_run_id, MissingRunIdError

try:
    run_id = get_current_run_id()
except MissingRunIdError:
    # this tool ran outside a bind_run_id() scope: fail closed, don't guess
    raise

save_pii_mapping(run_id, placeholder_mapping)
```

**Nested binding (mostly relevant for tests):**

```python
with bind_run_id("outer"):
    with bind_run_id("inner"):
        assert get_current_run_id() == "inner"
    assert get_current_run_id() == "outer"   # auto-restored
```

---

## Part 9: FAQ

**Q: Where is the entry point?**
Writer: `bind_run_id(...)` wrapped around the CrewAI kickoff in the
ingestion node. Reader: `get_current_run_id()` called from two spots inside
the ingestion tools file (redact, and the "assert input was redacted"
check).

**Q: What are "stages" here?**
Almost every pipeline stage doesn't touch this package at all; it just
reads `run_id` from shared state, which this package has nothing to do
with. Only the *ingestion* stage binds anything, because only its tools run
outside the normal state-passing path.

**Q: Who actually creates the `run_id`?**
Not this package. The pipeline's own entry point creates it (a fresh unique
id) and puts it into shared state before any of this runs.

**Q: Why not just pass `run_id` as a normal function argument to the
tool?**
Because CrewAI tools are invoked by the agent/LLM, not called directly by
your code. You don't want the model responsible for supplying (or possibly
mangling) the real pipeline id. Binding keeps the true id out of the prompt
entirely while still making it available in memory.

**Q: Why a module-level variable instead of something more modern like
`ContextVar`?**
Because CrewAI can execute tools on worker threads that don't inherit the
binding thread's `ContextVar`s. A `ContextVar` would look empty inside the
tool. A plain shared variable (protected by a lock) stays visible across
threads in the same process.

**Q: Does this affect logging or tracing?**
No. Logging correlation and tracing identity are separate systems this
package deliberately does not touch.

**Q: Can two resumes be tailored at the same time in one process?**
Not safely today. There's only one shared slot, so concurrent runs in the
same process would overwrite each other's `run_id`. This is a known,
documented limitation, not an oversight; today's deployment assumes one
active ingestion kickoff per process at a time.

---

## One sentence to keep

**`run_id_binding` is a short-lived, thread-visible parking spot for the
pipeline's `run_id`, used only so CrewAI ingestion tools can key their Redis
PII storage correctly. It creates nothing, and almost no other stage ever
touches it.**