# The Practical Guide to Run ID Flow in Resume Tailor
## How one orchestration `run_id` is created, carried through the pipeline, and exposed to ingestion tools

> **Scope:** `src/core/run_id_binding` and the code that calls it  
> **Audience:** New developers working on orchestration, ingestion, or PII handling  
> **Goal:** Give you a mental model first, then the exact code flow

---

## Table of Contents

1. [What Problem This Package Solves](#1-what-problem-this-package-solves)
2. [The One-Sentence Mental Model](#2-the-one-sentence-mental-model)
3. [The Full Flow, End to End](#3-the-full-flow-end-to-end)
4. [Why This Package Exists At All](#4-why-this-package-exists-at-all)
5. [What `run_id_binding` Owns and What It Does Not Own](#5-what-run_id_binding-owns-and-what-it-does-not-own)
6. [The Core API in Plain English](#6-the-core-api-in-plain-english)
7. [The Architecture Walkthrough, File by File](#7-the-architecture-walkthrough-file-by-file)
8. [What Happens During One Real Resume Ingestion Run](#8-what-happens-during-one-real-resume-ingestion-run)
9. [Failure Modes and Safety Rules](#9-failure-modes-and-safety-rules)
10. [Common Misunderstandings](#10-common-misunderstandings)
11. [When You Should Change This Package](#11-when-you-should-change-this-package)

---

## 1. What Problem This Package Solves

The orchestration pipeline has a `run_id`.

That `run_id` is created at the top of the pipeline and stored in LangGraph state.

Most code can read that state directly.

But the document-ingestion tools do not run in that same direct state-reading path.
They run inside CrewAI tool execution, and that execution does not have direct access
to the LangGraph state object that contains `run_id`.

This creates a small but important gap:

- the pipeline knows the `run_id`
- the ingestion tools need the `run_id`
- the ingestion tools cannot directly read the pipeline state

`src/core/run_id_binding` exists to bridge exactly that gap.

It does not do anything broader than that.

---

## 2. The One-Sentence Mental Model

`run_id_binding` temporarily makes the current orchestration `run_id` available to ingestion tools while the ingestion node is running.

If you remember only one sentence, remember that one.

---

## 3. The Full Flow, End to End

Here is the whole story in one compact diagram:

```text
tailor_resume(...)
    |
    | creates run_id
    v
ResumeEnhancementPipelineState["run_id"]
    |
    | passed into LangGraph nodes
    v
extract_resume(state)
    |
    | with bind_run_id(state["run_id"]):
    v
CrewAI ingestion agent kickoff
    |
    | tool calls happen here
    v
get_current_run_id()
    |
    +--> save_pii_mapping(run_id, mapping)
    |
    +--> assert_extraction_input_redacted(run_id, markdown)
```

This means:

- the `run_id` starts in orchestration
- `run_id_binding` does not create the `run_id`
- `run_id_binding` only exposes the current `run_id` to the ingestion-tool layer

---

## 4. Why This Package Exists At All

Without this package, a developer might ask:

> “Why not just pass `run_id` as a normal function argument?”

That would be the cleanest option if the tool layer were ordinary Python calls under
our control.

But this project uses CrewAI tool execution during resume ingestion. The tool wrappers
are called by the agent runtime, not by our orchestration code in a simple direct chain.

So the real situation is:

- the ingestion node knows the `run_id`
- the CrewAI tool wrapper needs the `run_id`
- the tool wrapper is not being called with `run_id` explicitly

That is why this package uses a temporary binding instead of ordinary parameter passing.

---

## 5. What `run_id_binding` Owns and What It Does Not Own

This package owns exactly one concern:

- binding the current orchestration `run_id` so ingestion tools can read it

This package does **not** own:

- full orchestration state
- LangGraph state management
- logging context
- tracing context
- LangSmith run/span identity
- the logging `correlation_id`
- PII storage itself

That distinction matters because the name `run_id_binding` is intentionally narrow.
If a future change is about logging, tracing, or general state propagation, it probably
does **not** belong in this package.

---

## 6. The Core API in Plain English

The package exposes three public names:

### `bind_run_id(run_id)`

Plain English:

> “For the duration of this block, treat this `run_id` as the current active pipeline run.”

This is used by the orchestration layer before it enters the CrewAI ingestion kickoff.

### `get_current_run_id()`

Plain English:

> “Give me the `run_id` for the pipeline run currently in progress.”

This is used by ingestion tools that need to attach their work to the current run.

### `MissingRunIdError`

Plain English:

> “You asked for the current `run_id`, but no run is bound right now.”

This is a deliberate fail-closed behavior. The system never guesses a run id.

---

## 7. The Architecture Walkthrough, File by File

### A. The run starts in `src/orchestration/runner.py`

The pipeline entry point is `tailor_resume(...)`.

That function creates the orchestration run id:

```python
run_id = uuid4().hex
```

Then it places that value into the LangGraph state:

```python
initial_state = {
    "run_id": run_id,
    ...
}
```

This is the source of truth.

Important:

- this is the real pipeline run identifier
- `run_id_binding` does not invent a second run id
- the same run id is later used for PII mapping storage and cleanup

### B. The run id travels through LangGraph state

The shared state type lives in `src/orchestration/state.py`.

That state contains:

```python
run_id: str
```

So from this point onward, every orchestration node can read `state["run_id"]`.

### C. The ingestion node binds the run id

The key bridge happens in `src/orchestration/nodes/ingestion.py`.

The important line is:

```python
with bind_run_id(state["run_id"]):
```

This means:

- the ingestion node takes the orchestration-owned `run_id`
- it temporarily binds it as the current active run
- then it starts the CrewAI ingestion task inside that binding scope

Everything inside that `with` block can rely on `get_current_run_id()`.

### D. The ingestion tools read the bound run id

The downstream read side is in `src/tools/agent_tools/ingestion_tools.py`.

Two places use it:

```python
save_pii_mapping(get_current_run_id(), placeholder_mapping)
```

and:

```python
assert_extraction_input_redacted(get_current_run_id(), redacted_markdown)
```

These tools do not know anything about LangGraph state directly.

They only know:

- “there is a current run”
- “I can ask `get_current_run_id()` what that run is”

That is the whole point of this package.

### E. The package implementation lives in `src/core/run_id_binding/current_run_id.py`

This file contains:

- a module-global `_bound_run_id`
- a lock `_run_id_binding_lock`
- `bind_run_id(...)`
- `get_current_run_id()`

The module-global is not fancy. It is intentionally simple.

The logic is:

1. bind a run id before the ingestion kickoff
2. keep it available while tools run
3. restore the previous value when the block exits

That is all.

---

## 8. What Happens During One Real Resume Ingestion Run

Let us walk through one realistic run in plain English.

### Step 1: the pipeline starts

`tailor_resume(resume_path, jd_path)` is called.

The runner creates:

- one `run_id`
- one initial pipeline state

Example:

```text
run_id = "a1b2c3d4..."
```

### Step 2: the ingestion node begins

LangGraph eventually reaches the resume ingestion node.

That node reads:

```text
state["run_id"] == "a1b2c3d4..."
```

### Step 3: the node binds that run id

The node enters:

```python
with bind_run_id(state["run_id"]):
```

Now the system has one active bound run id for this ingestion scope.

### Step 4: CrewAI tools run inside that scope

The resume parser agent kicks off and calls tool wrappers such as:

- convert document to markdown
- redact PII
- extract structured resume

### Step 5: tools that need the run id read it

When the redaction tool stores placeholder mappings, it calls:

```python
get_current_run_id()
```

That returns:

```text
"a1b2c3d4..."
```

The PII mapping store can now save the mapping under the correct run.

### Step 6: the scope exits

When the `with bind_run_id(...)` block finishes, the package restores the previous
binding value.

This is how the run id does not leak into unrelated work later.

---

## 9. Failure Modes and Safety Rules

This package is intentionally strict.

### If `get_current_run_id()` is called outside a binding scope

It raises `MissingRunIdError`.

Why this is good:

- it fails fast
- it avoids saving data under the wrong run
- it avoids “silent success with wrong state”

### If `bind_run_id("")` is called with an empty string

It raises `ValueError`.

Why this is good:

- an empty run id is not meaningful
- accepting it would hide a caller bug

### If concurrent pipeline runs happen in the same process

This is the main limitation to understand.

Today the implementation stores the bound run id in a module-global variable.
That is acceptable only because the current process model assumes one active resume
ingestion kickoff at a time per process.

So the current safety rule is:

- safe for the current one-active-ingestion-at-a-time assumption
- not a general solution for many concurrent in-process runs

That limitation is documented in the code on purpose.

---

## 10. Common Misunderstandings

### “Is this the same as logging `correlation_id`?”

No.

`correlation_id` is a logging concern used in resiliency/logging code.

`run_id` here is a pipeline run identifier used to tie ingestion-side PII work back
to the current orchestration run.

They solve different problems.

### “Is this the same as LangSmith trace identity?”

No.

LangSmith tracing is part of observability. It tracks spans and traces.

This package only exposes the current pipeline `run_id` to ingestion tools.

### “Does this package own pipeline state?”

No.

LangGraph state owns pipeline state.

This package only bridges one field from that state into the tool execution layer.

### “Why not put this in logger or tracing code?”

Because that would mix concerns.

This package is about one thing:

- run id binding for ingestion tools

That clear separation is intentional.

---

## 11. When You Should Change This Package

You should change this package when:

- ingestion tools need clearer run-id access rules
- the binding flow becomes hard to understand
- the concurrency model changes
- CrewAI starts supporting a cleaner built-in context propagation path

You should probably **not** change this package when:

- you are adding logging features
- you are changing LangSmith tracing
- you are changing general orchestration state handling
- you are adding unrelated runtime context

If a proposed change makes this package broader than “bind the current run id for
ingestion tools,” stop and reconsider the design first.

---

## Final Summary

If you want the shortest possible explanation:

1. `tailor_resume()` creates the real `run_id`.
2. LangGraph state carries that `run_id`.
3. The ingestion node binds that `run_id`.
4. Ingestion tools read it with `get_current_run_id()`.
5. PII mapping operations use it to stay attached to the correct pipeline run.

That is the entire architecture.
