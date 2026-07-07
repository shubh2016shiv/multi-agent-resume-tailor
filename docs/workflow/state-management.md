# State Management
## Every Kind of State in Resume Tailor, and Where Each One Lives

> **Scope:** `src/orchestration/state.py`, LangGraph's checkpointer and its
> msgpack allowlist, the paused-run persistence layer, the run-scoped PII
> mapping store, and the public `OrchestrationResult` returned to callers.
> **Audience:** Contributors adding a new field anywhere in the pipeline;
> anyone asking "where does this piece of data actually live, and for how long?"

---

## Table of Contents

1. [What Problem State Management Solves](#1-what-problem-state-management-solves)
2. [The Core Design Principle — One State Object, Explicit Ownership](#2-the-core-design-principle--one-state-object-explicit-ownership)
3. [ResumeEnhancementPipelineState — The In-Flight Ledger](#3-resumeenhancementpipelinestate--the-in-flight-ledger)
4. [The Merge Model — Why Nodes Return Partial Dicts](#4-the-merge-model--why-nodes-return-partial-dicts)
5. [Typed State as an Internal API, Not Just Storage](#5-typed-state-as-an-internal-api-not-just-storage)
6. [Durable State — LangGraph's Checkpointer and the msgpack Allowlist](#6-durable-state--langgraphs-checkpointer-and-the-msgpack-allowlist)
7. [The Paused-Run Directory — Three Files, One Resumable Unit](#7-the-paused-run-directory--three-files-one-resumable-unit)
8. [run_id — One Identifier Threading Through Four Separate Subsystems](#8-run_id--one-identifier-threading-through-four-separate-subsystems)
9. [PII Mapping — State Deliberately Kept OUT of the Typed Pipeline State](#9-pii-mapping--state-deliberately-kept-out-of-the-typed-pipeline-state)
10. [run_id_binding — Why a ContextVar Would Have Been Wrong Here](#10-run_id_binding--why-a-contextvar-would-have-been-wrong-here)
11. [Two Unrelated Things Both Called "Checkpoint" — A Naming Trap](#11-two-unrelated-things-both-called-checkpoint--a-naming-trap)
12. [OrchestrationResult — The Public State Shape vs. the Internal One](#12-orchestrationresult--the-public-state-shape-vs-the-internal-one)
13. [Design Rule — Where Should a New Piece of State Live?](#13-design-rule--where-should-a-new-piece-of-state-live)
14. [Future Considerations](#14-future-considerations)

---

## 1. What Problem State Management Solves

A resume tailoring run is not one function call — it's a process that can span
seconds for a clean pass, or days when a candidate needs to think about how to
answer a clarification question before the run can finish. That span forces a
question most single-shot LLM pipelines never have to answer honestly: when
the process isn't running, where does everything it has learned so far
actually live? The answer in this codebase is not one answer — it's four,
and knowing which of the four applies to a given piece of data is what this
document is about.

There are, distinctly:

```text
1. IN-FLIGHT STATE     -- ResumeEnhancementPipelineState, alive only while
                           the graph is actively executing in this process
2. DURABLE STATE        -- the same state, snapshotted to disk by LangGraph's
                           checkpointer so a paused run can resume later,
                           possibly in a different process entirely
3. OUT-OF-BAND STATE    -- the PII mapping: real, sensitive, and deliberately
                           NOT part of the typed pipeline state at all
4. PUBLIC STATE         -- OrchestrationResult, the one shape a caller ever sees,
                           distinct from anything above
```

Conflating these — treating "state" as one undifferentiated blob — is exactly
how a system leaks PII into a checkpoint file, or loses a candidate's paused
run because "state" was assumed to live only in memory. This codebase keeps
the four separate on purpose.

---

## 2. The Core Design Principle — One State Object, Explicit Ownership

`src/orchestration/state.py` states its own invariant directly: every field in
`ResumeEnhancementPipelineState` starts `None`, a node sets its own output
field(s) and returns a partial dict, and "downstream nodes must only read a
field after the node that produces it has run." The graph topology documented
in [Orchestration Graph](orchestration-graph.md) is what *enforces* that
invariant — a node simply has no edge into it before its inputs exist — so the
read-order rule is a structural guarantee, not a convention a contributor has
to remember to respect.

```text
   every field starts None
           |
           v
   exactly ONE node is ever responsible for setting a given field
           |
           v
   the graph's edges are the only proof a field is safe to read
   (no edge into a node => that node's inputs are guaranteed to exist)
```

---

## 3. ResumeEnhancementPipelineState — The In-Flight Ledger

```text
   --- inputs (set once, by the runner, before graph.invoke()) ---
   run_id, resume_path, jd_path, clarification_answers

   --- Stage 1: parallel ingestion ---
   resume, job_description

   --- Stage 2: sequential gap analysis ---
   requirement_match_report   (a ReviewResult -- see Tool Contracts)
   alignment_strategy

   --- Stage 3: parallel content generation ---
   professional_summary, optimized_experience, optimized_skills
   experience_clarifications   (HITL: questions for the candidate)

   --- Stage 4: sequential ATS assembly ---
   optimized_resume

   --- Stage 5: sequential quality assurance ---
   quality_report, rendered_structure_evaluation

   --- Stage 5b: conditional ATS recovery ---
   human_review_required

   --- Stage 6/7: conditional render ---
   rendered_artifacts
```

This is the same spine documented from the routing side in
[Orchestration Graph §3](orchestration-graph.md#3-graph-topology--the-full-pipeline-at-a-glance);
here the point is different. Notice that `requirement_match_report` is typed
as a `ReviewResult` — the exact shared contract from
[Tool Contracts](tool-contracts.md) — not a bespoke shape invented for this one
field. State fields reuse the system's existing typed vocabulary wherever
possible rather than inventing a parallel one.

---

## 4. The Merge Model — Why Nodes Return Partial Dicts

A LangGraph node never mutates the shared state object directly — it returns a
plain dict containing only the fields it produced, and LangGraph merges that
dict back into the running state. This is a small mechanical detail with a
real consequence: it makes every node's contract explicit and inspectable.
Reading a node function's `return` statement tells you exactly which fields it
claims ownership of, without needing to trace mutation through the rest of the
function body.

```text
   node function
     reads:  state["resume"], state["job_description"]     (already merged in)
     computes: ...
     returns: {"alignment_strategy": ..., "requirement_match_report": ...}
                        |
                        v
              LangGraph merges this partial dict into
              the one shared ResumeEnhancementPipelineState
```

The merge model is also what makes checkpointing tractable. Because state is
always a plain, serializable dict of typed fields — never a class instance
with open-ended internal mutation — there is no bespoke serialization story to
write per node. Whatever LangGraph's checkpointer can serialize once, generically,
covers every stage of the pipeline.

---

## 5. Typed State as an Internal API, Not Just Storage

A state field is not incidental storage — it is a contract between whichever
nodes read it and the one node that writes it. `optimized_experience` is not a
blob of text; it is the exact interface between the Experience Optimizer node,
ATS assembly, quality evaluation, and rendering, and changing its shape means
changing that interface for every one of those consumers at once, the same way
changing a function's public signature would.

This typed boundary is also what makes *partial* recovery possible. If
rendering fails, the system does not need to regenerate the resume from
scratch — the typed `optimized_resume`, `quality_report`, and every upstream
section output are still sitting in state, ready to feed a repaired render
node. State typing is what turns "the pipeline crashed" from "start over" into
"resume from exactly where the typed data stops."

---

## 6. Durable State — LangGraph's Checkpointer and the msgpack Allowlist

In-memory state alone cannot survive a pause that lasts days, or even a
process restart mid-run. `src/hitl/professional_experience/persistence.py`
opens a `SqliteSaver` — LangGraph's own supported durable checkpointer —
keyed by `run_id` as the LangGraph `thread_id`. Every state transition is
snapshotted to that SQLite file as the run progresses, which is what makes
`interrupt()` (documented in [Orchestration Graph §10](orchestration-graph.md#10-pause-and-resume--how-human-in-the-loop-works-at-the-graph-level))
resumable from a cold start in a different process entirely: resuming a run
means reopening this same file and handing LangGraph a `Command`, not
replaying computation.

This durability comes with a real, easy-to-miss maintenance obligation.
LangGraph's `JsonPlusSerializer` refuses to deserialize a custom type — any
Pydantic model, any `Enum` — unless it is explicitly named on an allowlist,
precisely so that a malicious or corrupted checkpoint file cannot be crafted
to instantiate an arbitrary Python class on load. `src/hitl/professional_experience/checkpoint_types.py`
is that allowlist, and its own docstring is blunt about the consequence of
letting it drift: *"Keep this list in sync with `ResumeEnhancementPipelineState`'s
field types... add an entry here whenever a new model or enum becomes reachable
from a state field, including through nesting."* Today a missing entry only
produces a warning (`LANGGRAPH_STRICT_MSGPACK=false`); once that enforcement
becomes strict by default, a missing entry becomes a hard failure the moment
someone tries to resume a paused run.

```text
   add a new field to ResumeEnhancementPipelineState
              |
              v
   does its type (or anything nested inside it) introduce a NEW
   Pydantic model or Enum not already on CHECKPOINT_ALLOWED_MSGPACK_MODULES?
              |
        yes ------------------------------> no
        |                                    |
        v                                    v
   add it to checkpoint_types.py        nothing to do
   or a future resume of a paused
   run through THIS field will fail
```

Every state-shape change is therefore, quietly, also a serialization-contract
change — this allowlist is the second place (after `state.py` itself) that has
to know about a new type reachable from state.

---

## 7. The Paused-Run Directory — Three Files, One Resumable Unit

When a run pauses at the candidate-clarification gate
(see [Orchestration Graph §10](orchestration-graph.md#10-pause-and-resume--how-human-in-the-loop-works-at-the-graph-level)),
its durable state is not left as a bare SQLite file — it is packaged into one
self-contained, movable directory:

```text
   <paused_run_dir>/
     clarifications_sheet.json     -- the questions; the CANDIDATE edits this file directly
     paused_run_manifest.json      -- run_id, original resume/JD paths, status,
                                       and the filenames needed to resume
     checkpoints.sqlite3           -- the LangGraph checkpoint history itself,
                                       produced entirely by SqliteSaver -- nothing
                                       in this codebase hand-serializes it
```

`archive_checkpoint_database` moves the checkpoint file into this directory
once the runner has closed its connection (it is a no-op if the file is
already there — the case of a run that pauses a second time after an earlier
resume). `load_paused_run_state` does the inverse: read the manifest, confirm
the checkpoint file actually exists at the path the manifest names, and reopen
it with the same allowlisted serializer. The entire resumability guarantee
rests on this directory being self-contained — it can be copied, backed up, or
handed to a different machine, and `resume_paused_run` needs nothing else to
pick a run back up.

---

## 8. run_id — One Identifier Threading Through Four Separate Subsystems

A single `run_id`, minted once by `tailor_resume()`, is the connective tissue
across every kind of state this document describes — and it is explicitly
*not* the same thing as logging's `correlation_id`, a distinction
`run_id_binding`'s own docstring draws on purpose:

```text
run_id is used, independently, as:

  1. LangGraph's thread_id       -- partitions checkpoint history per run
                                    in the SqliteSaver database
  2. the Redis key suffix         -- scopes the PII placeholder mapping to
                                    exactly this run (Section 9)
  3. the debug checkpoint         -- names the folder under checkpoints/
     folder name                   holding this run's agent I/O text dumps
                                    (Section 11 -- a DIFFERENT "checkpoint")
  4. the run_id_binding           -- lets ingestion tools (running in a
     module-global                 CrewAI worker thread) look up which
                                    run they belong to (Section 10)
```

Four subsystems, one identifier, and no shared implementation between them —
each was built independently against the same `run_id` string, which is
exactly what lets a piece of durable checkpoint state, a Redis-backed PII
mapping, and a debug text dump all be correlated back to one human-legible run
without any of those three systems depending on each other's internals.

---

## 9. PII Mapping — State Deliberately Kept OUT of the Typed Pipeline State

The PII placeholder mapping (`[PERSON_1]` → the candidate's real name, and so
on) is real, sensitive state that a run needs across its entire lifetime — from
the moment `resume_content_extractor` redacts it, to the moment
`rehydrate_pii` restores it just before rendering (see
[Orchestration Graph §4](orchestration-graph.md#4-stage-by-stage-walkthrough)).
It would be structurally simple to add it as one more field on
`ResumeEnhancementPipelineState`. It is deliberately not there.

The reason follows directly from Section 6 and 7: the entire pipeline state is
designed to be checkpointed to a SQLite file that can sit on disk for days and
be archived into a paused-run directory a human might open, copy, or inspect.
Putting real PII values inside that same durable snapshot would quietly defeat
the redaction pipeline's entire purpose — the one artifact explicitly designed
to be portable and inspectable would become the one place a candidate's real
name and contact details actually live in plaintext.

Instead, the mapping lives in `src/core/pii_mapping_store/`, a Redis-backed
store with its own, narrower guarantees:

```text
   save_pii_mapping(run_id, mapping)
     -- stored with a TTL (default 3600s), keyed by run_id, never logged by value

   assert_extraction_input_redacted(run_id, candidate_text)
     -- re-verifies no raw PII value survives in text about to reach an LLM --
        checking a "redaction ran" flag is not enough on its own

   load_pii_mapping(run_id)  /  delete_pii_mapping(run_id)
     -- read once for final rehydration; delete on every terminal disposition
        EXCEPT NEEDS_CANDIDATE_INPUT, where the mapping must survive the pause
        (see Orchestration Graph's pause/resume cleanup rules)

   EVERY path fails CLOSED: RedisUnavailableError if Redis is unset or
   unreachable -- there is no in-memory or skipped fallback, because
   silently losing the mapping would either leak placeholders into the
   final resume or, worse, drop the masking guarantee entirely.
```

This is the clearest example in the whole codebase of a general rule: not
every piece of run-scoped data belongs in the typed pipeline state just
because it's scoped to a run. Data that is sensitive, ephemeral by design, and
must never appear in a portable, human-inspectable snapshot gets its own
narrower, fail-closed store instead.

---

## 10. run_id_binding — Why a ContextVar Would Have Been Wrong Here

Ingestion tools (the four tools wired onto the Resume Content Extractor —
see [Agent Roles §4.1](agent-roles.md#41-resume-content-extractor)) need to
know the current `run_id` to scope PII storage, but the tools themselves are
plain functions with no access to graph state — CrewAI calls them directly.
The idiomatic Python answer to "make a value available to code deeper in the
call stack without threading it through every argument list" is a
`contextvars.ContextVar`. This codebase deliberately does not use one, and
says exactly why in `current_run_id.py`'s module docstring: *"CrewAI executes
tool calls in worker threads that do not inherit the binding thread's
contextvars."* A `ContextVar` set on the thread that calls `Crew.kickoff()`
would simply not be visible inside the worker thread CrewAI spins up to
actually execute a tool — the binding would silently fail to propagate, which
is a much worse failure mode than an explicit, working alternative.

```text
   bind_run_id(run_id):   [contextmanager]
     acquire lock -> save previous binding -> set module-global _bound_run_id
     yield (ingestion node's whole scope runs here, INCLUDING
            CrewAI worker threads it spawns for tool calls)
     acquire lock -> restore previous binding

   get_current_run_id():  [called from inside a tool, possibly on a worker thread]
     read _bound_run_id under lock
     raise MissingRunIdError if nothing is bound (fail closed, not "unknown")
```

The module is candid about the resulting limitation rather than hiding it: a
module-global is shared across the whole process, so two concurrent
`tailor_resume()` calls in the same process would collide on this binding.
The code accepts that limitation explicitly, with a dated `TODO` recording
why: this process runs one resume-ingestion kickoff at a time today, so the
tradeoff of a simple, thread-visible module-global outweighs building
per-caller isolation nothing yet needs.

---

## 11. Two Unrelated Things Both Called "Checkpoint" — A Naming Trap

Grepping this codebase for "checkpoint" surfaces two entirely unrelated
systems, and confusing them is an easy mistake for a newcomer to make:

```text
src/checkpointing.py                    LangGraph's SqliteSaver
------------------------                 ------------------------------
DEBUG-ONLY, gated by                     ALWAYS ACTIVE, part of the
DEBUG_CHECKPOINTS=1 env var               production pause/resume mechanism

Writes plain .txt file pairs             Writes a binary SQLite database
(one INPUT + one OUTPUT per               (msgpack-serialized state snapshots,
agent call) for human inspection          allowlisted per Section 6)

Zero production-path effect               Is THE durability mechanism behind
when disabled                             a paused run's ability to resume

Purpose: "what did this agent's           Purpose: "what was the graph's typed
LLM call actually see and return?"        state the instant it paused, and how
                                            do I restore it exactly?"
```

The debug checkpoint writer is careful never to interfere with the pipeline it
observes — every write is wrapped so that an `OSError` is logged and
swallowed rather than raised, because, in its own words, "Never let checkpoint
I/O break the pipeline." It exists purely to answer a debugging question by
writing to disk synchronously (input before the LLM call, output right after),
so the INPUT file is guaranteed to exist even if the agent call itself
crashes. It shares nothing — no code, no format, no lifecycle — with the
`SqliteSaver` checkpoint database described in Sections 6 and 7 beyond the
English word "checkpoint."

---

## 12. OrchestrationResult — The Public State Shape vs. the Internal One

Nothing outside `src/orchestration/` ever sees a raw
`ResumeEnhancementPipelineState`. The public return type of both
`tailor_resume()` and `resume_paused_run()` is `OrchestrationResult`
(`src/data_models/orchestration.py`) — a deliberately narrower, caller-facing
shape:

```text
   ResumeEnhancementPipelineState              OrchestrationResult
   (internal, TypedDict, every field           (public, Pydantic BaseModel,
    starts None, mutated stage by stage         one fully-formed object handed
    across eleven nodes)                        back once and never mutated again)

           |                                             ^
           +---------- derive_run_disposition() ---------+
                       (Orchestration Graph Section 8)
```

`OrchestrationResult` carries `original_resume`, `job_description`, `strategy`,
and then a set of fields that are `None` precisely when the run stopped before
producing them — `optimized_resume`, `quality_report`, and `rendered_artifacts`
are all `None` on a run that paused for clarification, by contract, not by
accident. The one field every caller actually branches on is `disposition`: a
single `RunDisposition` enum value — `RENDERED`, `NEEDS_CANDIDATE_INPUT`,
`NEEDS_HUMAN_REVIEW`, or `QUALITY_GATE_FAILED` — that collapses the internal
graph's entire routing history into the one decision a caller needs to make
next. This is the same principle as [Section 5](#5-typed-state-as-an-internal-api-not-just-storage)
applied at the system boundary: the internal state is optimized for what nodes
need to read and write; the public result is optimized for what a caller needs
to decide, and the two are intentionally not the same shape.

---

## 13. Design Rule — Where Should a New Piece of State Live?

```text
Does more than one node need to read it?
  YES -> it belongs in ResumeEnhancementPipelineState.
         Does its type introduce a new Pydantic model or Enum?
           YES -> add it to CHECKPOINT_ALLOWED_MSGPACK_MODULES too (Section 6).
  NO  -> keep it local to the one node/formatter that needs it while drafting;
         it does not need to be state at all.

Is it sensitive, or does it need TTL/expiry semantics a checkpoint
snapshot cannot give it?
  YES -> it does NOT belong in pipeline state, checkpointed or not.
         Give it its own narrow, fail-closed store (Section 9's PII
         mapping is the existing template for this).

Does a caller outside src/orchestration/ need to see it?
  YES -> it belongs on OrchestrationResult, not on the internal state --
         decide what a caller needs to DECIDE, not what a node needed
         to PRODUCE (Section 12).

Is it only useful for a human debugging one run, never for the
pipeline's own decisions?
  YES -> it belongs in the debug checkpoint writer (Section 11), gated
         behind DEBUG_CHECKPOINTS, never in production state at all.
```

---

## 14. Future Considerations

**Whether the msgpack allowlist should be structurally derived rather than
hand-maintained.** `CHECKPOINT_ALLOWED_MSGPACK_MODULES` is currently a manually
written list that must be kept in sync, by a human remembering to do so,
every time a state field's type changes. Since the actual set of reachable
types is fully determined by `ResumeEnhancementPipelineState`'s own type
annotations, there is an open architectural question of whether this
allowlist should instead be derived automatically from that type graph at
build or import time — turning a silent, easy-to-forget maintenance step into
a structural guarantee that can't drift.

**Whether the run_id_binding concurrency limitation needs to be resolved
before this system supports concurrent runs per process.** The module-global
binding documented in Section 10 is an explicit, accepted tradeoff for a
single-run-at-a-time process. If the architecture ever moves toward serving
multiple concurrent `tailor_resume()` calls from one long-lived process
(rather than one process per run), this binding mechanism becomes a
correctness boundary that has to be redesigned, not merely re-tuned — worth
treating as a prerequisite to that direction rather than something to
discover mid-incident.

**Whether "PII redaction is unavailable" should be a distinct run outcome.**
Today, a Redis outage during a run with `enable_pii_redaction` on fails the
whole run closed (`RedisUnavailableError`), which is the safe choice, but it
means an infrastructure dependency outside the pipeline's own logic can turn
into an opaque pipeline failure indistinguishable, to a caller, from any other
crash. Whether that deserves its own `RunDisposition` value, or its own
documented place in [Failure Handling](failure-handling.md), is an open
question about how visible an infrastructure-level failure mode should be at
the architecture's own boundary.
