# Idempotency
## What "Safe to Retry" Actually Means at Each Layer of This System

> **Scope:** The LangGraph merge model, the PII mapping store, deterministic
> experience/bullet IDs, checkpoint archiving, and the output-path/result-persistence
> layer — and the places this system deliberately chose *not* to be idempotent.
> **Audience:** Contributors retrying a failed stage; anyone asking "if I run this
> twice, what actually happens the second time?"

---

## Table of Contents

1. [What Problem Idempotency Solves](#1-what-problem-idempotency-solves)
2. [The Core Distinction — Idempotent WITHIN a Run vs. ACROSS Runs](#2-the-core-distinction--idempotent-within-a-run-vs-across-runs)
3. [Node-Level Idempotency — Replace, Never Append](#3-node-level-idempotency--replace-never-append)
4. [Storage-Key Idempotency — Sets, Guarded No-ops, Best-Effort Deletes](#4-storage-key-idempotency--sets-guarded-no-ops-best-effort-deletes)
5. [Deterministic IDs Within a Run — experience_id and bullet_id](#5-deterministic-ids-within-a-run--experience_id-and-bullet_id)
6. [Case Study: Where This System Deliberately Chooses NOT to Be Idempotent](#6-case-study-where-this-system-deliberately-chooses-not-to-be-idempotent)
7. [The One Place Idempotency and the Response Cache Meet](#7-the-one-place-idempotency-and-the-response-cache-meet)
8. [Design Rule — Classifying a New Piece of State or Storage](#8-design-rule--classifying-a-new-piece-of-state-or-storage)
9. [Future Considerations](#9-future-considerations)

---

## 1. What Problem Idempotency Solves

An LLM pipeline fails in inconvenient places — mid-render, after quality
evaluation but before rehydration, or in the middle of a candidate's paused
clarification loop. What happens when that failed stage runs again is a
question this system cannot avoid answering, because [Retry Strategy](retry-strategy.md)
guarantees that *something* will be retried, at nearly every layer, as a
matter of course. If a retried stage silently duplicates a skills section, or
resurrects a PII mapping that was already restored, or corrupts a paused run's
checkpoint, the candidate loses trust in the artifact regardless of whether
the retry itself "worked." Idempotency is the property that makes retrying
*safe* rather than merely *possible*.

The subtlety this document exists to untangle is that "idempotent" does not
mean one thing everywhere in this codebase — it means a different, specific
guarantee at each layer, and at least two layers were built to deliberately
**not** be idempotent, for good reasons that are worth stating explicitly
rather than discovering by surprise.

---

## 2. The Core Distinction — Idempotent WITHIN a Run vs. ACROSS Runs

The single most important thing to get right before looking at any specific
mechanism:

```text
   WITHIN one run (one run_id / LangGraph thread_id):
     retrying a NODE against the same upstream state must be safe --
     it must replace its own output, never duplicate or corrupt it

   ACROSS runs (calling the pipeline's own entry point twice):
     tailor_resume() is NEVER idempotent by design -- every call mints
     a brand-new run_id (uuid4().hex, runner.py) and does fresh work,
     with no deduplication against any previous run, ever
```

Everything in Sections 3-5 is about the first guarantee. Section 6 is
specifically about the second — and about two more places this system takes
the same "do not silently reuse a prior result" stance even *within* what
looks, at first glance, like it should be a retry-safe boundary.

---

## 3. Node-Level Idempotency — Replace, Never Append

Every LangGraph node returns a partial dict that LangGraph merges into
`ResumeEnhancementPipelineState` (see [State Management §4](state-management.md#4-the-merge-model--why-nodes-return-partial-dicts)).
Because a node owns a small, fixed set of state fields and always returns a
**complete replacement** for those fields — never a delta or an append — a
retried node is safe by construction:

```text
   retry optimize_skills
     -> returns {"optimized_skills": <a whole new OptimizedSkillsSection>}
     -> REPLACES the field entirely; the previous attempt's section
        is simply overwritten, never appended to or merged with

   retry assemble_ats_resume
     -> returns {"optimized_resume": <a whole new AtsOptimizedResume>}
     -> same replacement guarantee
```

A rerun of any node produces a fresh, complete value for exactly the field(s)
it owns. There is no accumulation to worry about because there is nothing
resembling a list-append or in-place mutation anywhere in this pattern — every
node's contract is "read some state, compute a whole new value, hand back a
whole new value." This is the same property that makes the
[Orchestration Graph](orchestration-graph.md)'s deterministic ATS patch
(`patch_ats_assembly`) safe to reason about as "exhaustive in one pass": it,
too, returns a complete replacement `optimized_resume`, never a partial patch
applied in place.

---

## 4. Storage-Key Idempotency — Sets, Guarded No-ops, Best-Effort Deletes

Outside typed pipeline state, the run-scoped stores documented in
[State Management](state-management.md) each carry their own, explicit
idempotency guarantee:

```text
save_pii_mapping(run_id, mapping)
  -- a Redis SET keyed by run_id. Calling it twice with the same run_id
     simply overwrites the stored mapping; there is no append, no list
     of historical mappings to accumulate.

delete_pii_mapping(run_id)
  -- documented, in its own docstring, as "Idempotent and best-effort."
     Deleting an already-deleted or never-existing key is a no-op, and
     a Redis outage during cleanup is logged and swallowed rather than
     raised, because cleanup must never mask the real error on a
     failing run.

archive_checkpoint_database(db_path, paused_run_dir)
  -- an explicit GUARDED no-op: "if db_path == target: return" before
     ever attempting the move. This is what makes it safe for a run to
     pause, resume, and pause AGAIN -- the second archive call finds
     the checkpoint file already exactly where it needs to be and does
     nothing, rather than erroring on a file that can't be moved onto
     itself.
```

Each of these three took a different concrete form of idempotence — an
overwriting SET, a "delete of nothing is fine" convention, and an explicit
same-path guard — because each was solving its own specific "what if this
runs twice" question, not applying one generic idempotency library.

---

## 5. Deterministic IDs Within a Run — experience_id and bullet_id

`assign_experience_ids` (`src/tools/engines/document_ingestion/resume_extraction.py`)
stamps every parsed work-experience entry with a code-owned ID immediately
after extraction, and `build_experience_bullet_id` derives a stable per-bullet
ID from it (see [Agent Roles §4.5](agent-roles.md#45-experience-section-optimizer)).
The ID is built from resume order plus durable role fields — company, title,
and start date — normalized and slugified:

```text
   build_experience_id(index, experience)
     -> f"exp_{index:03d}_{company_slug}_{title_slug}_{start_date}"

   own docstring: "so it is readable and repeatable within a run"
   and: "suitable for per-run role correlation"
```

Read those two phrases precisely: this ID scheme is deterministic **within**
one run — the same extracted `Resume` object always yields the same IDs no
matter how many times downstream code reads it — but it is explicitly scoped
to *one run's* correlation needs (routing a candidate's clarification answer
back to the exact bullet that asked the question, across a pause/resume
boundary), not offered as a cross-run content hash. A second, independent
`tailor_resume()` call on the identical source resume will, in practice, very
likely produce the same IDs (extraction runs at temperature `0.0` — see
[Agent Roles §9](agent-roles.md#9-per-role-temperature--tuning-determinism-vs-creativity))
— but nothing in this scheme *guarantees* that the way a SHA256 hash of file
contents would. It is deterministic enough for its actual job — and its
actual job is entirely intra-run.

---

## 6. Case Study: Where This System Deliberately Chooses NOT to Be Idempotent

### 6.1 Rendered Artifacts and Persisted Results Are Timestamped, Never Overwritten

`resume_filename` (`src/tools/engines/document_rendering/output_paths.py`) is
explicit, in its own docstring, about a design choice that runs directly
counter to a naive idea of idempotency: *"The timestamp makes repeated runs
land on distinct, sortable names."* Every render — even a render of the exact
same final `Resume` object, retried after a transient failure — produces a
brand-new file:

```text
   <candidate>_<designation>_<YYYYMMDD>_<HHMMSS>.<ext>
                              ^^^^^^^^^^^^^^^^^^
                              guarantees two renders of the SAME resume
                              never collide on the SAME filename
```

`_persist_result` in `runner.py` follows the identical pattern for the JSON
`OrchestrationResult` written after every run:
`run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json`. Neither of these is
"idempotent" in the classical sense of "the same operation performed twice
yields the same end state" — performed twice, they yield **two** files. That
is the correct behavior for a user-facing deliverable: a candidate's resume
file is never silently overwritten by a later run, even an identical one. The
safety property this system actually wants here isn't "retrying produces no
new artifact" — it's "retrying never destroys a prior one." Timestamped,
append-only naming is how that property is achieved, and it is a different,
in some ways stronger, guarantee than idempotency proper.

### 6.2 tailor_resume() Never Deduplicates Against a Previous Run

`tailor_resume(resume_path, jd_path)` mints `run_id = uuid4().hex` on every
single call, unconditionally. There is no lookup against a previous run with
the same resume and job-description paths, no memoization, no "have we
already tailored this exact pair" check anywhere in the runner. Calling this
function twice with byte-identical inputs performs the entire pipeline twice,
end to end, as two entirely independent runs with two different `run_id`s,
two different checkpoint databases, and two different sets of rendered
files. This is a deliberate scope boundary, not an oversight: idempotency in
this system is a property of *retrying a stage within a run*, never a
property of *the pipeline's own entry point*.

### 6.3 Resuming an Already-Completed Run Is a Checked Error, Not a Silent No-Op

`_settle_resumed_run_checkpoint` deletes a paused run's checkpoint database
only once that run reaches a truly terminal disposition (anything other than
`NEEDS_CANDIDATE_INPUT` — see [Orchestration Graph §10](orchestration-graph.md#10-pause-and-resume--how-human-in-the-loop-works-at-the-graph-level)).
Once that deletion happens, calling `resume_paused_run` a second time on the
same `paused_run_path` does not silently succeed or silently no-op — `load_paused_run_state`
raises `FileNotFoundError` because the checkpoint database the manifest points
to no longer exists. This is intentional: a second resume attempt on a run
that already finished is a genuine logic error a caller should see loudly,
not a case to paper over with "well, it's probably fine, nothing happened."
Idempotency does not mean pretending a nonsensical repeated call was harmless
— it means making sure the calls that *are* expected to repeat (a retried
node, a re-archived checkpoint) behave safely, while calls that make no sense
to repeat fail clearly.

---

## 7. The One Place Idempotency and the Response Cache Meet

[Memory Boundaries §9](memory-boundaries.md#9-persistence-that-looks-like-memory-but-isnt--the-response-cache)
documents LiteLLM's disk-backed response cache as explicitly *not* a form of
memory. It is, however, incidentally the one mechanism that makes a full,
byte-identical re-run of the same extraction or judgment call **cheap**
without making it **the same run**: if `enable_cache` is on and a formatter
produces the exact same rendered context twice — whether across two attempts
in one run, or across two entirely separate `tailor_resume()` calls — the
second identical request is served from disk rather than re-billed to the
provider. This does not change anything described in Section 6.2: two
`tailor_resume()` calls on the same input still produce two independent runs,
two `run_id`s, and two sets of output files — the cache only means the *LLM
calls inside* those two runs are cheap the second time, not that the runs
themselves collapse into one.

---

## 8. Design Rule — Classifying a New Piece of State or Storage

```text
Is it a field on ResumeEnhancementPipelineState?
  YES -> the node that owns it must return a COMPLETE replacement value
         on every call, never a delta or an in-place mutation (Section 3).

Is it a key in an external store (Redis, a file, a database)?
  Does writing it twice with the same key mean anything OTHER than
  "the current value is now this"?
    NO  -> use overwrite/SET semantics; that alone gives you idempotency
           (Section 4's PII mapping is the template).
    YES -> you need an explicit guard (Section 4's checkpoint-archive
           same-path check is the template) or a real content-addressed
           key (Section 5's experience_id, understood as intra-run only).

Is it a user-facing deliverable (a rendered file, a persisted result)?
  YES -> do NOT make it idempotent in the classical sense. Timestamp it
         so retries can never destroy or silently replace a prior
         version (Section 6.1). The guarantee you want here is
         "never overwrites," not "same result twice."

Is it the pipeline's own entry point?
  Nothing about it should be idempotent across calls at all (Section 6.2).
  Every call is a new run. If you find yourself wanting to deduplicate
  at this level, that is almost certainly the LLM response cache's job
  (Section 7), not the runner's.
```

---

## 9. Future Considerations

**Whether experience_id's within-run determinism claim should be tested
against real extraction variance.** The ID scheme is deterministic in
principle at temperature `0.0`, but nothing in the codebase today verifies
that two independent extractions of the same source resume actually produce
identical IDs — the docstring's "repeatable within a run" claim has never
been asked to hold *across* two runs, because nothing currently depends on it
doing so. If a future feature ever wants to correlate the same candidate's
role across two separate runs (rather than within one paused/resumed run),
this is the exact boundary that would need re-examining first.

**Whether the timestamped, non-overwriting output convention should extend to
debug checkpoints.** Rendered artifacts and persisted results both timestamp
every write; the debug checkpoint writer (`src/checkpointing.py`, see
[State Management §11](state-management.md#11-two-unrelated-things-both-called-checkpoint--a-naming-trap))
already does the same at the folder level (one timestamped folder per
`run_id`) but sequences files *within* that folder by a per-run counter rather
than a timestamp. Whether that's a meaningful inconsistency worth resolving,
or simply two different naming strategies solving two genuinely different
problems (uniqueness across runs vs. ordering within one run), is worth a
deliberate look rather than an assumption either way.

**Whether a run-level "have I already tailored this exact pair" cache belongs
anywhere.** Section 6.2 documents that `tailor_resume()` deliberately performs
no deduplication across calls today. As usage patterns emerge — a user
accidentally re-submitting the same resume and job description, for
instance — whether that belongs as a new, explicit layer (a run-level cache
keyed on input file hashes, distinct from the LLM response cache) is an
architectural question this document surfaces without answering, since no
observed need for it exists yet.
