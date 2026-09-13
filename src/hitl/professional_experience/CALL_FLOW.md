# Call Flow — the HITL clarification loop, function by function

> Companion to `README.md`. The README explains *concepts*; this document is a
> pure **execution trace** — which function calls which, in what order, and what
> each one hands to the next. Read it with the source open.
>
> Every function name here is real. File paths are repo-relative.

---

## First: why this is not one program run

A normal program runs start to finish in one go. Request in, work happens,
result out — one process, one continuous execution.

A clarification loop cannot work that way, for one reason: the work now includes
**waiting for a human**, and a human might take three days. You cannot hold a
process alive and idle for three days — it ties up a worker, and it loses
everything on a reboot or a deploy.

So the program does something that looks wrong the first time you see it. When
it needs a human answer, it **writes everything it knows to a folder on disk and
exits.** The process ends. Nothing runs. Later, a *different* execution of the
program opens that folder and continues from where the first one stopped.

That is why the full story — "tailor this resume, with one clarification
question in the middle" — is not one program run. It is **three**. Throughout
this document, "journey" just means "one of those three executions."

```
   what triggers it        what it does                          how it ends            where it starts (file · module · function)
   ────────────────        ───────────                           ───────────            ──────────────────────────────────────────

   JOURNEY 1               parse, analyze, rewrite bullets.       writes                 src/orchestration/runner.py
   candidate uploads       one bullet needs a fact only the      paused_run_<id>/        module: src.orchestration.runner  (re-exported: src.orchestration.tailor_resume)
   resume + JD             candidate has -> interrupt().          then PROCESS EXITS     function: tailor_resume()  (def at runner.py:47)
                           (entry: tailor_resume)

        . . . the candidate reads the questions in the UI. hours or days pass.
              NOTHING is running. the folder on disk is the only state. . . .

   JOURNEY 2               takes the submitted answers, writes    returns HTTP 202,      web_app/server.py
   candidate submits       them into the folder (sheet + an       spawns Journey 3,      module: web_app.server
   answers via the web     append-only audit log).               then PROCESS EXITS     function: resume_run(request)  (async def at server.py:83)
   form                    (entry: resume_run)                                          route: POST /api/runs/{run_id}/resume

   JOURNEY 3               opens the folder Journey 1 wrote,      renders the final      src/orchestration/runner.py
   spawned by Journey 2    reloads the frozen state, feeds in     resume, then           module: src.orchestration.runner  (re-exported: src.orchestration.resume_paused_run)
   as a background task    the answers, RE-RUNS the rewrite       PROCESS EXITS         function: resume_paused_run()  (def at runner.py:115)
                           from where Journey 1 stopped.                                 invoked via: web_app/server.py::_execute_resumed_run (BackgroundTask)
                           (entry: resume_paused_run)
```

### The analogy, if the diagram is still abstract

A form that needs a manager's signature:

1. You fill out the form and drop it in the manager's inbox. You go home.
   *(Journey 1 — writes the folder, exits)*
2. Days later the manager signs it and drops it in the "processed" tray. They go
   home. *(Journey 2 — records the answer)*
3. Next morning a clerk picks signed forms out of the tray and files them.
   *(Journey 3 — resumes and finishes)*

Three people, three separate sittings, one physical folder passed between them.
**Nobody stood around waiting.** That is the whole point of the design.

### Why Journeys 2 and 3 are separate

Journey 3's work — reloading state, re-running LLM calls — is slow. It cannot
happen *inside* the HTTP request or the browser would time out. So the HTTP
handler (Journey 2) does only the fast part: save the answers, return `202
Accepted`, and hand the slow part to a background task (Journey 3).

A CLI tool would merge them: `--resume-from <folder>` reads the answers you typed
into the file and resumes, all in one process. The split exists only because
HTTP responses must be fast.

### The one rule to carry into the rest of this document

> **Nothing runs between the journeys.** Each one starts cold, with no memory of
> the others. The only thing connecting them is the `paused_run_<id>/` directory
> on disk. Every "hand-off" you see below is a file being written by one journey
> and read by the next.

---
---

# JOURNEY 1 — Fresh run, up to the pause

## 1.1 The call stack at a glance

```
tailor_resume(resume_path, jd_path)                         runner.py
  │
  ├─ 1. open_checkpoint_database(<run>.sqlite3)             orchestration/checkpointing.py
  ├─ 2. _build_pipeline(checkpointer)                        runner.py -> graph.py
  ├─ 3. _invoke_pipeline(pipeline, initial_state, config)    runner.py
  │       │
  │       └─ pipeline.invoke(...)   [LangGraph runs the graph]
  │             │
  │             ├─ extract_resume ─┐
  │             ├─ analyze_job ────┴─> run_gap_analysis
  │             │                          │
  │             │        ┌─────────────────┼─────────────────┐
  │             │        v                 v                 v
  │             │  write_professional  optimize_experience  optimize_skills
  │             │  _summary                 │
  │             │                           │   <=========  THIS IS THE HITL WORK
  │             │                           v
  │             │                  await_candidate_clarifications
  │             │                           │
  │             │                     interrupt(...)  ──> raises GraphInterrupt
  │             │                           │
  │             │                     LangGraph catches it, writes checkpoint,
  │             │                     returns {"__interrupt__": ...}
  │             v
  ├─ 4. _finalize_pipeline_output(output, ...)              runner.py
  │       │
  │       ├─ _pipeline_interrupted(output)  -> True
  │       ├─ pipeline.get_state(config)     -> the frozen state snapshot
  │       ├─ build ExperienceClarificationPausedRunManifest  (paused_at, expires_at)
  │       ├─ save_paused_run_state(layout, manifest, clarifications)   persistence.py
  │       │     ├─ write_clarification_sheet(layout, clarifications)   persistence.py
  │       │     └─ layout.manifest.write_text(...)
  │       └─ _persist_result(result)         writes run_<timestamp>.json
  │
  ├─ 5. close_checkpoint_database(checkpointer)             checkpointing.py
  └─ 6. _settle_fresh_run_checkpoint(...)                   runner.py
          └─ archive_checkpoint_database(db_path, layout)   persistence.py
                (moves <run>.sqlite3  ->  paused_run_<id>/checkpoints.sqlite3)

PROCESS EXITS. The OrchestrationResult has .paused_run_path set and
.disposition == NEEDS_CANDIDATE_INPUT.
```

## 1.1a The same call stack, explained in plain English

If the tree above is dense, read this instead. It walks the *exact same path*,
top to bottom, in words.

### The one-line summary

`tailor_resume()` is a normal Python function call. It opens a small database,
runs a graph of steps (extract, analyze, rewrite), and — the moment one of those
steps discovers it needs a fact only the human candidate can supply — it does
**not** wait. It freezes everything it has learned into a folder on disk, returns
a result that says "I'm paused, here's the folder," and the process ends.

Nothing is running after that. The folder is the entire state of the world.

### Walking it step by step

**Setup (before any real work):**

1. **`open_checkpoint_database(<run>.sqlite3)`** — Before running a single step,
   the function creates a fresh SQLite file named after this run's id. Think of it
   as a flight recorder. From here on, *every step's output is written to this
   file automatically* as the graph runs. This is what makes a pause survivable:
   when the pause happens, the state is already on disk — nothing extra needs to
   be dumped.

2. **`_build_pipeline(checkpointer)`** — Compiles the graph defined in
   `graph.py` and wires it to that recorder. "Graph" here just means a fixed set
   of steps with arrows between them (do A, then B and C in parallel, then D).
   LangGraph is the library that runs such graphs and knows how to save/reload
   them.

3. **`_invoke_pipeline(...)`** — Presses "go." Control now lives inside LangGraph,
   which executes the steps in dependency order:

   - **`extract_resume`** — parse the uploaded resume file into structured data.
   - **`analyze_job`** — parse the job description the same way.
   - **`run_gap_analysis`** — compare the two: what does the job want that the
     resume doesn't clearly show?
   - Then three steps run **in parallel**, because none depends on the others:
     - `write_professional_summary` — draft the top-of-resume summary.
     - **`optimize_experience`** — rewrite the work-history bullet points for
       impact. **This is the step that can trigger a pause.**
     - `optimize_skills` — tune the skills section for keyword matching.

**The pause itself:**

4. All three parallel steps must finish, then control flows into
   **`await_candidate_clarifications`**. This step asks one question: *did
   `optimize_experience` produce any clarification questions that don't yet have
   answers?* On a fresh run, if the answer is yes, it calls **`interrupt(...)`**.

5. **`interrupt()`** is LangGraph's "stop here" signal. Under the hood it raises
   a `GraphInterrupt` exception. LangGraph catches that exception itself, writes
   a final checkpoint to the SQLite recorder (so the graph can later be resumed
   from exactly this spot), and instead of crashing, returns a normal-looking
   result dictionary that contains a special key: `{"__interrupt__": ...}`.

   So from `_invoke_pipeline`'s point of view, the graph "finished" — it just
   finished with a result that means *paused*, not *done*.

**Reacting to the pause:**

6. **`_finalize_pipeline_output(output, ...)`** inspects that result:

   - **`_pipeline_interrupted(output)`** — sees the `"__interrupt__"` key,
     returns `True`. This branch means "we paused; persist everything."
   - **`pipeline.get_state(config)`** — asks LangGraph for the frozen snapshot of
     all state as it stood at the pause (the parsed resume, the job data, the
     gap analysis, the draft summary, the rewritten bullets, and crucially the
     list of unanswered questions).
   - **Build the manifest** — an `ExperienceClarificationPausedRunManifest`
     records the bookkeeping: which run this is, where the source files were,
     the timestamp it paused at, and an `expires_at` (default 168 hours / 7 days
     later). After that expiry the candidate must start over.
   - **`save_paused_run_state(layout, manifest, clarifications)`** — writes two
     files into the `paused_run_<id>/` folder:
     - `clarifications_sheet.json` — the questions, each with a blank `answer`
       field waiting to be filled.
     - `paused_run_manifest.json` — the manifest from the previous step.
   - **`_persist_result(result)`** — also drops a `run_<timestamp>.json` audit
     record of this run's outcome.

**Cleanup and exit:**

7. **`close_checkpoint_database(checkpointer)`** — releases the SQLite file
   handle so it can be moved.

8. **`_settle_fresh_run_checkpoint(...)`** → **`archive_checkpoint_database(...)`**
   — moves the recorder file from its temporary in-flight location
   (`checkpoints/<run>.sqlite3`) into the paused-run folder, renamed
   `checkpoints.sqlite3`. Now the folder is self-contained: questions + manifest
   + the complete frozen graph state, all in one place.

9. **The function returns.** The `OrchestrationResult` has `.paused_run_path`
   pointing at the folder and `.disposition == NEEDS_CANDIDATE_INPUT`. The web
   layer reads those two fields, shows the candidate the questions, and the
   process exits.

### What's on disk when Journey 1 ends

```
paused_run_<run_id>/
  ├─ clarifications_sheet.json     the questions, every answer still ""
  ├─ paused_run_manifest.json      run id, source paths, paused_at, expires_at
  └─ checkpoints.sqlite3           the entire frozen pipeline state
```

`answers_audit.jsonl` is **not** here yet — it's created by Journey 2 when the
first answer arrives.

### The mental model to keep

`tailor_resume()` behaves like a function that can return in one of two ways:
"here is your finished resume," or "here is a folder — come back when the human
has answered." Everything above is just the machinery that makes the second
kind of return safe: write it all down first, *then* stop.

## 1.2 Step by step

### Step 1 — `tailor_resume()` sets up durability *before* running anything
`src/orchestration/runner.py`

```python
run_id = uuid4().hex                                    # identity for this whole run
checkpoint_db_path = _in_flight_checkpoint_db_path(run_id)   # .../checkpoints/<run_id>.sqlite3
checkpointer = open_checkpoint_database(checkpoint_db_path)  # SqliteSaver + msgpack allowlist
pipeline = _build_pipeline(checkpointer)                # compiles graph.py, bound to this saver
config = {"configurable": {"thread_id": run_id}}        # thread_id == run_id, ALWAYS
```

The checkpointer exists from the first line. Every node's output is written to
SQLite as the graph runs — not only at the pause. By the time `interrupt()`
fires, the state is already durable.

#### Step 1, explained in plain English

The whole point of this step is: **set up the safety net before doing anything
that might need it.** Four small pieces:

- **`run_id = uuid4().hex`** — a random 32-character id, generated once. It is
  the name for *this entire run* — every file, database, and log line for this
  execution is tagged with it. When Journey 3 comes back days later, this id is
  how it finds the right folder and the right slot inside the database.

- **`checkpoint_db_path = _in_flight_checkpoint_db_path(run_id)`** — decides
  *where* the recorder file lives while the run is active: a shared
  `checkpoints/` directory, filename `<run_id>.sqlite3`. "In-flight" because this
  is only its temporary home; if the run pauses, Journey 1's cleanup step moves
  it into the paused-run folder. If the run finishes normally, it's deleted.

- **`checkpointer = open_checkpoint_database(...)`** — creates that SQLite file
  and wraps it in a LangGraph `SqliteSaver`. Two details worth knowing:
  - The DB connection is opened with `check_same_thread=False` because the graph
    runs some steps in worker threads, and all of them share this one recorder.
  - The serializer is given an **explicit allowlist** of every custom class
    (`Resume`, `JobDescription`, `ExperienceBulletClarification`, …) that is
    allowed to be reconstructed when a checkpoint is loaded back. Anything not on
    the list is refused. This is a security measure: a checkpoint file is
    basically pickled program state, and without the allowlist a tampered file
    could make the loader instantiate arbitrary Python classes
    (CVE-2026-28277). A test walks the real graph and fails if the list falls
    behind the state shape.

- **`pipeline = _build_pipeline(checkpointer)`** — compiles the step graph from
  `graph.py` and permanently binds it to *this* recorder. From now on, running
  the pipeline and saving its state are the same action.

- **`config = {"configurable": {"thread_id": run_id}}`** — LangGraph stores
  checkpoints in "threads." A thread is just a named timeline of state. This
  line says "the thread name is the run id." It is set the same way in Journey 3
  (`thread_id == manifest.run_id`), and that identical name is the only reason
  Journey 3 can reopen the database and land on exactly the checkpoint Journey 1
  left behind. If the two used different thread ids, resume would find nothing.

**Why do all this first?** Because a pause can happen deep inside step 3, in the
middle of rewriting bullet points. There is no opportunity at that moment to
"start saving." The recorder has to already be running, capturing every step's
output as it completes, so that when `interrupt()` fires the durable state is
simply *already there* — the pause just stops adding to it.

### Step 2 — the graph runs its normal stages
`src/orchestration/graph.py` defines the topology. The pause node sits at a
fan-in point:

```
run_gap_analysis ──┬─> write_professional_summary ──┐
                   ├─> optimize_experience ─────────┼─> await_candidate_clarifications
                   └─> optimize_skills ─────────────┘
```

All three stage-3 nodes must finish before `await_candidate_clarifications`
runs. `optimize_experience` is the one that produces questions.

#### Step 2, explained in plain English

`graph.py` is just a **wiring diagram**. It contains no resume logic at all — it
only says which step feeds which, and which steps are allowed to run at the same
time. LangGraph reads that diagram and does the scheduling.

The full pipeline has seven stages; the clarification pause sits at the boundary
between stage 3 and stage 4:

- **Stage 1 (parallel):** `extract_resume` and `analyze_job` run together —
  neither needs the other's output.
- **Stage 2 (fan-in):** `run_gap_analysis` waits for *both* stage-1 steps, then
  compares resume against job.
- **Stage 3 (parallel):** gap analysis fans back out to three independent steps
  — `write_professional_summary`, `optimize_experience`, `optimize_skills`.
- **Stage 4 (fan-in):** `await_candidate_clarifications` waits for *all three*
  stage-3 steps before it runs.

Two words to understand:

- **fan-out** — one step has several outgoing arrows, so LangGraph starts all of
  those next steps at once.
- **fan-in** — one step has several incoming arrows, so LangGraph will not start
  it until *every* arrow's source step has finished.

`await_candidate_clarifications` is a fan-in node with three incoming arrows.
That placement is deliberate: it guarantees the summary and the skills sections
are already done before the pipeline ever pauses. If the run then pauses for
days, no half-finished work from those other steps is lost — it's all in the
checkpoint, complete.

Only **`optimize_experience`** can create clarification questions. The other two
stage-3 steps just do their work and finish. So when `await_candidate_clarifications`
runs, it's really only inspecting what `optimize_experience` left in the state:
a list of questions (possibly empty) under `experience_clarifications`.

One more edge that matters later: `await_candidate_clarifications` has a
**conditional** exit. On a fresh run it either continues to `assemble_ats_resume`
(no questions) or pauses (questions, no answers). On a *resumed* run it routes
**backward** to `optimize_experience` to rewrite once more with the answers.
That back-edge is why the same node function has to handle three different
situations — covered in Step 7.

### Step 3 — inside `optimize_experience`, the HITL work happens per role
`src/orchestration/nodes/experience.py`

```
optimize_experience(state)
  │  clarification_answers = state["clarification_answers"]  # [] on a fresh run
  │
  └─ _optimize_experience_entries(resume, jd, strategy, [], run_id)
       │
       └─ _run_experience_optimization_workers(...)          # ThreadPoolExecutor, ≤4 roles at once
            │   for each Experience in resume.work_experience:
            │
            └─ _run_single_experience_optimization(resume, jd, strategy, experience, role_answers=[], run_id)
                 │
                 ├─ (a) build the rewrite context, call the writer agent
                 │      rewrite_proposal = _request_role_rewrite_proposal(context)
                 │
                 ├─ (b) truth floor + one repair pass
                 │      rewrite_decision = _decide_role_rewrite_outcome(...)
                 │
                 └─ (c) THE TRIGGER  ───────────────────────────────────────────
                        clarifications = build_bullet_clarifications(         clarifications.py
                            experience        = evidence_experience,
                            shipped_bullets   = rewrite_decision.finalized_section...achievements,
                            rewritten_bullets = rewrite_decision.selected_rewrite_proposal.rewritten_bullets,
                            run_id            = run_id,
                        )
                 returns (finalized_section, clarifications)   # one role's output + its questions
```

#### Step 3, explained in plain English

This is where the actual resume rewriting happens, and where the questions are
born. Read it as four nested layers, outermost first.

**Layer 1 — `optimize_experience(state)`** — the graph node itself. It pulls
`clarification_answers` out of the shared state. On a fresh run that list is
empty (`[]`); on a resumed run it holds the candidate's answers. Everything
below branches on that one fact. Fresh run: it passes `[]` down and proceeds to
rewrite from scratch.

**Layer 2 — `_optimize_experience_entries(...)`** — unpacks what the rewrite
needs from state: the parsed `resume`, the `job_description`, the
`alignment_strategy` from gap analysis, the (empty) answers list, and the
`run_id`. It doesn't do work itself; it hands off to the worker pool.

**Layer 3 — `_run_experience_optimization_workers(...)`** — the resume has
several jobs in its work history. Each job is rewritten **independently**, so
this runs them on a `ThreadPoolExecutor`, up to 4 at a time. One job's rewrite
never depends on another's. The output is a list of
`(optimized_section, [questions])` pairs — one pair per job.

**Layer 4 — `_run_single_experience_optimization(...)`** — the real work, for
**one** job. Three sub-steps:

- **(a) Ask the writer agent for a rewrite.**
  `format_experience_optimizer_context(...)` assembles everything the LLM needs
  (this role's bullets, the job description, the strategy, and — on a resumed
  run — the candidate's answers as structured evidence). Then
  `_request_role_rewrite_proposal(context)` makes the LLM call and gets back a
  `rewrite_proposal`: a proposed new version of every bullet.

- **(b) Gate the rewrite — `_decide_role_rewrite_outcome(...)`.** The proposal
  is not trusted blindly. Two checks, with different force:
  - **Truth floor (non-negotiable):** does the rewrite claim anything the
    source evidence doesn't support? If yes, it gets **one** repair attempt. If
    the repair still fails, the *original untouched bullets* ship instead — the
    pipeline would rather be boring than lie.
  - **Substance (best-effort):** are some bullets thin/weak? They earn one
    repair pass too. Whatever is *still* thin afterward is not invented away —
    it's flagged to become a candidate question.
  The result is a `rewrite_decision` holding the `finalized_section` (what will
  actually ship for this role) and the winning proposal.

- **(c) THE TRIGGER — `build_bullet_clarifications(...)`.** Now compare what
  shipped against what the writer wanted to say. If a strong bullet needs a
  concrete fact that only the candidate has (a number, a team size, a
  timeframe), this produces a question for it. Note it's passed the
  `evidence_experience` — the role *including any answers already given* — so on
  a resumed run it won't re-ask something the candidate already answered.
  (Its own internals are Step 4.)

`_run_single_experience_optimization` returns `(finalized_section, clarifications)`
for its one job. The worker pool collects all of them; the next step (Step 5)
merges the sections and caps the total number of questions.

**The key idea:** questions are a *last resort*. The pipeline tries to write a
good, truthful bullet; only when it can't — without either lying or going vague
— does it stop and ask the human.

### Step 4 — `build_bullet_clarifications`: the four internal steps
`src/hitl/professional_experience/clarifications.py`

```
build_bullet_clarifications(experience, shipped_bullets, rewritten_bullets, run_id)
  │
  ├─ STEP 1  if not shipped_bullets or not rewritten_bullets: return []
  │
  ├─ STEP 2  fact_gap_review = audit_experience_candidate_fact_gaps(...)   # ONE LLM call
  │            │  (wrapped in try/except -> on failure, return [] and ship anyway)
  │            │
  │            ├─ _build_fact_gap_review_input(experience, shipped, rewritten)
  │            │     └─ _render_bullet_block(...) per bullet
  │            │        └─ _shipped_bullet_text(...)          # rewrite text, or source if fallback
  │            │     returns one plain-text string
  │            │
  │            └─ request_structured_output(                  src/tools/llm_gateway/
  │                   ExperienceBulletFactGapReview,          #   output_model : schema
  │                   EXPERIENCE_CANDIDATE_FACT_GAP_RUBRIC,   #   system_prompt: rubric .md file
  │                   review_input,                           #   user_content : the bullets text
  │                   temperature=0.0)                        #   determinism
  │               returns ExperienceBulletFactGapReview
  │                 .findings: [ExperienceBulletFactGapFinding, ...]   one per bullet
  │                    .gap == None       -> ship as written
  │                    .gap == CandidateFactGap  -> needs a question
  │
  ├─ STEP 3  clarifications = clarifications_from_findings(experience, shipped, rewritten, findings)
  │            │
  │            ├─ index rewritten_bullets by bullet_id
  │            └─ for finding in findings:
  │                 if finding.gap is None:            continue      # 3b-i
  │                 if bullet_id not in the index:     log + continue  # 3b-ii  (hallucinated id)
  │                 append ExperienceBulletClarification(              # 3b-iii
  │                     **finding.gap.model_dump(),   # 4 fields, NOT hand-copied (inheritance)
  │                     bullet_id, company_name, job_title, start_date,
  │                     bullet = _shipped_bullet_text(...))
  │            returns list[ExperienceBulletClarification]   (answer="" on each)
  │
  └─ STEP 4  logger.info("experience_clarifications_built", findings=N, questions=M)
            return clarifications
```

#### Step 4, explained in plain English

`build_bullet_clarifications` is **the trigger** — the one function in the whole
pipeline that decides whether a human gets asked anything. It's small on
purpose, and it runs once per job. Four steps.

**STEP 1 — the cheap guard.** If this role has no bullets, or the rewrite
produced nothing (e.g. the truth floor rejected everything and the *original*
bullets shipped untouched), there's nothing to review. Return `[]` immediately —
don't spend an LLM call.

**STEP 2 — the one LLM call that makes the decision:
`audit_experience_candidate_fact_gaps(...)`.**

- First, **`_build_fact_gap_review_input(...)`** turns the role into one plain
  block of text the model can read: a header (job title, company, role
  description, skills used) followed by one numbered block per bullet. Each
  bullet block carries its stable **`bullet_id`**, the original text, the text
  that actually shipped, the ownership level the writer declared, the evidence
  the writer used, and any question the writer itself already flagged. Helpers
  `_render_bullet_block` and `_shipped_bullet_text` build those blocks —
  `_shipped_bullet_text` exists to handle the fallback case: if no rewrite
  survived, show the candidate the *source* bullet, not a rewrite that never
  shipped.
- Then **`request_structured_output(...)`** makes the call through the LLM
  gateway with four inputs: the **schema** the reply must match
  (`ExperienceBulletFactGapReview`), the **rubric** as system prompt (a
  Markdown file, `EXPERIENCE_CANDIDATE_FACT_GAP_RUBRIC`, loaded once at import —
  so "when is a bullet too thin?" is reviewable English under version control,
  not a buried Python string), the **bullet text** from the previous step, and
  **`temperature=0.0`** so the same bullet can't flip between "fine" and "needs
  a question" on identical runs. It returns a *validated object*, never raw text
  (one retry on malformed JSON, then it raises).
- The result has **`.findings`** — one `ExperienceBulletFactGapFinding` per
  bullet. Each finding's `.gap` is either `None` ("ship this bullet as written")
  or a `CandidateFactGap` ("this needs a candidate fact"). There is no separate
  boolean — the *presence* of a gap is the signal.
- The whole call is wrapped in `try/except`. If it fails (timeout, bad JSON
  twice, budget), the code logs it and returns `[]` — ship the good rewrite,
  ask nothing. This review is an *enhancement* on top of an already-truthful
  rewrite, so its failure must never throw away a finished role.

**STEP 3 — turn verdicts into routable questions:
`clarifications_from_findings(...)`.** Pure code, no LLM.

- Index the rewrite records by `bullet_id` for fast lookup.
- Walk every finding:
  - `gap is None` → skip (bullet is fine).
  - `bullet_id` not in the index → the model echoed back an id we never gave it
    (a hallucination). Log a warning and drop it — don't crash.
  - Otherwise → build an **`ExperienceBulletClarification`**. The four *content*
    fields (category, missing-fact summary, why it was flagged, the question
    text) are spread straight from `finding.gap` via `**model_dump()` — not
    retyped by hand, because `ExperienceBulletClarification` inherits them from
    `CandidateFactGap`. The rest is **routing identity**: `bullet_id`,
    `company_name`, `job_title`, `start_date`, and the bullet text the candidate
    will see. That identity is what lets an answer find its way back to the
    exact bullet after a pause of several days.
- Every clarification comes out with `answer=""` — an empty slot waiting for the
  candidate.

**STEP 4 — one telemetry line.** Log `experience_clarifications_built` with
`findings=N` (bullets the model looked at) and `questions=M` (bullets it decided
to ask about). The gap between N and M is how selective the trigger was on this
run. Then return the list.

**What leaves this function:** a `list[ExperienceBulletClarification]` for one
role — often empty. It travels up to Step 5, where every role's list is
flattened and capped.

### Step 5 — roles merge, then the volume cap
Back in `_optimize_experience_entries`:

```python
role_outcomes = _run_experience_optimization_workers(...)      # list of (section, [clarifications])
sections       = [section for section, _ in role_outcomes]
clarifications = [c for _, role_cs in role_outcomes for c in role_cs]   # flatten across all roles
return (
    _merge_optimized_experience_sections(sections),
    _cap_clarifications(clarifications, run_id),                # trim to max_clarifications_per_run (12)
)
```

`_cap_clarifications` keeps the first N (roles are in resume order, so recent
jobs survive whole) and logs `experience_clarifications_capped` if it trimmed.

#### Step 5, explained in plain English

Steps 3 and 4 ran *per job*, in parallel worker threads. This step is where
those independent results come back together into one answer for the whole
resume.

**Collect the worker results.** `_run_experience_optimization_workers(...)`
returns `role_outcomes` — a list with one entry per job, and each entry is a
pair: `(optimized_section_for_that_job, [questions_for_that_job])`.

**Split the pair into two piles:**

- `sections` — just the rewritten-section halves, one per job.
- `clarifications` — every job's question list, **flattened** into a single
  list. Three jobs with 2, 0, and 4 questions become one list of 6.

**Merge the sections.** `_merge_optimized_experience_sections(sections)` stitches
the per-job rewrites back into one `OptimizedExperienceSection` — concatenating
the optimized bullets, unioning the integrated keywords, joining the notes —
so downstream ATS assembly sees one experience section, not a scattered set.

**Cap the questions — `_cap_clarifications(clarifications, run_id)`.** There is a
configured ceiling, `workflow.max_clarifications_per_run` (default 12). The
reason is human, not technical: a candidate handed 40 questions answers none of
them carefully, so past a point the review's own thoroughness defeats itself.

- If the total is at or under the limit, the list passes through untouched.
- If it's over, keep the **first N** and drop the rest — then log
  `experience_clarifications_capped` with `requested` vs `kept` so nothing
  vanishes silently.
- *Why "first N" is the right truncation:* jobs are processed in resume order,
  which is newest-first. Taking the first N keeps the most recent roles'
  question sets **whole**, rather than leaving every role with a random handful
  of its questions answered and the rest missing.

**What leaves this step:** a tuple — `(merged_section, capped_clarifications)` —
handed back up to `optimize_experience`, which is Step 6.

### Step 6 — `optimize_experience` returns its partial state
```python
return {
    "optimized_experience":      merged_section,
    "experience_clarifications":  capped_clarifications,   # <- the pause node reads this
    "clarification_answers":      [],
    "resume":                    source_resume_for_downstream_review,
}
```

LangGraph merges that dict into the shared state.

#### Step 6, explained in plain English

A LangGraph node never mutates the shared state directly. It **returns a dict of
just the keys it wants to change**, and LangGraph merges that dict in. So this
`return` *is* the entire output of the experience stage. Four keys:

- **`optimized_experience: merged_section`** — the rewritten, merged experience
  section from Step 5. This is what ATS assembly will actually use to build the
  final resume.

- **`experience_clarifications: capped_clarifications`** — the (possibly empty,
  possibly trimmed-to-12) list of questions. **This is the key the pause node
  reads.** Everything the pause decision does in Step 7 hinges on whether this
  list is empty.

- **`clarification_answers: []`** — set back to empty, deliberately. On a fresh
  run it was already empty. On a *resumed* run it held the candidate's answers
  when this node started — and clearing it here is what lets the pipeline pause
  a *second* time if the answered rewrite surfaced brand-new gaps. (If this
  stayed populated, Step 7's `if clarification_answers:` check would wrongly
  treat the second pass as "already answered" and skip the pause. This is the
  double-execution trap — Step 8d and README §12.)

- **`resume: source_resume_for_downstream_review`** — on a fresh run this is
  just the original parsed resume, unchanged. On a resumed run,
  `_resume_with_candidate_answers_as_source(...)` has folded the candidate's
  answered facts *into* the role descriptions, so every downstream truthfulness
  check treats those facts as original source evidence — not as something the
  writer invented. Answer-sourced specifics are never flagged as fabrication.

Note what is **not** in the dict: `professional_summary`, `optimized_skills`,
`job_description`, etc. Those were written by other nodes and this node leaves
them untouched — merging a partial dict, not replacing the whole state, is why
the three parallel stage-3 nodes don't clobber each other.

### Step 7 — `await_candidate_clarifications`: the pause decision
`src/orchestration/nodes/experience.py`

```python
def await_candidate_clarifications(state):
    clarifications        = state.get("experience_clarifications") or []
    clarification_answers = state.get("clarification_answers") or []

    if not clarifications:            return {}      # CASE 1: nothing to ask -> carry on
    if clarification_answers:                        # CASE 2: resumed run -> Journey 3, step 7
        logger.info("candidate_clarifications_received", ...)
        return {}
    # CASE 3: questions, no answers -> PAUSE
    logger.info("candidate_clarifications_requested", ...)
    interrupt({
        "type": "candidate_clarifications_required",
        "questions": [c.model_dump(mode="json") for c in clarifications],
    })
    return {}
```

On a fresh run with questions, **CASE 3** fires. `interrupt()` raises
`GraphInterrupt`.

#### Step 7, explained in plain English

This tiny node is **the pause switch**. It does no rewriting — it only reads two
values from state and decides one of three things. It's deliberately kept short
and its checks must stay in this exact order.

It reads:

- `clarifications` — the question list `optimize_experience` just wrote
  (Step 6).
- `clarification_answers` — the candidate's answers. Empty on a fresh run;
  populated only on a resumed run (Journey 3 injects them with
  `Command(update=...)` *before* this node re-runs).

Three cases, checked top to bottom:

- **CASE 1 — `not clarifications`:** the experience stage found nothing that
  needs a human. Return `{}` and the graph flows straight on to ATS assembly.
  Most runs end here — no pause ever happens.

- **CASE 2 — `clarification_answers` is non-empty:** we're on a resumed run and
  the candidate has already answered. Log `candidate_clarifications_received`,
  return `{}`, carry on. **This check is the whole reason resume is safe.**
  LangGraph re-runs this node *from its first line* on resume — not from the
  `interrupt()` call — so without this branch it would pause a second time and
  the candidate would see the same questions forever. Because Journey 3
  populated `clarification_answers` first, CASE 2 catches the re-run and returns
  before `interrupt()` is ever reached again. (README §12, "the double-execution
  trap." Do not reorder these two checks.)

- **CASE 3 — questions exist, no answers yet:** a fresh run that needs the
  human. Log `candidate_clarifications_requested`, then call **`interrupt(...)`**
  with a payload containing the type tag and every question serialized to JSON.

`interrupt()` is LangGraph's built-in pause primitive. Calling it raises a
`GraphInterrupt` exception, which propagates out of the node. The `return {}`
line right after it **never executes on a fresh run** — it's there only for the
resumed pass, where LangGraph replays the node and `interrupt()` returns a value
instead of raising.

What happens to that raised exception is Step 8.

### Step 8 — LangGraph catches the raise, `_invoke_pipeline` returns
LangGraph writes the checkpoint (as part of executing `interrupt()`) and returns
`{"__interrupt__": (...), ...}` to `_invoke_pipeline`, which returns it to
`tailor_resume`, which passes it to `_finalize_pipeline_output`.

### Step 9 — `_finalize_pipeline_output` writes the paused-run directory
`src/orchestration/runner.py`

```python
if _pipeline_interrupted(output):                     # "__interrupt__" in output -> True
    snapshot       = pipeline.get_state(config)        # the frozen state
    snapshot_state = snapshot.values
    layout         = PausedRunLayout.at(_paused_run_directory(snapshot_state, run_id))

    paused_at = datetime.now(UTC)
    manifest  = ExperienceClarificationPausedRunManifest(
        run_id      = run_id,
        resume_path = resume_path,
        jd_path     = jd_path,
        paused_at   = paused_at,
        expires_at  = paused_at + timedelta(hours=<clarification_ttl_hours, default 168>),
    )
    save_paused_run_state(layout, manifest,
                          snapshot_state["experience_clarifications"])   # persistence.py
    result = _build_paused_orchestration_result(snapshot_state, paused_run_path=str(layout.root))
    _persist_result(result)                            # run_<timestamp>.json
    return result
```

### Step 10 — `save_paused_run_state` lays down two of the four files
`src/hitl/professional_experience/persistence.py`

```
save_paused_run_state(layout, manifest, clarifications)
  ├─ layout.root.mkdir(...)
  ├─ write_clarification_sheet(layout, clarifications)      -> clarifications_sheet.json
  │     sheet = {"_instructions": ..., "clarifications": [ c.model_dump() ... ]}
  └─ layout.manifest.write_text(manifest.model_dump_json()) -> paused_run_manifest.json
```

### Step 11 — `finally:` block archives the checkpoint db
```
close_checkpoint_database(checkpointer)                 # release the SQLite handle
_settle_fresh_run_checkpoint(checkpoint_db_path, result)
  └─ result.paused_run_path is set  -> archive_checkpoint_database(db_path, layout)
       shutil.move(<run>.sqlite3  ->  paused_run_<id>/checkpoints.sqlite3)
```

**End state on disk:**

```
paused_run_<run_id>/
  ├─ clarifications_sheet.json     (questions, every answer "")
  ├─ paused_run_manifest.json      (run_id, paths, paused_at, expires_at)
  └─ checkpoints.sqlite3           (the entire frozen pipeline state)
```

The process exits. `answers_audit.jsonl` does not exist yet — it is created on
the first answer.

---
---

# JOURNEY 2 — The candidate answers (web path)

> The CLI path is simpler: the candidate edits `clarifications_sheet.json` by
> hand and runs `--resume-from <dir>`, which jumps straight to Journey 3.

## 2.1 The call stack

```
POST /api/runs/{run_id}/resume   {"answers": [{"bullet_id": "...", "answer": "..."}]}
  │
  └─ resume_run(request)                                    web_app/server.py
       │
       ├─ _get_run(run_id)  -> the in-memory run record; must have "paused_path"
       ├─ payload = await request.json()
       ├─ _save_clarification_answers(paused_path, payload["answers"])   web_app/server.py
       │     │
       │     ├─ submitted = { str(a["bullet_id"]): str(a["answer"]) for a in answers if a.get("bullet_id") }
       │     │       (keyed by bullet_id — NEVER by list position)
       │     │
       │     └─ record_clarification_answers(                persistence.py
       │            PausedRunLayout.at(paused_path),
       │            submitted,
       │            answered_by="web-ui (unauthenticated)")
       │          │
       │          ├─ STEP 1  read_clarification_sheet(layout)          -> current questions
       │          │            reject any bullet_id not in the sheet   (ValueError)
       │          ├─ STEP 2  for each question: if a fresh non-blank answer,
       │          │            model_copy(answer=, answered_at=, answered_by=)
       │          │            and queue a ClarificationAnswerRecord
       │          ├─ STEP 3  if nothing got answered -> ValueError
       │          ├─ STEP 4  append_answer_records(layout, new_records)  -> answers_audit.jsonl  (FIRST)
       │          │          write_clarification_sheet(layout, updated)  -> clarifications_sheet.json
       │          └─ return updated list
       │
       ├─ run_id = _register_run()                          # a NEW run id for the resumed execution
       └─ return JSONResponse({"run_id": ...}, status_code=202,
                              background=BackgroundTask(_execute_resumed_run, run_id, paused_path))
              │
              └─ _execute_resumed_run(new_run_id, paused_path)   web_app/server.py
                    └─ resume_paused_run(paused_path, progress_callback=...)   <== JOURNEY 3
```

### 2.1a The call stack, explained in plain English

Journey 2 is a **web request handler**, and its whole job is to be *fast*. It
records the candidate's answers to disk and immediately hands the slow work
(actually resuming the pipeline) to a background task. It does not tailor
anything itself.

**The request.** The browser POSTs to `/api/runs/{run_id}/resume` with a JSON
body: a list of `{bullet_id, answer}` objects. Note it's keyed by `bullet_id`,
never by position — the browser may show the questions in a different order than
the file stores them.

**`resume_run(request)` — the handler:**

1. **`_get_run(run_id)`** — look up the in-memory record for that run. The
   server keeps a small dict of recent runs; the paused one has a
   `paused_path` pointing at its `paused_run_<id>/` folder. If the record is
   missing or has no `paused_path`, return **409** ("this run is not waiting for
   answers") — nothing to resume.

2. **`payload = await request.json()`** — parse the body. Malformed JSON →
   **422**.

3. **`_save_clarification_answers(paused_path, answers)`** — the one piece of
   real work. Two parts:

   - **Reshape the payload.** Turn the list of `{bullet_id, answer}` into a
     `{bullet_id -> answer_text}` map, dropping any entry with no `bullet_id`.
     If that leaves nothing, raise `ValueError` → **422** ("answer at least
     one"). The web layer does **no file I/O** — it deliberately doesn't know
     the sheet's on-disk shape. (It used to parse the sheet itself, its idea of
     the format drifted from the writer's, and every browser submission broke
     while the CLI path kept working. Now it just hands the map to the module
     that owns the format.)

   - **`record_clarification_answers(layout, submitted, answered_by=...)`** in
     `persistence.py` does the durable write, in four steps:
     - **STEP 1** — read the current sheet. Reject any submitted `bullet_id`
       that isn't a real question in it (`ValueError`).
     - **STEP 2** — for each question that got a fresh, non-blank answer, make an
       updated copy (`answer=`, `answered_at=`, `answered_by=`) and queue a
       `ClarificationAnswerRecord` for the audit log.
     - **STEP 3** — if *nothing* was actually answered, `ValueError`.
     - **STEP 4** — write the audit log **first**
       (`append_answer_records` → `answers_audit.jsonl`), *then* the updated
       working sheet (`write_clarification_sheet` → `clarifications_sheet.json`).
       Order matters: if the process dies between the two writes, the
       append-only record of what the candidate submitted still survives.
     - `answered_by` is `"web-ui (unauthenticated)"` — there's no auth on this
       endpoint yet; anyone with the run id can answer, and the audit is honest
       about that.

4. **`run_id = _register_run()`** — mint a **new** run id for the resumed
   execution. Journey 3 is tracked as its own run, separate from the original.

5. **Return `202 Accepted`** with that new run id, and attach a Starlette
   `BackgroundTask`. The HTTP response goes back to the browser *now*; the
   background task runs *after* the response is sent.

**`_execute_resumed_run(new_run_id, paused_path)`** — the background task. It
just calls **`resume_paused_run(paused_path, ...)`** — that's the entry into
Journey 3. It runs in the same process as the web worker, but the browser is no
longer waiting on it; it polls the run's event stream for progress instead.

**Why the split?** Reloading state and re-running LLM calls takes minutes — far
too long to hold an HTTP connection open. So Journey 2 does only the fast,
must-be-durable part (save the answers), and Journey 3 does the slow part. A CLI
tool wouldn't split them: `--resume-from <dir>` reads the edited file and
resumes in one process, because there's no HTTP response to keep fast.

## 2.2 Step by step

### Step 1 — the endpoint validates the run is actually paused
`resume_run` looks up the source run. If it has no `paused_path`, it returns
**409** ("not waiting for answers"). Malformed JSON or a validation error from
`_save_clarification_answers` returns **422**.

> Historical note: an unhandled `KeyError` used to escape here as a **500**,
> because `_save_clarification_answers` parsed the sheet itself and got its shape
> wrong. It no longer touches the file — see README §11.

### Step 2 — `_save_clarification_answers` reshapes the payload
```python
submitted = {
    str(item["bullet_id"]): str(item.get("answer", ""))
    for item in answers
    if item.get("bullet_id")
}
```
Browser sends a list of `{bullet_id, answer}`; this becomes a `{bullet_id -> text}`
map. The web layer does no file I/O — it hands `submitted` to `persistence.py`.

### Step 3 — `record_clarification_answers` does the durable write
`src/hitl/professional_experience/persistence.py` — the 4 internal steps are in
the call stack above. The ordering that matters:

```
append_answer_records(...)      # <- audit log written FIRST
write_clarification_sheet(...)  # <- then the mutable working copy
```

If the process died between these two lines, the audit of what the candidate
submitted still survives.

### Step 4 — the endpoint returns immediately, resume runs in the background
`202 Accepted` goes back to the browser with a **new** `run_id`. A Starlette
`BackgroundTask` then calls `_execute_resumed_run`, which calls
`resume_paused_run` — Journey 3, in the same process as the HTTP worker but after
the response has been sent.

**End state on disk:**

```
paused_run_<run_id>/
  ├─ clarifications_sheet.json     (some answers now filled + answered_at/by)
  ├─ paused_run_manifest.json      (unchanged)
  ├─ answers_audit.jsonl           (NEW — one line per answer accepted)
  └─ checkpoints.sqlite3           (unchanged — still the frozen state)
```

---
---

# JOURNEY 3 — Resume to completion

## 3.1 The call stack

```
resume_paused_run(paused_run_path, progress_callback)        runner.py
  │
  ├─ 1. layout, manifest = load_paused_run(paused_run_path)   persistence.py
  │        ├─ CHECK 1: paused_run_manifest.json exists?  else FileNotFoundError
  │        └─ CHECK 2: checkpoints.sqlite3 exists?       else FileNotFoundError
  │
  ├─ 2. if manifest.is_expired:  raise ValueError("expired on ...")   # models.py property
  │
  ├─ 3. answered_clarifications = read_answered_clarifications(layout)   persistence.py
  │        └─ read_clarification_sheet(layout) filtered by .is_answered
  │        if empty:  raise ValueError("no answered questions yet")
  │
  ├─ 4. checkpointer = open_checkpoint_database(layout.checkpoint_db)   checkpointing.py
  ├─ 5. pipeline = _build_pipeline(checkpointer)
  ├─ 6. config = {"configurable": {"thread_id": manifest.run_id}}      # SAME thread_id as Journey 1
  │
  ├─ 7. command = Command(
  │        resume = {"status": "candidate_answers_submitted"},         # becomes interrupt()'s return
  │        update = {"clarification_answers": answered_clarifications}) # merged into state
  │
  ├─ 8. output = _invoke_pipeline(pipeline, command, config, progress_callback)
  │        │
  │        └─ LangGraph reloads the checkpoint for thread_id and RESUMES
  │             │
  │             └─ await_candidate_clarifications(state)   RE-RUNS FROM THE TOP
  │                  clarifications        = state["experience_clarifications"]   # still set
  │                  clarification_answers = state["clarification_answers"]        # NOW POPULATED (step 7)
  │                  if clarification_answers:                                     # CASE 2 this time
  │                       logger.info("candidate_clarifications_received")
  │                       return {}                        # interrupt() NEVER reached again
  │             │
  │             └─ _route_after_candidate_clarifications(state)          graph.py
  │                  state["clarification_answers"] is truthy -> "rewrite_experience"
  │                  (routes BACK to optimize_experience, not forward)
  │             │
  │             └─ optimize_experience(state)   RUNS A SECOND TIME
  │                  clarification_answers = state["clarification_answers"]   # the answers
  │                  │
  │                  ├─ _resume_with_candidate_answers_as_source(resume, answers)   experience.py
  │                  │     for each role:
  │                  │       answers_for_role(experience, answers)                   answers.py
  │                  │       experience_with_candidate_answers(experience, ...)      answers.py
  │                  │
  │                  └─ _run_single_experience_optimization(..., role_answers=<this role's answers>)
  │                       logger.info("experience_clarification_answers_applied")
  │                       evidence_experience = experience_with_candidate_answers(experience, role_answers)
  │                       ... rewrite with the new facts as evidence ...
  │                       build_bullet_clarifications(evidence_experience, ...)   # may find NEW gaps
  │             │
  │             └─ await_candidate_clarifications  runs AGAIN
  │                  new clarifications? and clarification_answers was cleared? -> PAUSE AGAIN
  │                  otherwise -> return {} and route "assemble_resume"
  │             │
  │             └─ assemble_ats_resume -> evaluate_resume_quality -> ... -> render_final_resume
  │
  ├─ 9. result = _finalize_pipeline_output(output, ..., paused_run_layout=layout)
  │        _pipeline_interrupted(output) is False -> _build_completed_orchestration_result
  │
  └─ 10. finally:
          close_checkpoint_database(checkpointer)
          _settle_resumed_run_checkpoint(layout, result)
            └─ result.paused_run_path is None (completed) -> layout.checkpoint_db.unlink()
```

### 3.1a The call stack, explained in plain English

Journey 3 is a **cold start** — a brand-new process (well, a background task)
with no memory of Journey 1. Its job: open the folder Journey 1 left, load the
frozen pipeline back into memory, feed in the answers, and let the graph run to
the end.

**Steps 1–3 — refuse cheaply, before touching any machinery.** Every check that
can *reject* a resume happens first, so a bad resume fails in milliseconds
instead of after a database connection and a graph compile:

1. **`load_paused_run(paused_run_path)`** — confirm the folder is intact: both
   `paused_run_manifest.json` and `checkpoints.sqlite3` must exist, else
   `FileNotFoundError`. Returns a `layout` (paths to the four files) and the
   parsed `manifest`.
2. **`manifest.is_expired`** — a property on the model: `datetime.now(UTC) >=
   expires_at`, computed live. Journey 1 set `expires_at` to 7 days out. Past
   that, `ValueError` — the candidate must start a fresh run. (The pipeline
   state may reference uploaded files or PII mappings that were cleaned up.)
3. **`read_answered_clarifications(layout)`** — read the sheet, keep only
   questions whose `answer` is non-blank. If that's empty, `ValueError` ("answer
   at least one first"). Journey 2's validation already enforced this, but
   Journey 3 re-checks because the CLI path edits the sheet by hand.

**Steps 4–6 — reconnect to the *exact* paused thread.**

4. **`open_checkpoint_database(layout.checkpoint_db)`** — open the same SQLite
   file Journey 1 archived into this folder, with the same class allowlist.
5. **`_build_pipeline(checkpointer)`** — compile the identical graph again, bound
   to this checkpointer.
6. **`config = {"configurable": {"thread_id": manifest.run_id}}`** — the
   critical line. `thread_id` is exactly the `run_id` from Journey 1 (stored in
   the manifest). That identical name is *the entire mechanism* by which
   LangGraph finds the checkpoint history and knows where to resume. A different
   thread id here would find nothing and start from scratch.

**Step 7 — package the answers as a resume command.**

```
command = Command(
    resume = {"status": "candidate_answers_submitted"},          # the value interrupt() returns on replay
    update = {"clarification_answers": answered_clarifications},  # merged into state BEFORE any node re-runs
)
```

`Command(resume=...)` is what makes `interrupt()` *return* instead of raise on
the replayed node. `Command(update=...)` writes the answers into state first, so
by the time `await_candidate_clarifications` re-runs, `clarification_answers` is
already populated.

**Step 8 — LangGraph reloads the checkpoint and runs forward.** This is the
whole pipeline resuming:

- **`await_candidate_clarifications` re-runs from its first line** (LangGraph
  always replays the interrupted node from the top, not from the `interrupt()`
  call). This time `clarification_answers` is non-empty → **CASE 2** → log
  `candidate_clarifications_received`, `return {}`. `interrupt()` is never
  reached. This is the double-execution trap, avoided by the state check.
- **`_route_after_candidate_clarifications` routes *backward*.** Truthy
  `clarification_answers` → `"rewrite_experience"` → control goes back to
  `optimize_experience`, not forward to assembly.
- **`optimize_experience` runs a second time, now with the answers.**
  `_resume_with_candidate_answers_as_source` folds each answer into its role's
  description (so downstream truthfulness checks treat it as source evidence).
  Then per role, `_run_single_experience_optimization(..., role_answers=...)`
  rewrites the bullets again with the new facts as evidence, logs
  `experience_clarification_answers_applied`, and calls
  `build_bullet_clarifications` once more — which *may* surface brand-new gaps.
- **`await_candidate_clarifications` runs a third time.** If the second rewrite
  produced new questions (and `optimize_experience`'s return cleared
  `clarification_answers`), it **pauses again** — same folder, a second cycle.
  Usually it finds nothing new, returns `{}`, and routes `"assemble_resume"`.
- **The rest of the graph runs to the end:** `assemble_ats_resume` →
  `evaluate_resume_quality` → (optional patch) → `rehydrate_pii` →
  `render_final_resume`.

**Steps 9–10 — finalize and clean up.**

9. **`_finalize_pipeline_output(output, ...)`** — this time there's no
   `"__interrupt__"` key, so it builds a *completed* `OrchestrationResult` with
   the rendered artifacts and `.paused_run_path` left `None`.
10. **`finally:`** — always runs. Close the SQLite handle, then
    `_settle_resumed_run_checkpoint`: because the result is completed
    (`.paused_run_path is None`), **delete `checkpoints.sqlite3`** — the run is
    done, there is nothing left to resume. The `paused_run_<id>/` folder stays
    (sheet + audit + manifest) as a permanent record.

**The mental model:** Journey 3 is Journey 1 *continued* — same graph, same
thread id, same state — but entered from a folder on disk instead of from
uploaded files, and with the missing facts now filled in.

## 3.2 Step by step

### Steps 1–3 — refuse cheaply, before opening anything
`resume_paused_run` does every check that can *reject* the resume **before** it
opens a database or compiles a graph:

1. `load_paused_run` — manifest and checkpoint db both present, else `FileNotFoundError`.
2. `manifest.is_expired` — `datetime.now(UTC) >= expires_at`, computed live. If
   expired, `ValueError` and the candidate must start fresh.
3. `read_answered_clarifications` — if the sheet has zero non-blank answers,
   `ValueError` ("answer at least one first").

### Steps 4–7 — reconnect to the exact paused thread
```python
checkpointer = open_checkpoint_database(layout.checkpoint_db)   # the SAME file from Journey 1
config = {"configurable": {"thread_id": manifest.run_id}}       # the SAME thread_id
command = Command(
    resume = {"status": "candidate_answers_submitted"},
    update = {"clarification_answers": answered_clarifications},
)
```

`thread_id` is how LangGraph finds the checkpoint history. `Command(update=...)`
merges the answers into state **before** any node re-runs.

### Step 8a — `await_candidate_clarifications` re-runs, takes the other branch
LangGraph restarts the interrupted node from its top. This time
`state["clarification_answers"]` is populated (from step 7's `update`), so
**CASE 2** fires: log `candidate_clarifications_received`, `return {}`.
`interrupt()` is never reached. This is the double-execution trap, avoided by a
state check — README §12.

### Step 8b — the graph routes *backward* to re-rewrite
`_route_after_candidate_clarifications` (`graph.py`) sees truthy
`clarification_answers` and returns `"rewrite_experience"`, sending control back
to `optimize_experience` rather than forward to assembly.

### Step 8c — `optimize_experience` runs a second time, now with the answers
```
optimize_experience(state)
  clarification_answers = state["clarification_answers"]        # the candidate's facts
  │
  ├─ _resume_with_candidate_answers_as_source(resume, answers)
  │     rebuilds the resume so downstream truthfulness checks treat
  │     candidate facts as source evidence
  │
  └─ per role: _run_single_experience_optimization(..., role_answers)
       ├─ answers_for_role(experience, all_answers)    -> just this role's answers   answers.py
       ├─ experience_with_candidate_answers(...)        -> role.description augmented  answers.py
       ├─ logger.info("experience_clarification_answers_applied")
       ├─ rewrite the bullets again, now with the new evidence
       └─ build_bullet_clarifications(evidence_experience, ...)   # answered facts won't be re-asked
```

### Step 8d — pause again, or finish
`await_candidate_clarifications` runs once more. If the second rewrite surfaced
*new* gaps (and `clarification_answers` was cleared by `optimize_experience`'s
return), it pauses again — a second `paused_run` cycle, same directory. Usually
it finds nothing new, returns `{}`, and the graph continues:

```
assemble_ats_resume -> evaluate_resume_quality -> (patch?) -> rehydrate_pii -> render_final_resume
```

### Steps 9–10 — finalize and clean up
`_finalize_pipeline_output` sees no `"__interrupt__"` this time and builds a
*completed* result. The `finally:` block deletes `checkpoints.sqlite3` — the run
is done, there is nothing left to resume. The `paused_run_<id>/` directory
remains (sheet + audit + manifest) as a record.

---
---

# Quick reference — every HITL function

## `src/hitl/professional_experience/clarifications.py`

| Function | Called by | Calls | Returns |
|---|---|---|---|
| `build_bullet_clarifications` | `_run_single_experience_optimization` [experience.py] | `audit_experience_candidate_fact_gaps`, `clarifications_from_findings` | `list[ExperienceBulletClarification]` |
| `audit_experience_candidate_fact_gaps` | `build_bullet_clarifications` | `_build_fact_gap_review_input`, `request_structured_output` | `ExperienceBulletFactGapReview` |
| `clarifications_from_findings` | `build_bullet_clarifications`; tests | `_shipped_bullet_text` | `list[ExperienceBulletClarification]` |
| `_build_fact_gap_review_input` | `audit_experience_candidate_fact_gaps` | `_render_bullet_block` | `str` (the prompt) |
| `_render_bullet_block` | `_build_fact_gap_review_input` | `_shipped_bullet_text` | `str` (one bullet block) |
| `_shipped_bullet_text` | `clarifications_from_findings`, `_render_bullet_block` | — | `str` |

## `src/hitl/professional_experience/answers.py`

| Function | Called by | Calls | Returns |
|---|---|---|---|
| `answers_for_role` | `_resume_with_candidate_answers_as_source`, `_run_experience_optimization_workers` [experience.py] | `build_experience_bullet_id` [models.py] | filtered `list[ExperienceBulletClarification]` |
| `experience_with_candidate_answers` | `_resume_with_candidate_answers_as_source`, `_run_single_experience_optimization` [experience.py] | — | a new `Experience` |

## `src/hitl/professional_experience/persistence.py`

| Function | Called by | Calls | Returns |
|---|---|---|---|
| `PausedRunLayout.at` | `runner.py`, `web_app/server.py` | — | `PausedRunLayout` |
| `write_clarification_sheet` | `save_paused_run_state`, `record_clarification_answers` | — | sheet path `str` |
| `read_clarification_sheet` | `read_answered_clarifications`, `record_clarification_answers` | — | `list[ExperienceBulletClarification]` |
| `read_answered_clarifications` | `resume_paused_run` [runner.py] | `read_clarification_sheet` | answered-only `list` |
| `record_clarification_answers` | `_save_clarification_answers` [web_app/server.py] | `read_clarification_sheet`, `append_answer_records`, `write_clarification_sheet` | updated `list` |
| `append_answer_records` | `record_clarification_answers` | — | `None` (appends `answers_audit.jsonl`) |
| `save_paused_run_state` | `_finalize_pipeline_output` [runner.py] | `write_clarification_sheet` | dir path `str` |
| `load_paused_run` | `resume_paused_run` [runner.py] | — | `(PausedRunLayout, manifest)` |
| `archive_checkpoint_database` | `_settle_fresh_run_checkpoint` [runner.py] | — | `None` (moves the db file) |

## `src/hitl/professional_experience/models.py`

| Shape | Constructed by | Read by |
|---|---|---|
| `ExperienceBulletFactGapFinding` | `request_structured_output` (one per bullet) | `clarifications_from_findings` |
| `CandidateFactGap` | the LLM (nested in a finding) | inherited by `ExperienceBulletClarification` |
| `ExperienceBulletFactGapReview` | `request_structured_output` (the wrapper) | `build_bullet_clarifications` (`.findings`) |
| `ExperienceBulletClarification` | `clarifications_from_findings` | the sheet, `answers_for_role`, the pause node, pipeline state |
| `ClarificationAnswerRecord` | `record_clarification_answers` | `answers_audit.jsonl` only (human/compliance) |
| `ExperienceClarificationPausedRunManifest` | `_finalize_pipeline_output` [runner.py] | `load_paused_run`, `resume_paused_run` (`.is_expired`) |
| `build_experience_bullet_id` (fn) | `_collect_rewrite_truthfulness_findings` [experience.py], `answers_for_role` | — |

## Outside the package but part of the loop

| Function | File | Role |
|---|---|---|
| `tailor_resume` | `orchestration/runner.py` | Journey 1 entry |
| `resume_paused_run` | `orchestration/runner.py` | Journey 3 entry |
| `_finalize_pipeline_output` | `orchestration/runner.py` | detects interrupt, writes paused-run dir |
| `optimize_experience` | `orchestration/nodes/experience.py` | runs the rewrite; produces questions |
| `_run_single_experience_optimization` | `orchestration/nodes/experience.py` | per-role rewrite + `build_bullet_clarifications` |
| `_cap_clarifications` | `orchestration/nodes/experience.py` | the volume bound |
| `await_candidate_clarifications` | `orchestration/nodes/experience.py` | the pause node (`interrupt()`) |
| `_route_after_candidate_clarifications` | `orchestration/graph.py` | routes a resumed run back to rewrite |
| `open_checkpoint_database` / `close_checkpoint_database` | `orchestration/checkpointing.py` | the SqliteSaver lifecycle |
| `resume_run` / `_save_clarification_answers` / `_execute_resumed_run` | `web_app/server.py` | Journey 2, the web surface |
