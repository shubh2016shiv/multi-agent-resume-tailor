# Orchestration Graph
## The LangGraph Spine That Owns Every Routing Decision in Resume Tailor

> **Scope:** `src/orchestration/` — the `StateGraph` topology, its node modules,
> the CrewAI execution adapter, and the escalation/failure policies that decide
> what happens when a stage does not simply succeed.
> **Audience:** Contributors adding a pipeline stage; anyone debugging why a run
> paused, failed, or escalated to human review.

---

## Table of Contents

1. [What Problem the Graph Solves](#1-what-problem-the-graph-solves)
2. [The Core Design Principle — LangGraph Decides, CrewAI Executes](#2-the-core-design-principle--langgraph-decides-crewai-executes)
3. [Graph Topology — The Full Pipeline at a Glance](#3-graph-topology--the-full-pipeline-at-a-glance)
4. [Stage-by-Stage Walkthrough](#4-stage-by-stage-walkthrough)
5. [The Three Routing Decisions](#5-the-three-routing-decisions)
6. [run_agent_task — The One Seam Between the Graph and CrewAI](#6-run_agent_task--the-one-seam-between-the-graph-and-crewai)
7. [The "No Retry Loops" Philosophy](#7-the-no-retry-loops-philosophy)
8. [The Human Review Escalation Policy — One Documented Home](#8-the-human-review-escalation-policy--one-documented-home)
9. [Failure Taxonomy — Three Kinds of "The Run Stopped"](#9-failure-taxonomy--three-kinds-of-the-run-stopped)
10. [Pause and Resume — How Human-in-the-Loop Works at the Graph Level](#10-pause-and-resume--how-human-in-the-loop-works-at-the-graph-level)
11. [The Shared-Resource Lock — A Real Concurrency Bug and Its Fix](#11-the-shared-resource-lock--a-real-concurrency-bug-and-its-fix)
12. [Why This Shape Fits the Project](#12-why-this-shape-fits-the-project)
13. [Future Considerations](#13-future-considerations)

---

## 1. What Problem the Graph Solves

Resume tailoring is not a single generation step; it is a pipeline with real
dependencies. A job description cannot be aligned against a resume before both
are parsed. A skills section cannot be assembled before the gap analysis has
decided what to emphasize. A resume cannot render until it has passed a
truthfulness and ATS-compliance check. And, uniquely to this domain, the
pipeline sometimes cannot finish at all without asking the candidate a
question — a bullet may be truthful but too thin to ship without a fact only
the candidate has.

A linear script handles the happy path of that dependency chain easily enough.
It falls apart at exactly the points that matter most: what happens when a
draft fails a quality gate, when an essential section renders empty, when the
pipeline needs to pause mid-run and resume days later with a candidate's
answers in hand. Those are not edge cases bolted onto the side of this system —
they are load-bearing requirements, because the product's core promise
(truthful, ATS-safe, quality-gated resumes) only holds if failure and pause are
first-class citizens of the control flow, not exceptions to it.

`src/orchestration/graph.py` answers this with a LangGraph `StateGraph`: a
directed graph of named nodes and typed state, where every edge — including
every conditional branch — is declared once, in one file, and is the only
legal way control ever moves from one stage to the next.

---

## 2. The Core Design Principle — LangGraph Decides, CrewAI Executes

The clearest way to state the division of labor in this codebase: **CrewAI runs
inside nodes; LangGraph decides what runs next.** An agent's job ends the
moment it returns a validated Pydantic object. Whether that object is good
enough to hand to the next stage, whether the pipeline should retry, patch,
pause, or terminate — none of that is a question any agent ever answers. It is
answered by plain Python functions living beside the graph, and by the graph's
own conditional edges.

```text
   +-------------------------+       +--------------------------------+
   |      LangGraph graph     | ----> |   node function (plain Python)  |
   |  (owns state + routing)  |       |   - formats context              |
   +-------------------------+       |   - calls a CrewAI agent (drafts) |
              ^                       |   - runs code-owned validation   |
              |                       |   - writes typed state fields    |
              +-----------------------+--------------------------------+
                        returns a partial dict; graph merges it,
                        then evaluates its OWN routing functions
                        to decide the next node -- never the agent
```

This is the same "agents reason, code decides" principle documented in
[Agent Roles](agent-roles.md), applied one level up: agents decide *content*,
nodes decide *validity*, and the graph alone decides *sequence*.

---

## 3. Graph Topology — The Full Pipeline at a Glance

### 3.1 In Plain Terms, First

Before any diagram, here is the whole pipeline in one paragraph, no node names
required. The system reads the resume and the job posting at the same time and
waits until both are done. It then runs one comparison pass to figure out how
the two line up. From that comparison, it writes three parts of the new resume
at the same time — the summary, the experience section, and the skills section
— and waits until all three are finished. If, while rewriting the experience
section, the system realizes it needs a fact only the candidate knows (a
number, a result, a scale), it stops right there and waits — possibly for
days — until the candidate answers. Once experience is settled, the three
parts are assembled into one resume, which is then checked for quality and for
ATS (resume-scanner) safety. If that check finds a section came out empty, the
system repairs it automatically from data it already has, with no further
guessing. Personal information that was hidden from the AI is then restored,
and only at the very end does the system decide whether the result is good
enough to actually write out as files.

Everything below is that same paragraph, redrawn as the exact technical graph.

### 3.2 The Full Technical Diagram

`build_resume_enhancement_graph` in `graph.py` registers eleven nodes plus the
implicit `START`/`END`. The main spine — ingest, compare, draft, assemble,
grade, render — reads top to bottom. The one loop in the graph (the
clarification gate) is called out here but drawn in full in
[Section 3.3](#33-zoom-in-the-one-loop-in-the-graph) so it doesn't tangle the
main flow.

```text
                                     START
                                       |
                +----------------------+----------------------+
                |                                              |
                v                                              v
        +----------------+                             +--------------+
        | extract_resume |                             | analyze_job  |
        +----------------+                             +--------------+
                |                                              |
                +----------------------+----------------------+
                                       |
                                       |   FAN-IN: both must finish
                                       v
                            +--------------------+
                            |  run_gap_analysis   |
                            +--------------------+
                                       |
          +----------------------------+----------------------------+
          |                            |                             |
          v                            v                             v
+-----------------------+  +----------------------+     +--------------------+
| write_professional_    |  |  optimize_experience |     |  optimize_skills   |
| summary                |  |                      |     |                    |
+-----------------------+  +----------------------+     +--------------------+
          |                            |                             |
          +----------------------------+----------------------------+
                                       |
                                       |   FAN-IN: all three must finish
                                       v
                       +-----------------------------------+
                       |   await_candidate_clarifications   |
                       |     (see Section 3.3 for the       |
                       |      two paths out of this box)    |
                       +-----------------------------------+
                                       |
                                       |   both paths eventually reach:
                                       v
                            +------------------------+
                            |   assemble_ats_resume   |
                            +------------------------+
                                       |
                                       v
                          +-------------------------+
                          | evaluate_resume_quality  |
                          +-------------------------+
                                       |
                         ROUTING: rendered ATS status
                    +---------------------+---------------------+
                    | FAIL -> "patch"       | PASS/other -> "continue"
                    v                       v
          +----------------------+   +----------------+
          | patch_ats_assembly    |-->|  rehydrate_pii |
          +----------------------+   +----------------+
                                                |
                                    ROUTING: quality gate outcome
                              +---------------------+---------------------+
                              | gate passed -> "render" | gate failed -> "end"
                              v                          v
                    +------------------------+          END
                    |  render_final_resume    |
                    +------------------------+
                              |
                              v
                             END
```

### 3.3 Zoom-In: The One Loop in the Graph

`await_candidate_clarifications` is the only box in the whole graph that can
send execution backward. Pulled out on its own, the two paths look like this:

```text
                    +-----------------------------------+
              +---->|   await_candidate_clarifications   |
              |     +-----------------------------------+
              |                       |
              |        ROUTING: does state already carry
              |        answered clarification_answers?
              |                       |
              |     +-----------------+-----------------+
              |     | NO answers yet:  | YES, answers present:
              |     | PAUSE the graph   | apply them and move on
              |     | here (interrupt)  |
              |     | until a human      |
              |     | supplies answers   |
              |     | -- see Section 10  |
              |     v                    v
              |  (waits, then    +----------------------+
              |   re-enters      | optimize_experience   |
              |   this routing   | (re-run, this time    |
              |   step once      |  WITH the answers)    |
              |   answers        +----------------------+
              |   arrive)                   |
              |                             v
              +----------------- back to await_candidate_clarifications,
                                  which now finds no answers left to apply
                                  and falls through to assemble_ats_resume
```

The loop runs **at most once per unanswered question**: `optimize_experience`
always clears `clarification_answers` back to empty the moment it consumes
them, so the second pass through the routing step is guaranteed to find
nothing left to re-apply and fall through to assembly. This is not an
open-ended retry loop — it is one bounded round trip, gated entirely by
whether the state already contains answers or not.

### 3.4 Two Structural Properties Worth Naming

First, Stage 1 and Stage 3 are true fan-out/fan-in: `extract_resume` and
`analyze_job` run concurrently and both must complete before `run_gap_analysis`
starts; the same holds for the three Stage 3 content agents converging on
`await_candidate_clarifications`. Second, every path out of quality evaluation
— the clean-continue path and the ATS-patch path alike — reconverges on
`rehydrate_pii` before the graph ever considers rendering. There is exactly one
door into rendering, and PII restoration guards it on every route.

---

## 4. Stage-by-Stage Walkthrough

### Stage 1 — Parallel Ingestion (`nodes/ingestion.py`)

`extract_resume` and `analyze_job` run concurrently from `START`. `extract_resume`
hands the resume's file path straight to the Resume Content Extractor agent,
which owns the entire convert → quality-check → redact → extract tool chain
itself; the node's own code only runs `assign_experience_ids` afterward,
stamping a stable ID onto every work-experience bullet. Those IDs are not
cosmetic — they are the addressing scheme the Experience Optimizer's truth
floor uses later to verify bullet identity survived a rewrite.

`analyze_job` does the Markdown conversion *in code*, before the agent ever
runs, and hands the converted text straight into the Job Description Analyst's
context. There is no tool call inside that agent's turn at all — the document
is already text by the time the agent sees it.

### Stage 2 — Gap Analysis, the First Fan-In (`nodes/strategy.py`)

`run_gap_analysis` is where the code-computed-context pattern lives (see
[Agent Roles §6](agent-roles.md#6-the-code-computed-context-pattern--why-gap-analysis-has-no-tools)
for the agent-side view). The node's own docstring records *why* this shape
exists in blunt operational terms: the agent used to reconstruct the resume
itself to call a matching tool, and that reconstruction path "previously
looped the stage to timeout" in production. The fix moved the match
computation (`match_resume_to_job`) into deterministic code, run once, before
the agent starts, with its result persisted as a typed `requirement_match_report`
*and* rendered into the agent's context as ground truth. This stage is a fan-in
in the literal LangGraph sense (`extract_resume` and `analyze_job` both feed it)
and it produces the two artifacts — the match report and the `AlignmentStrategy`
— that everything in Stage 3 depends on.

### Stage 3 — Parallel Content Generation

Three independent writers fan out from Stage 2 and reconverge before assembly:

- **`write_professional_summary`** (`nodes/summary.py`) runs the writer agent,
  then immediately enforces a quality gate on whichever draft will actually
  ship — `select_recommended_draft` falls back to the first draft if the
  agent's own recommended name doesn't match any draft it produced, so the
  gate always audits the real output, never an aspirational one. A blocking
  finding (MAJOR or BLOCKER severity — banned phrasing, wrong length, wrong
  voice) raises immediately. There is no retry here; see [Section 7](#7-the-no-retry-loops-philosophy).
- **`optimize_experience`** (`nodes/experience.py`) is documented in full in
  [Agent Roles §11](agent-roles.md#11-case-study-how-experience-optimization-actually-runs) —
  the per-role fan-out, truth floor, one-repair budget, and candidate
  clarification questions all live here.
- **`optimize_skills`** (`nodes/skills.py`) runs a seven-step recipe: build
  context, run the agent, run a code-owned evidence audit
  (`validate_skills_evidence`), and — only if that audit is both *serious*
  (BLOCKER/MAJOR) and *confidently* unsupported (HIGH confidence) — spend
  exactly one scoped rewrite that tells the agent precisely which skill to
  drop, nothing else. Whatever survives that, the node then unconditionally
  re-adds any skill from the original resume that the optimizer silently
  dropped. The reasoning is explicit in the code: "the candidate's listed
  skills are facts, and deleting one only loses an ATS keyword match... the
  LLM is unreliable at preserving a long list verbatim, so completeness is
  guaranteed here in code, not left to the agent." Skill removal is trusted to
  a model only when it is loudly, confidently sure; skill *preservation* is
  never left to the model at all.

### Stage 3b — The Candidate Clarification Gate (`nodes/experience.py`)

`await_candidate_clarifications` is a fan-in with a genuine pause built into
it. If the Experience Optimizer's rewrite pass produced no unanswered
questions, the node is a no-op and the graph proceeds. If it did, and no
answers have arrived yet, the node calls LangGraph's `interrupt()` — a true
suspension of graph execution, not a queued retry — and the run stops here
until it is resumed with a candidate's answers. See [Section 10](#10-pause-and-resume--how-human-in-the-loop-works-at-the-graph-level).

### Stage 4 — ATS Assembly (`nodes/assembly.py`)

`assemble_ats_resume` builds the context from all three Stage 3 outputs plus
the original resume and job description, and asks the ATS Optimization
Specialist to produce one coherent `Resume`. The node does not trust that
assembly blindly, though: after the agent returns, `_preserve_verified_experience`
unconditionally overwrites whatever work-experience text the assembler
produced with the truth-verified `optimized_experience` entries from Stage 3.
Whatever the ATS agent did to experience content while composing the resume —
paraphrase, reformat, drop a bullet — the version that ships is always the one
that already survived the Experience Optimizer's truth floor. This is a second,
independent enforcement of the same truthfulness guarantee, applied at the
seam where two agents' outputs meet.

### Stage 5 — Quality Evaluation, the Grading Floor (`nodes/resume_quality.py`)

`evaluate_resume_quality` asks the Quality Feedback agent for narrative
commentary — wrapped in a `try/except` so that if the advisory call fails for
any reason, the stage substitutes a placeholder feedback string rather than
blocking evaluation on an optional narrative. It then does something more
consequential: `_ground_quality_dimensions` **discards** the agent's own
self-assessed accuracy, relevance, and ATS scores entirely and replaces all
three with independently computed values from deterministic engines
(`evaluate_resume_truthfulness`, `evaluate_job_alignment`,
`evaluate_rendered_structure`). The module's own comment names the reason: "the
LLM's narration is where the false positives live" — supported facts flagged
as exaggerations, single-column resumes called multi-column. After re-blending
the overall score from the grounded dimensions and applying the pass/fail gate,
the node applies one more hard override: a non-`PASS` rendered-ATS status, or
an inconclusive relevance evaluation, forces `passes_quality_gate = False`
regardless of what the blended score says. Self-certification never overrides
the rendered artifact.

### Stage 5b — Deterministic ATS Recovery (`nodes/ats_patch.py`)

`patch_ats_assembly` runs only when the rendered-ATS check comes back `FAIL` —
meaning an essential section (experience, education, or skills) rendered empty
in the assembled resume. It contains **no LLM call at all**. Its entire
strategy is one rule applied per section: if a section is empty in the
assembled resume but present upstream in already-verified typed state, refill
it from that upstream source (`optimized_experience`, `optimized_skills`, or
the original resume's education) and re-grade. The module's docstring is
explicit about why this is a single deterministic pass rather than a retry
loop: "a single typed-state restore is exhaustive; if it does not fix the FAIL,
a second identical pass cannot either." If the restore still doesn't clear the
gate — because the section was genuinely empty upstream too — the run is
flagged for human review via `is_ats_unrecoverable`. A dated `TODO` in this
file records an open architectural question the team has deliberately
deferred: *why* the ATS assembler drops essential sections in its own context
in the first place, noting the same underlying cause as the Stage 2 timeout —
an agent working from a lossy view of state can omit what it cannot fully see
— and flags this patch node as a safety net whose permanence is still an open
question, not settled infrastructure.

### Stage 6 — PII Rehydration, on Every Path (`nodes/rehydrate_pii.py`)

`rehydrate_pii` is plain code with no agent involved, and it is the one node
every route out of quality evaluation shares — the clean-continue path and the
post-patch path both land here before the render gate is ever evaluated. It
walks every string leaf of the assembled resume, replacing PII placeholders
(`[PERSON_1]`, etc.) with real values from a run-scoped mapping store, longest
placeholder first so `[PERSON_1]` is never mistaken for a prefix of
`[PERSON_10]`. Running this unconditionally on every path — rather than only
on the path that goes on to render — guarantees the resume returned to the
caller always carries real candidate data, whether or not the run ultimately
renders anything.

### Stage 7 — Conditional Render (`nodes/render.py`)

`render_final_resume` is also plain code, gated by the final routing decision
(see [Section 5](#5-the-three-routing-decisions)). It always writes Markdown
and DOCX — both pure-Python, both guaranteed to succeed on any OS — and treats
PDF rendering as best-effort: if the LaTeX toolchain is missing or compilation
fails, the PDF is skipped with a recorded, human-readable reason and the run
still completes successfully with the other two formats.

---

## 5. The Three Routing Decisions

Every branch point in the graph is a named function in `graph.py`, never an
inline conditional buried in a node:

```text
_route_after_candidate_clarifications   (after await_candidate_clarifications)
  clarification_answers present  -> "rewrite_experience" -> optimize_experience
  no answers                     -> "assemble_resume"    -> assemble_ats_resume

_route_after_ats_check                  (after evaluate_resume_quality)
  rendered_structure_evaluation.status == FAIL  -> "patch"    -> patch_ats_assembly
  PASS or INCONCLUSIVE                          -> "continue" -> rehydrate_pii
  (INCONCLUSIVE was already flagged for human review inside the
   QA node itself -- nothing was built for the patch node to inspect)

_route_after_quality                    (after rehydrate_pii)
  should_render_resume(quality_report) is True          -> "render" -> render_final_resume
  gate failed, but render_draft_on_gate_fail flag is on -> "render" -> render_final_resume
  gate failed, flag is off                              -> "end"   -> END
```

The `render_draft_on_gate_fail` branch is worth calling out: it exists so a
development-mode run can still produce inspectable Markdown/DOCX drafts even
when the quality gate fails, without that draft ever being mistaken for a
passing result — `quality_report.passes_quality_gate` stays `False` on the
returned state regardless of whether files were written.

---

## 6. run_agent_task — The One Seam Between the Graph and CrewAI

Every node that calls an agent does so through exactly one function:
`run_agent_task` in `src/orchestration/crew_task_execution.py`. This is
deliberately not named `runner` — that name is reserved for the pipeline's
public entry points (`runner.py`) — because this module has one narrower job:
turn one CrewAI `Agent` plus one configured task into one validated Pydantic
object.

```text
   run_agent_task(agent, task_name, context, output_model, run_id)
           |
           v
   load task_name's description + expected_output from tasks.yaml
           |
           v
   agent HAS tools?  ---- yes ---->  leave response_format unset,
           |                          constrain via Task(output_pydantic=...)
           no                         (tools need the tool-call loop to run;
           |                          response_format would short-circuit it)
           v
   agent HAS NO tools -----------> set agent.llm.response_format = output_model
                                    (guarantees well-formed JSON on first try)
           |
           v
   Crew(agents=[agent], tasks=[task], process=Process.sequential).kickoff()
   -- serialized process-wide by _KICKOFF_LOCK, see Section 11 --
           |
           v
   _validate_agent_output(result.raw, output_model, ...)
     -- extracts the outermost {...} block, ignoring code fences/prose
     -- output_model.model_validate_json(...)
     -- raises AgentOutputError on anything that doesn't parse/validate
           |
           v
   validated Pydantic instance returned to the calling node
```

The branch on `agent.tools` is a real, documented lesson rather than a stylistic
choice: setting `response_format` tells the provider to return structured JSON
in the *first* response, which causes tool-carrying agents to skip their tool
calls entirely and return an empty schema skeleton. Tool-free agents get the
opposite treatment — `response_format` is safe and guarantees well-formed JSON
without needing a coercion retry. Either way, the node still runs its own
`_validate_agent_output` pass on the raw text, so no node ever depends on
CrewAI's internal schema coercion succeeding silently.

---

## 7. The "No Retry Loops" Philosophy

Two independent node modules — `summary.py` and `ats_patch.py` — state the same
policy in their own words, which is worth surfacing as one explicit,
system-wide rule rather than two coincidentally similar comments: **this
pipeline does not retry an LLM call until it happens to pass a quality bar.**

```text
   INSTEAD OF:                              THIS PIPELINE DOES:
   loop { call LLM; grade; if fail,          call LLM once; grade once;
          retry with feedback; }             if it fails a HARD gate,
          (unbounded, can mask a              stop the run and surface
           systematically bad draft            the exact findings to a
           as "eventually passing")            human or the candidate

   exception: exactly ONE bounded repair pass exists in the system --
   the Experience Optimizer's rewrite/review loop (Section 11 of Agent
   Roles) and the Skills Optimizer's one scoped rewrite -- and both are
   disclosed, single-shot, and fall back to a known-safe prior state
   (source bullets; the pre-rewrite skill list) rather than looping again.
```

The `ats_patch` module makes the underlying reasoning explicit: because its
recovery step is deterministic (fill an empty section from typed state that
either does or does not exist), a second identical attempt cannot produce a
different outcome — retrying would be pure latency with no chance of a
different result. The `summary` module's version of the same policy is about
honesty rather than determinism: silently looping an LLM until it stops
tripping a banned-phrase check would eventually produce a technically-passing
draft that says nothing true about a thin resume. Failing loudly, once, with
named findings, is the design's answer to that risk.

---

## 8. The Human Review Escalation Policy — One Documented Home

Before `src/orchestration/human_review_policy.py` existed, the same escalation
logic lived as two separate inline boolean expressions inside two different
nodes (quality evaluation and ATS patching) — meaning the question "when does
this system hand a run to a human?" had two possible answers to go find. The
module's own docstring records this as the reason it was centralized.

The policy itself is narrow and precisely two-pronged:

```text
ESCALATE TO HUMAN REVIEW when, and only when:

  1. UNVERIFIABLE  -- rendered-ATS check returned INCONCLUSIVE (the resume
     could not even be rendered to inspect). Discovered at the QA stage.
     There is nothing to patch, because there is nothing to look at.

  2. UNRECOVERABLE -- rendered-ATS check FAILed, the deterministic section
     restore ran, and the re-graded result is STILL not PASS (the section
     was empty upstream too). Discovered at the patch stage, after
     recovery was already attempted.

A plain FAIL at the QA stage, by itself, is NOT an escalation -- it is
recoverable and routes to patch_ats_assembly first. Only exhausted
recovery becomes a human-review flag.
```

The same module owns `derive_run_disposition`, which folds `human_review_required`,
the quality gate's pass/fail state, and whether the candidate has unanswered
clarification questions into one `RunDisposition` enum with an explicit
precedence order: a human-review escalation always wins, a failed quality gate
beats unanswered candidate questions, and unanswered candidate questions beat a
plain successful render (because a resume that rendered but still has open
questions is better served by answering the sheet and re-running than by
treating it as finished).

---

## 9. Failure Taxonomy — Three Kinds of "The Run Stopped"

`src/orchestration/exceptions.py` defines exactly two custom exception types,
and its own docstring is explicit that the third category — everything else —
is deliberately left as ordinary Python exceptions:

```text
PipelineQualityGateError
  -- a stage's output was WELL-FORMED but failed a content/style judgment
     (banned phrase, wrong length, generic opener).
  -- expected, input-dependent: most often the resume is too thin in
     measurable evidence for this job, not a system bug.
  -- user_action tells the candidate what to change in THEIR input.

AgentOutputError
  -- the LLM's raw response could not even be parsed into its schema
     (no JSON object found, or it didn't match output_model).
  -- a formatting hiccup, most often transient.
  -- user_action tells the user to simply retry.

everything else (ValueError, RuntimeError, AssertionError, ...)
  -- a genuine programming error: a pipeline invariant was violated
     (e.g. state a node should have populated is still None).
  -- crashes loudly with a full traceback on purpose.
  -- there is no user-facing action for a bug -- a developer needs to see it.
```

The CLI-facing layer catches only the first two types and turns them into a
readable message instead of a stack trace; it never suppresses the third
category. This taxonomy is what lets the same `raise` statement mean two very
different things to two different audiences — a candidate reading "add
measurable outcomes to your bullets," and a developer reading a full traceback
for a state invariant that should never have been violated — without either
audience seeing the wrong one.

---

## 10. Pause and Resume — How Human-in-the-Loop Works at the Graph Level

`src/orchestration/runner.py` exposes exactly two public entry points, and
their relationship is the core of how this system supports a pause that can
last arbitrarily long — hours or days, however long a candidate takes to
answer a clarification sheet — without holding any process alive in the
meantime.

```text
tailor_resume(resume_path, jd_path)
  -- opens a fresh, run_id-scoped SQLite checkpointer
  -- invokes the compiled graph from empty state
  -- if the graph interrupts (Stage 3b, unanswered clarifications):
       snapshot the state, write a paused-run manifest + clarification
       sheet to disk, ARCHIVE the checkpoint db into that paused-run folder
       (moved, not deleted -- the run's history is preserved for resume)
  -- otherwise: persist the completed OrchestrationResult and clean up
     the checkpoint db and the PII mapping

resume_paused_run(paused_run_path)
  -- loads the manifest + the checkpointer archived at that path
  -- loads the candidate's ANSWERED clarifications from the sheet on disk
  -- invokes the graph with a LangGraph Command(resume=..., update={
       "clarification_answers": answered_clarifications})
     -- this resumes execution from the exact interrupt() boundary,
        with the new state field populated
  -- the routing function _route_after_candidate_clarifications now sees
     clarification_answers is non-empty, so the graph re-enters
     optimize_experience once more -- only for the roles that had
     questions -- then proceeds to assembly as normal
```

A run's disposition governs what gets cleaned up. A `NEEDS_CANDIDATE_INPUT`
disposition deliberately preserves both the checkpoint database and the PII
mapping, because `rehydrate_pii` will need that mapping again after the run
resumes, and the checkpoint history is what makes the resume possible at all.
Every other terminal disposition — a clean render, a quality-gate failure, an
escalation to human review — deletes both, because none of those outcomes will
ever be resumed through this same run.

This is the payoff of treating the graph, not any agent, as the sole owner of
control flow: pausing and resuming a multi-agent pipeline reduces to
serializing and restoring one `TypedDict`, because that `TypedDict` is the
complete and only record of where the run is.

---

## 11. The Shared-Resource Lock — A Real Concurrency Bug and Its Fix

`crew_task_execution.py` carries a process-wide `threading.Lock` around every
`Crew(...).kickoff()` call, and the comment explaining it is worth reading as a
case study in when a lock is actually the right tool, because the file states
the general principle before applying it.

A lock is needed only when three things are simultaneously true: real
concurrency, a shared mutable resource, and a resource that does not
synchronize itself. This pipeline hits all three at once. The graph itself
runs Stage 1 and Stage 3 nodes concurrently, and the Experience Optimizer node
adds its own thread pool on top of that — so several `run_agent_task` calls can
be in flight at the same moment. Every one of those calls, regardless of which
node it came from, drives a CrewAI `kickoff()` that writes to the *same* SQLite
file on disk (CrewAI's internal task-output store), opened without a busy
timeout. The first time two of those writes genuinely overlapped, one failed
immediately with a locked-database error — an intermittent, load-dependent
failure that is the signature of a race, not a logic bug, and it surfaced
during a real end-to-end run in the experience stage specifically because that
is where the thread pool concentrates the most concurrent `kickoff()` calls.

Since the pipeline does not own CrewAI's internal connection and cannot make it
tolerant of concurrent writers, the only lever available was to remove the
overlap entirely: serialize every `kickoff()` call process-wide behind one
lock. The critical section is deliberately as small as correctness allows, but
it must wrap the entire `kickoff()` call, because that is where the hidden
write happens.

---

## 12. Why This Shape Fits the Project

Resume tailoring needs to tolerate LLM variation without letting that
variation leak into *when* things happen or *whether* a bad result reaches a
candidate. The graph is where that boundary is drawn. Every place an agent's
output might be wrong — a thin summary, an unsupported skill, a truncated
section, an ATS-unsafe render — has a corresponding, code-owned check sitting
in the graph's node layer or its routing functions, never inside the agent's
own reasoning. The architecture is deterministic exactly where it needs to be
(sequencing, gating, escalation, recovery) and probabilistic only where
probability is actually useful (drafting language, synthesizing strategy). That
split, not any single node, is what makes the system's truthfulness and
quality guarantees hold across an unbounded range of resumes and job
descriptions it has never seen.

---

## 13. Future Considerations

**Resolving the open question behind the ATS patch's safety net.** The `TODO`
in `ats_patch.py` names a real architectural question the team has not yet
answered: whether an agent working from a lossy context genuinely cannot see a
section it needs to include, or whether the deeper fix is to pre-compute those
sections into the assembler's context the same way Stage 2 pre-computes the
match report. Until that investigation resolves, the patch node's role is
ambiguous between "permanent, disclosed recovery step" and "temporary guard
around an upstream defect" — worth resolving deliberately rather than letting
the safety net become load-bearing infrastructure by default.

**Whether relevance should be re-graded after a patch recovery.** The same file
notes that a restored skills section can raise JD keyword coverage, but the
patch node currently reuses the pre-patch relevance score rather than
re-evaluating job alignment against the recovered resume. This is a real gap
between what the patched resume actually contains and what its quality report
claims about relevance — an architectural inconsistency between "the ATS
dimension is re-graded after recovery" and "the relevance dimension is not,"
worth deciding one way or the other rather than leaving asymmetric.

**Whether the escalation policy should account for repeated patch cycles.**
Today, a single deterministic patch attempt either resolves a FAIL or escalates
immediately — there is no notion of a run that patches successfully but does so
repeatedly across many candidate resumes in a way that would suggest an
upstream systemic issue rather than a per-run anomaly. Whether that pattern
belongs in this graph's policy or in a layer above it (see
[Observability](observability.md)) is an open architectural boundary question.
