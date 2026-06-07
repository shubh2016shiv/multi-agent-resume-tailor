# Resume Tailor — Orchestration & System Design
## How the Multi-Agent Pipeline Is Wired, Why It Is Wired That Way, and How to Extend It

> **Scope:** `src/orchestration/` · the agent modules in `src/agents/` · the contracts in `src/data_models/`
> **Audience:** A developer who knows Python and the basics of generative / agentic AI, joining this project
> **Edition:** 2026
> **Read this before you read any code.** The code shows you *what* runs. This document shows you *why it is shaped this way* — the design decisions and patterns you cannot recover by reading function bodies.

---

## Table of Contents

1. [The Problem This System Solves](#1-the-problem-this-system-solves)
2. [The 60-Second Mental Model](#2-the-60-second-mental-model)
3. [The Whole Pipeline at a Glance](#3-the-whole-pipeline-at-a-glance)
4. [The Single Most Important Idea: Two Frameworks, Two Jobs](#4-the-single-most-important-idea-two-frameworks-two-jobs)
5. [The Shared State: A Typed Blackboard](#5-the-shared-state-a-typed-blackboard)
6. [The Six Recurring Design Patterns](#6-the-six-recurring-design-patterns)
7. [Walking the Pipeline, Stage by Stage](#7-walking-the-pipeline-stage-by-stage)
8. [The Context Compression Layer](#8-the-context-compression-layer)
9. [How Failure Is Handled](#9-how-failure-is-handled)
10. [How to Extend the System](#10-how-to-extend-the-system)
11. [Glossary of Patterns and Terms](#11-glossary-of-patterns-and-terms)

---

## 1. The Problem This System Solves

The product takes a candidate's resume (a PDF or Word file) and a target job description, and produces a tailored, ATS-safe resume as a PDF — one that is rewritten to match the job without inventing anything the original resume does not support.

That sounds like a single task you could hand to one large language model: "here is a resume and a job, rewrite it." In practice, doing that with one prompt fails in predictable ways. The model loses track of which claims were real and which it embellished. It optimizes the summary and forgets the skills section. It stuffs keywords until the text reads like spam. It cannot tell you *why* it made a choice, so you cannot trust it, debug it, or improve it.

The discipline that solves this is **decomposition**. Instead of one model doing everything, the work is split into a sequence of narrow, single-purpose specialists — one that parses the resume, one that reads the job, one that finds the gaps, one that writes the summary, one that rewrites experience, one that optimizes skills, one that assembles and checks ATS compatibility, and one that audits the final quality. Each specialist is an **agent**: a language model given a focused role, a small set of tools, and a strict output shape.

But decomposition creates a new problem. Now something has to *coordinate* eight specialists: decide what runs when, pass each one's output to the next, run independent ones in parallel, and decide whether the result is good enough to ship. That coordinator is the **orchestration layer** — the subject of this document. It lives entirely in `src/orchestration/`.

```
THE CORE TENSION THIS ARCHITECTURE RESOLVES

   One giant prompt                         Eight coordinated specialists
   ───────────────                          ─────────────────────────────
   + simple to write                        + each piece is testable
   - untraceable                            + each output is a typed contract
   - cannot debug one step                  + independent steps run in parallel
   - silently invents facts                 + a final gate enforces honesty
   - no quality gate                        - SOMETHING must coordinate them
                                              └──► that something is src/orchestration/
```

---

## 2. The 60-Second Mental Model

Picture a factory assembly line with stations. A chassis enters at one end; each station does one operation and passes the product to the next; at the end an inspector decides whether it ships.

```
   RAW INPUT                  STATIONS (each = one specialist agent)            OUTPUT
   ─────────                  ──────────────────────────────────────           ──────
   resume.pdf  ┐
               ├─►  parse ─► analyze ─► strategize ─► write ─► assemble ─► inspect ─►  resume.pdf
   job.txt     ┘                                                              │        (tailored,
                                                                              │         ATS-safe)
                                                                    passed? ──┘
                                                                       │
                                                              no ──────┴────── yes
                                                          (stop, no PDF)   (render PDF)
```

Three facts carry most of the understanding:

1. **The "stations" are AI agents.** Each is a language model with a single job. They live in `src/agents/`, one folder per agent.
2. **The "assembly line" is a graph.** The order, the parallelism, and the final yes/no decision are defined declaratively in `src/orchestration/graph.py`. Nothing in an agent folder knows about the line; the line knows about them.
3. **The "product moving down the line" is a typed object.** Every station reads a shared clipboard, does its work, and writes its result back onto the clipboard. That clipboard is `src/orchestration/state.py`.

If you remember only one sentence: **agents reason, the orchestrator coordinates, and code — never an agent — makes the final go/no-go decision.**

---

## 3. The Whole Pipeline at a Glance

This is the entire system. Every box is a node in the graph. Read it top to bottom; arrows are the order of execution; boxes side by side run **at the same time**.

```
                                   ┌─────────┐
                                   │  START  │   inputs: resume_path, jd_path
                                   └────┬────┘
                          ┌─────────────┴─────────────┐
                          ▼                           ▼          STAGE 1: INGESTION
                 ┌─────────────────┐         ┌─────────────────┐ (parallel — independent)
                 │ extract_resume  │         │   analyze_job   │
                 │  PDF → Resume   │         │  text → Job     │
                 └────────┬────────┘         └────────┬────────┘
                          └─────────────┬─────────────┘
                                        ▼                        STAGE 2: STRATEGY
                              ┌───────────────────┐              (fan-in — needs both)
                              │  run_gap_analysis │
                              │  → AlignmentStrategy
                              └─────────┬─────────┘
              ┌───────────────────────┬─┴───────────────────────┐
              ▼                       ▼                          ▼   STAGE 3: CONTENT
   ┌────────────────────┐ ┌────────────────────┐ ┌────────────────────┐ (parallel —
   │ write_professional │ │ optimize_experience│ │   optimize_skills  │  three writers
   │      _summary      │ │  → Experience[]    │ │  → Skills[]        │  run together)
   │ → ProfessionalSumm │ │  (+ audit loop)    │ │  (+ audit loop)    │
   └─────────┬──────────┘ └─────────┬──────────┘ └─────────┬──────────┘
             └───────────────────────┬┴───────────────────-┘
                                     ▼                       STAGE 4: ASSEMBLY
                          ┌─────────────────────┐            (fan-in — needs all three)
                          │ assemble_ats_resume │
                          │ → AtsOptimizedResume│
                          └──────────┬──────────┘
                                     ▼                       STAGE 5: QUALITY
                          ┌─────────────────────┐
                          │ run_quality_assurance
                          │ → QualityReport     │
                          │ + CODE-OWNED GATE   │ ◄── code sets pass/fail from the score
                          └──────────┬──────────┘
                                     ▼
                              ╱ passed the gate? ╲           STAGE 6: CONDITIONAL RENDER
                            ╱                      ╲
                      no  ╱                          ╲  yes
                        ▼                              ▼
                   ┌─────────┐              ┌──────────────────────┐
                   │   END   │              │  render_final_resume │
                   │ (no PDF,│              │  Resume → PDF (code, │
                   │  show   │              │  no agent)           │
                   │ feedback)│             └──────────┬───────────┘
                   └─────────┘                         ▼
                                                  ┌─────────┐
                                                  │   END   │  output: rendered_resume_path
                                                  └─────────┘
```

Everything else in this document is a zoom-in on one part of that picture.

---

## 4. The Single Most Important Idea: Two Frameworks, Two Jobs

This system uses **two** orchestration technologies, and confusing their roles is the number-one way to get lost. They operate at different altitudes and never do each other's job.

```
   ALTITUDE                 FRAMEWORK        OWNS                         LIVES IN
   ────────                 ─────────        ────                         ────────
   MACRO (the whole line)   LangGraph        the graph: which node runs   src/orchestration/
                                             when, what runs in parallel, graph.py, state.py,
                                             the shared state, the yes/no  runner.py, nodes/
                                             routing at the end

   MICRO (one station)      CrewAI           one agent doing one          src/agents/*/agent.py
                                             reasoning task: the LLM       (the factories) and
                                             call, the tools, the          crew_task_execution.py
                                             retries, the typed output     (the seam)
```

**LangGraph** is the macro conductor. It knows there are nine stations and the shape of the line: ingestion runs two stations in parallel, strategy waits for both, three writers then run together, assembly waits for all three, quality runs, and a final branch decides whether to render. LangGraph knows *nothing* about how any single agent thinks. It only knows "call this node, hand it the clipboard, take back what it writes."

**CrewAI** is the micro engine for a single station. When a node needs an agent to actually reason, it hands one CrewAI agent one task and waits for one typed result. CrewAI manages the messy parts of a single LLM interaction: building the prompt, exposing tools the agent may call, retrying on transient failures, and coercing the model's free-form text into a strict Pydantic object.

The **seam** between the two altitudes is a single small adapter (`crew_task_execution.py`). A node calls it with: *an agent, the name of a task, the context string, and the output type it must produce.* The adapter runs exactly one CrewAI agent-and-task and returns one validated object — or raises if the model failed to produce that type. This is the only place the two frameworks touch.

```
   ONE NODE'S INTERNAL ANATOMY (this is the repeating unit of the whole system)

   LangGraph calls the node ──►  ┌──────────────────────────────────────────────┐
   with the shared state         │  NODE (a plain Python function)               │
                                 │                                              │
                                 │  1. read what it needs from the state        │
                                 │  2. FORMATTER: compress that into a tight     │
                                 │     context string (token discipline)        │
                                 │  3. FACTORY: build the right CrewAI agent     │
                                 │  4. SEAM: run one agent + one task ──────────┐│
                                 │     and get back ONE typed object           ││
                                 │  5. (some nodes) AUDIT the object in code,   ││
                                 │     rewrite once if it fails                 ││
                                 │  6. return { state_field: typed object }     ││
                                 └──────────────────────────────────────────────┘│
                                                  │                              │
                                                  └── CrewAI runs the LLM here ──┘
   LangGraph merges the returned
   dict back into the shared state ◄─────────────
```

Once you see that every node is the same five-or-six-step shape, the whole codebase becomes legible. The differences between nodes are only: which formatter, which agent, and whether there is an audit loop.

---

## 5. The Shared State: A Typed Blackboard

The nine nodes never call each other. They communicate **only** through a shared object that flows down the line. This is a classic multi-agent pattern called the **blackboard**: a common workspace that every specialist reads from and writes to, with no specialist needing to know who runs before or after it.

That object is defined in `state.py`. Think of it as a form with one slot per stage. Every slot starts empty (`None`). A node fills in its slot and leaves the rest untouched.

```
   THE BLACKBOARD FILLS UP AS THE LINE RUNS
   (each row = state after that stage; "·" = still empty)

   field                  after  after   after   after    after   after
                          START  ingest  strat.  content  assem.  quality
   ─────────────────────  ─────  ──────  ──────  ───────  ──────  ───────
   resume_path / jd_path   set    set     set     set      set     set
   resume                   ·     SET     set     set      set     set
   job_description          ·     SET     set     set      set     set
   alignment_strategy       ·      ·      SET     set      set     set
   professional_summary     ·      ·       ·      SET      set     set
   optimized_experience     ·      ·       ·      SET      set     set
   optimized_skills         ·      ·       ·      SET      set     set
   optimized_resume         ·      ·       ·       ·       SET     set
   qa_report                ·      ·       ·       ·        ·       SET
   rendered_resume_path     ·      ·       ·       ·        ·        ·   ← only set if gate passes
```

Two design rules make this safe, and they are the whole reason the blackboard works:

1. **A `None` slot means "the node that produces this has not run yet."** A node must never read a slot that an earlier stage was supposed to fill unless the graph guarantees that earlier stage already ran. The *graph topology is what provides that guarantee* — this is why the order lives in `graph.py` and not in the agents.

2. **A node returns only the slot(s) it owns**, as a small partial update. LangGraph merges that partial back into the whole. This is what makes parallelism safe: three writers in Stage 3 each return a *different* slot, so their writes never collide when they finish at different times and get merged back.

The blackboard is also why you can reason about the system without reading every agent. To know what `optimize_skills` can depend on, you do not read its code — you look at the graph, see it runs after strategy, and conclude it may read `resume`, `job_description`, and `alignment_strategy`, because those slots are guaranteed filled by then.

---

## 6. The Six Recurring Design Patterns

The same handful of patterns appear over and over. Learn them once here; you will then recognize them everywhere in the code.

### 6.1 The Agent Factory — separate *building* an agent from *running* it

Every agent folder exposes a single **factory**: a function that builds a configured agent and hands it back, and does nothing else. It does not run the agent, does not construct a task, does not know about the graph. Building and running are deliberately split: the agent module owns *how this specialist is configured*; the orchestration layer owns *when and with what input it runs*. This separation is why agents are testable in isolation and why the same agent could be reused in a different pipeline without change.

### 6.2 Agents Reason, Code Measures — never let an LLM grade itself

This is the spine of the whole system's trustworthiness. Anything that is a *judgment* (is this bullet strong? is this claim honest? which summary is best?) is the agent's job. Anything that is a *measurement* (does this text contain keyword X? is the keyword density between 2% and 5%? is the score above the threshold?) is **code's** job, because measurements must be deterministic and must not be subject to a model's mood.

```
   THE DIVIDING LINE, APPLIED THROUGHOUT

   JUDGMENT  → the agent (LLM)              MEASUREMENT → code (deterministic)
   ──────────────────────────              ──────────────────────────────────
   rewrite this bullet for impact          count keyword occurrences
   is this skill actually evidenced?       compute keyword density %
   score accuracy / relevance / ATS        decide pass/fail from the score
   choose the strongest summary draft      render the final PDF
```

You will see this concretely in two places: the assembly agent produces the *assembled resume and its reasoning*, but the ATS *score* is computed by code; and the quality agent produces *dimension scores*, but the final *pass/fail boolean* is computed by code. An agent never decides whether its own work is good enough to ship.

### 6.3 The Code-Owned Audit + Bounded Rewrite Loop — self-correction without runaway

Two of the writing stages (experience and skills) do not trust the first draft. After the agent writes, **code** runs a mechanical/judgment audit over the *typed output*. If the audit finds a serious problem (for example, a skill the resume does not actually support), the node feeds that specific feedback back to the agent and asks for **exactly one** rewrite.

```
   THE BOUNDED REWRITE LOOP (used by experience & skills)

   agent writes ─► code audits the typed output ─► serious findings?
                                                        │
                                          no ───────────┴─────────── yes
                                           │                          │
                                    accept, done            feed findings back,
                                                            agent rewrites ONCE
                                                                       │
                                                            accept the rewrite, done
                                                            (no second loop — bounded)
```

The loop is **bounded to one retry on purpose**. An unbounded "keep fixing until perfect" loop is how multi-agent systems burn money and hang. One correction pass catches the common, fixable mistakes; anything still wrong after that is surfaced honestly rather than ground on forever.

### 6.4 Fan-Out / Fan-In Parallelism — independence is a resource

Where steps do not depend on each other, they run at the same time. Parsing the resume and reading the job are independent, so they fan out from START and run together. The three content writers (summary, experience, skills) all depend only on the strategy, so they fan out from it and run together. A **fan-in** node (strategy after ingestion; assembly after the three writers) is a barrier: LangGraph waits for *all* incoming branches to finish before it starts. The graph's shape encodes the project's true dependency structure — nothing runs before its inputs exist, and nothing waits longer than it must.

### 6.5 The Deterministic Quality Gate — the one decision that must not be a guess

The final go/no-go — is this resume good enough to render and hand to the candidate? — is the most consequential decision in the system, so it is deliberately the *least* AI-driven. The quality agent produces scores; then code applies one fixed rule (the overall score must clear a fixed threshold) to set an authoritative boolean. The graph routes on *that boolean*. The agent's own opinion of whether it passed is treated as advisory and overwritten. This is pattern 6.2 applied to the highest-stakes moment in the pipeline.

### 6.6 Contract-First Boundaries — every handoff is a typed object

No node ever passes free-form text to the next. Every agent's output is coerced into a specific Pydantic model (the "contracts" in `src/data_models/` and a few in the agent folders). The pipeline is therefore a chain of typed handoffs: `Resume → JobDescription → AlignmentStrategy → (three section contracts) → AtsOptimizedResume → QualityReport → PDF`. If an agent fails to produce its contract, the seam raises immediately rather than passing malformed data downstream. The contracts are the guardrails that keep eight independent models composable.

---

## 7. Walking the Pipeline, Stage by Stage

Now we apply the patterns to the actual line. For each stage: what enters, what the specialist decides, what code owns, and what leaves.

### Stage 1 — Ingestion (two specialists, in parallel)

Two independent jobs: turn the resume document into a structured `Resume`, and turn the job posting into a structured `JobDescription`. They share no inputs, so they fan out from START and run together. Note an important sub-pattern: turning a PDF into clean text is *mechanical*, so it is done by a tool in code **before** the agent is involved; the agent only does the *judgment* part (reading messy text into clean structured fields). This is pattern 6.2 again, at the very front door.

```
   START ─┬─►  extract_resume :  PDF ──(code: convert)──► text ──(agent: structure)──► Resume
          └─►  analyze_job    :  job text ──────────────(agent: structure)──────────► JobDescription
```

### Stage 2 — Strategy (fan-in)

`run_gap_analysis` is a barrier: it needs both the resume and the job. It compares them and produces an `AlignmentStrategy` — the tailoring plan that every downstream writer obeys (which gaps to address, which keywords to integrate, how to position the candidate). This is the one place the system decides *strategy*; the writers that follow are executors of it.

### Stage 3 — Content generation (three specialists, in parallel, two with audit loops)

Given the strategy, three writers run together because each owns a different section and they do not depend on one another:

- **Summary writer** — produces several summary drafts and recommends one. It is given a self-check tool to consult while it reasons.
- **Experience optimizer** — rewrites work-experience bullets, then runs the *bounded rewrite loop* (6.3): code audits the rewritten experience for unsupported or inflated claims and asks for one correction pass if needed.
- **Skills optimizer** — selects, orders, and categorizes skills, then runs the same *bounded rewrite loop*: code audits whether each listed skill is actually evidenced in the resume and asks for one correction if not.

```
   strategy ─┬─►  summary writer      → ProfessionalSummary
             ├─►  experience optimizer → Experience[]   (write → audit → rewrite once?)
             └─►  skills optimizer     → Skills[]        (write → audit → rewrite once?)
```

### Stage 4 — ATS assembly (fan-in)

`assemble_ats_resume` waits for all three writers. The assembly agent stitches the summary, the rewritten experience, the optimized skills, and the untouched contact/education into one coherent `Resume`, standardizes the section headers, orders the sections for machine parsing, and checks its own work against ATS rules using read-only tools. Crucially it produces a **slim** output — the assembled resume plus its decision notes — and does **not** render anything or score itself; rendering and ATS scoring are code's job (6.2). Its contract is `AtsOptimizedResume`.

### Stage 5 — Quality assessment (the gate)

`run_quality_assurance` is the final auditor. The quality agent scores the optimized resume on three weighted dimensions — accuracy (is every claim supported by the original?), relevance (does it answer the job?), and ATS compatibility — grounding each score in a real tool check rather than impression. It produces a `QualityReport` with an overall score. Then **code applies the gate** (6.5): it sets the authoritative pass/fail boolean from the score against a fixed threshold, overwriting whatever the agent guessed.

### Stage 6 — Conditional render (code, not an agent)

This is the only **conditional** branch in the pipeline. A small routing function reads the code-owned gate boolean and sends the line one of two ways:

```
                        ┌─ gate passed?  (a deterministic boolean, not an LLM opinion)
                        │
              no ───────┴─────── yes
               │                  │
            END (no PDF;      render_final_resume
            the report's      (mechanical: structured Resume → LaTeX → PDF;
            feedback tells     a PLAIN CODE node, no agent, because there is
            the user why)      no judgment left to make)
                                  │
                                 END  → the PDF path is recorded on the blackboard
```

Rendering is a plain code node for the same reason ingestion's PDF-to-text step is: it is mechanical. There is no decision left to make once the resume has passed — only the deterministic act of laying out already-decided content into a template and compiling it.

### The public entry point

A single function in `runner.py` is the front door for the entire system. It builds the graph once (the graph is stateless and reused across calls), creates a fresh blackboard for each run, invokes the line, and returns one final typed result containing the optimized resume, the quality report, and — if the gate passed — the PDF path. A caller never touches LangGraph, CrewAI, or any agent directly; they call one function and get one typed result.

---

## 8. The Context Compression Layer

Between reading the blackboard and calling an agent, every node passes its data through a **formatter** (`src/formatters/`). This is not cosmetic — it is a deliberate cost-control layer, and it connects directly to the token-efficiency discipline documented in `docs/token-efficiency-guide.md`.

An agent is charged for every token of context you send it, on every internal step it takes. Handing an agent five full JSON objects with all their metadata is the pipeline-level version of the "read 25 files to answer a question about three" waste. The formatter for each stage filters the blackboard down to *only the fields that stage's agent actually needs*, and renders them in a compact notation rather than verbose JSON.

```
   WHY THE FORMATTER EXISTS

   full blackboard objects  ──►  FORMATTER  ──►  tight, stage-specific context  ──►  agent
   (everything, verbose JSON)    (keep only      (only what this agent needs,
                                  what's needed,   compact notation)
                                  compress)
                                                  └─► fewer tokens, lower cost,
                                                      sharper agent focus
```

The design rule: a node's formatter is the single place that decides *what an agent is allowed to see*. If an agent is making poor decisions, the formatter is the first place to look — it may be starving the agent of a field, or drowning it in irrelevant ones.

---

## 9. How Failure Is Handled

The system is built to **fail loudly and early** rather than limp along producing quietly wrong output. Three layers enforce this:

```
   LAYER                WHAT IT CATCHES                         WHAT IT DOES
   ─────                ───────────────                         ────────────
   agent config         a misconfigured agent (missing role,    raises immediately at
   validation           goal, or model in config)               build time — never runs
                                                                 with half a config

   the seam             an agent that ran but failed to          raises immediately —
   (typed output)       produce its required contract            malformed data never
                                                                 flows to the next stage

   final assembly       the line finished but a required         raises with the exact
   (the runner)         slot on the blackboard is still empty    list of empty slots
```

The philosophy is that a wrong resume is worse than no resume. Every guardrail prefers a clear, immediate error that names the broken stage over a plausible-looking output that silently dropped a section or invented a claim. The one place this is softened is the quality gate: a resume that *fails* the gate is not an error — it is a valid outcome (the line ends cleanly with no PDF and actionable feedback for why).

---

## 10. How to Extend the System

When you add a capability, you will almost always be adding a **stage** (a new agent in the line) or a **node** (a code step). Follow the grain of the existing design; do not invent a new shape.

**To add a new agent stage:**

```
   1. Create the agent module:  src/agents/<your_agent>/
        - a factory that builds and returns the configured agent (and nothing else)
        - its output contract (reuse a model in src/data_models/ if one fits;
          only add a new model if the shape genuinely does not exist yet)
        - an architecture .md describing it (like the other agent folders)
   2. Add its config:  a block in src/config/agents.yaml + a task in src/config/tasks.yaml
   3. Add the orchestration node:  src/orchestration/nodes/<stage>.py
        - read the blackboard → formatter → factory → seam → return its slot
        - add an audit loop ONLY if its output needs code-checking (follow 6.3)
   4. Add a slot for its output in state.py
   5. Wire it into graph.py:  register the node, add the edges that place it in the
      dependency order (parallel if independent, fan-in if it needs several inputs)
   6. Prove it in isolation first with a trigger script before wiring it in
```

**The decisions that are yours to get right, not the agent's:**

- *Does this stage depend on another, or is it independent?* That single answer determines whether it gets a sequential edge or joins a parallel fan-out — and it is the only thing that determines correctness of ordering.
- *Is any part of this stage a measurement?* If so, that part is code, not the agent (6.2).
- *Does its output need self-correction?* If yes, add a bounded one-shot audit loop (6.3); never an unbounded one.
- *Does it introduce a yes/no decision?* If so, the decision boolean is computed by code and the graph routes on it (6.5) — an agent never routes the graph.

**What you must not do:** do not let nodes call each other directly (use the blackboard); do not let an agent grade its own work or decide its own routing; do not send an agent the full blackboard (use a formatter); do not add a second rewrite loop on top of a bounded one.

---

## 11. Glossary of Patterns and Terms

```
   TERM                     WHAT IT MEANS IN THIS SYSTEM
   ────                     ────────────────────────────
   Agent                    A language model given one focused role, a few tools, and a
                            strict output type. One per folder in src/agents/.

   Orchestration            The coordination layer (src/orchestration/) that decides what
                            runs when, runs independent work in parallel, moves data between
                            stages, and makes the final go/no-go decision.

   Node                     A plain Python function that is one step of the line. Most nodes
                            wrap one agent; two (PDF→text, render) are pure code.

   Graph / topology         The declarative shape of the line (graph.py): the nodes, their
                            order, what runs in parallel, and the one conditional branch.

   Blackboard / state       The shared typed object (state.py) that flows down the line. Each
                            node reads from it and writes only its own slot back.

   Contract                 A Pydantic model that defines the exact shape of a stage's output.
                            Every handoff between stages is a contract, not free text.

   Factory                  The function in an agent folder that builds (but does not run) the
                            agent. Separates configuration from execution.

   The seam                 The single adapter (crew_task_execution.py) that runs one CrewAI
                            agent + one task and returns one validated contract — the only
                            place LangGraph (macro) and CrewAI (micro) touch.

   Formatter                The per-stage filter (src/formatters/) that compresses the
                            blackboard into the minimal context an agent needs. Cost control.

   Fan-out / fan-in         Running independent stages in parallel (fan-out) and waiting for
                            several of them to finish before the next stage (fan-in).

   Bounded rewrite loop     write → code audits the typed output → at most ONE corrective
                            rewrite. Self-correction that cannot run away.

   The quality gate         The deterministic, code-owned pass/fail decision that gates PDF
                            rendering. The agent scores; code decides; the graph routes.

   "Agents reason,          The spine of the design: judgment belongs to the LLM; measurement
    code measures"          and every consequential decision belong to deterministic code.
```

---

*You are now equipped to read the code. Start at `runner.py` (the front door), then `graph.py` (the shape of the line), then `state.py` (the blackboard), then any single file in `nodes/` (they all share the same shape). When a node calls an agent, open that agent's folder and its architecture doc. The patterns in section 6 are the lens; the code is just their concrete spelling.*
