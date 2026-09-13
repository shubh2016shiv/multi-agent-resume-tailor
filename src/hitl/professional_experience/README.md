# Human-in-the-Loop, From First Principles

### A working guide to building HITL for any system — taught through one real implementation

---

## How to read this document

This is written to be read front to back, like a short book, rather than searched
like a reference. It assumes you know Python and roughly what an agent pipeline
does, and it assumes **nothing** about interrupts, checkpoints, or durable
workflows.

It has two jobs, and it does them together in every chapter:

1. **Teach the general pattern**, so you can build production human-in-the-loop
   for a refund system, a deploy pipeline, a content moderation queue, or
   anything else.
2. **Show one real implementation** — the clarification loop in
   `src/hitl/professional_experience/` — so the general pattern is never abstract.

Each component chapter follows the same rhythm:

```
   THE CONCEPT          what problem this part solves, universally
        |
        v
   THE DECISIONS        the choices you must make, and how to choose
        |
        v
   THIS PROJECT         real code from this repository, step by step
        |
        v
   ELSEWHERE            how the same part looks for a different problem
```

Every code reference points at real code in this repository — if you see a
function name, you can open that file and find it. Some snippets are lightly
condensed for reading (a logging call elided, a long identifier shortened); the
file path above each block is always the authority. Snippets attributed to
*Problem B* or to other frameworks are illustrative patterns, not code from this
tree.

Where this document says something is missing or imperfect, that is a statement
about the code as it stands today, not an aspiration.

### The controlling sentence

> A correct HITL system **detects** that a decision exceeds machine authority,
> **durably saves** the workflow, **exposes a structured request** to an
> authorized human, and **resumes the same workflow** only after a validated
> decision — **without duplicating protected side effects.**

Every chapter that follows is an elaboration of one clause in that sentence. If
you remember nothing else, remember it, and remember the six verbs:

> **STOP → FREEZE → ASK → STORE → SHOW → GO**

### What you will be able to do afterwards

- Explain mechanically what HITL is, and how it differs from a chatbot asking a
  follow-up question.
- Choose *where* in an architecture the human belongs, and defend the choice.
- Decide which of the four kinds of ask you are building, before writing any UI.
- Implement pause/resume on LangGraph without falling into node-replay bugs.
- Know which production controls your problem actually needs, and which are
  ceremony for your risk level.
- Review someone else's HITL design against six invariants and a test plan.

---

## Contents

**Part I — Foundations**
1. [Two problems, one anatomy](#1-two-problems-one-anatomy)
2. [The mental model: the night-shift pharmacy](#2-the-mental-model-the-night-shift-pharmacy)
3. [HITL is a control plane, not another agent](#3-hitl-is-a-control-plane-not-another-agent)
4. [The central illusion: waiting without waiting](#4-the-central-illusion-waiting-without-waiting)
5. [Where a human can enter the system](#5-where-a-human-can-enter-the-system)
6. [The four kinds of ask](#6-the-four-kinds-of-ask)

**Part II — The six components**

7. [Component 1 — Trigger: when must the machine stop?](#7-component-1--trigger-when-must-the-machine-stop)
8. [Component 2 — Pause: how does execution halt?](#8-component-2--pause-how-does-execution-halt)
9. [Component 3 — Contract: what exactly is being asked?](#9-component-3--contract-what-exactly-is-being-asked)
10. [Component 4 — Durable state: what must survive?](#10-component-4--durable-state-what-must-survive)
11. [Component 5 — Surface: who sees it, and how?](#11-component-5--surface-who-sees-it-and-how)
12. [Component 6 — Resume: how does the answer get back in?](#12-component-6--resume-how-does-the-answer-get-back-in)
13. [The whole lifecycle in one picture](#13-the-whole-lifecycle-in-one-picture)

**Part III — Production hardening**

14. [The approval state machine](#14-the-approval-state-machine)
15. [Idempotency: the most important reliability pattern](#15-idempotency-the-most-important-reliability-pattern)
16. [Concurrency: exactly one decision wins](#16-concurrency-exactly-one-decision-wins)
17. [Authorization: approve the action, not the person](#17-authorization-approve-the-action-not-the-person)
18. [Freshness: revalidate before the side effect](#18-freshness-revalidate-before-the-side-effect)
19. [Timeouts and escalation](#19-timeouts-and-escalation)
20. [Versioning the pause](#20-versioning-the-pause)
21. [Privacy, secrets, and what not to persist](#21-privacy-secrets-and-what-not-to-persist)
22. [Observability: age beats count](#22-observability-age-beats-count)
23. [This project's honest scorecard](#23-this-projects-honest-scorecard)

**Part IV — Building your own**

24. [Choosing a framework](#24-choosing-a-framework)
25. [The implementation path, in order](#25-the-implementation-path-in-order)
26. [Six anti-patterns](#26-six-anti-patterns)
27. [The test plan that proves it](#27-the-test-plan-that-proves-it)
28. [Design review invariants and checklist](#28-design-review-invariants-and-checklist)
29. [The 90-second explanation](#29-the-90-second-explanation)
30. [Mnemonic sheet](#30-mnemonic-sheet)

**Appendices** — [file map](#appendix-a--file-map) · [glossary](#appendix-b--glossary) · [references](#appendix-c--references)

---
---

# Part I — Foundations

## 1. Two problems, one anatomy

Learning HITL from a single example teaches you that example. So we will carry
two, deliberately chosen to sit at opposite ends of the design space. Everything
in this guide applies to both; almost every *decision* comes out differently.

### Problem A — the one this repository solves

A candidate uploads a resume and a job description. The pipeline rewrites his
bullet points to be sharper. One bullet says:

> *"Built and maintained microservices for the payments platform."*

The rewriting agent tightens the phrasing and then hits a wall no cleverness gets
past: the resume never says what that work **achieved**. No latency figure, no
uptime number, no scale. The sentence is true and it is thin, and the only honest
way to improve it is to learn a fact that exists in exactly one place — the
candidate's head.

Three options exist. Invent a number, which is fraud. Ship the thin bullet
silently, which wastes the candidate's best material. Or **stop and ask**.

### Problem B — the classic one, which we will keep comparing against

An e-commerce refund workflow. A supervisor agent delegates evidence gathering,
policy interpretation, and execution. Policy allows autonomous refunds up to $50.
Above that a manager must approve; above $1,000 a finance approver must; and
enterprise accounts always require a human regardless of amount.

| | Problem A (this repo) | Problem B (refunds) |
|---|---|---|
| What the human supplies | A missing fact | A decision on a proposed action |
| Is anything irreversible? | **No** | **Yes** — money moves |
| Cost of a wrong "no ask" | A weaker resume line | Fraud, or a policy breach |
| Cost of a wrong "ask" | Candidate annoyance | Reviewer fatigue, slow refunds |
| Who may answer? | Only the candidate | A specific authorized role |
| Does duplicate execution matter? | Not really | **Catastrophically** |

Hold that table. It is the single most useful thing in this guide, because it
determines which production controls you must build and which you may skip. A
system that treats Problem A with Problem B's machinery is over-engineered; a
system that treats Problem B with Problem A's machinery is dangerous.

---

## 2. The mental model: the night-shift pharmacy

Picture a 24-hour pharmacy at 2 a.m. A technician handles routine work, but a
controlled prescription requires a licensed pharmacist, and the pharmacist will
not be in until morning. The technician must not guess — and must not stand
frozen holding the prescription for six hours either.

What a competent technician does:

1. **Recognises** the prescription crosses an authority boundary.
2. **Stops** before the protected action occurs.
3. **Prepares** the exact context and decision the pharmacist will need.
4. **Puts** the work-in-progress in a durable, labelled holding place.
5. **Routes** the request to the right pharmacist.
6. **Continues** the same work exactly once when the answer arrives.

Substitute *agent* for technician, *reviewer* for pharmacist, *proposed action*
for prescription, and *durable checkpoint* for holding bin. That is the entire
mechanism, and those six steps are the six verbs:

> **STOP → FREEZE → ASK → STORE → SHOW → GO**

The pharmacy image is worth keeping because its failure modes are the software
failure modes, and we will name them again in [chapter 26](#26-six-anti-patterns):
losing the prescription because it was only on a sticky note; standing frozen for
six hours; handing over a signature line with no information; dispensing to the
wrong patient; dispensing twice; and rubber-stamping everything because the
technician escalates too much.

---

## 3. HITL is a control plane, not another agent

Any system doing real work has two layers, and confusing them is the most common
architectural error in this area.

```
     +------------------------------------------------------------------+
     |  CONTROL PLANE          "may this proceed, and on whose word?"    |
     |                                                                   |
     |   trigger policy   authorization   checkpointing   resume         |
     |   deadlines        escalation      audit           idempotency    |
     +------------------------------------------------------------------+
                                    |  governs
                                    v
     +------------------------------------------------------------------+
     |  DATA PLANE             "do the task"                             |
     |                                                                   |
     |   retrieve   reason   plan   rewrite   call tools   render        |
     +------------------------------------------------------------------+
```

| Layer | Responsibility | Examples |
|---|---|---|
| Data plane | Do the task | Research agent, planner, rewriter, tool executor |
| Control plane | Govern the task | Approval policy, authorization, checkpointing, resume, audit, timeout |

**HITL belongs entirely in the control plane.** The reason this matters is blunt:

> **A prompt is not an enforcement boundary.**

You may instruct a supervisor agent to "always ask before sending money." That
instruction can be bypassed by a different route through the graph, by a
delegated subagent that calls the tool directly, by prompt injection in retrieved
content, or by a refactor six months from now that nobody connects to the rule.
For anything irreversible, the gate must live in code at or immediately before
the capability itself — not in text that a model is asked to honour.

> **ARCHITECTURE RULE**
> If an action can cause irreversible or externally visible change, the last
> mandatory gate must be adjacent to the capability that performs that change.

This project is an interesting edge case: it has **no protected capability at
all.** Nothing it does is irreversible, so its gate can safely live in
orchestration rather than wrapped around a tool. Notice that this is a conclusion
derived from the risk profile, not a default.

---

## 4. The central illusion: waiting without waiting

Before any component, one idea has to land, because everything else depends on
it — and it is the thing most people get wrong when they first imagine how this
works.

The naive mental model: the pipeline reaches the point where it needs an answer
and *waits* — a thread parks, a loop spins, a coroutine sleeps — until the human
responds, then wakes and carries on.

**That is not what happens. There is no wait loop.**

Search this codebase for a loop that waits on the candidate and you will not find
one, because the pause is not a state the program is *in*. The pause is the
program **ending**.

```
  WALL CLOCK  ------------------------------------------------------------->

  Monday 10:00        Monday 10:01              (three days)        Thursday
       |                   |                                            |
  +----+-------------------+----+                              +--------+-----+
  |  PROCESS #1                 |                              |  PROCESS #2  |
  |  tailor_resume()            |                              |  resume_     |
  |                             |                              |  paused_run()|
  |  extract resume             |                              |              |
  |  analyze job                |                              |  read files  |
  |  rewrite bullets            |                              |  reopen db   |
  |  interrupt()  --> raise     |                              |  continue    |
  |  write files to disk        |                              |  finish      |
  |  PROCESS EXITS              |                              |  EXITS       |
  +-----------------------------+                              +--------------+
                 \                                            /
                  \      N O T H I N G   I S   R U N N I N G /
                   \     no thread. no loop. no timer.      /
                    \    no memory. no open socket.        /
                     v                                    v
              +--------------------------------------------------+
              |   paused_run_<id>/     <-- the only thing that    |
              |     checkpoints.sqlite3     exists during the     |
              |     clarifications_sheet.json    "wait"           |
              |     paused_run_manifest.json                      |
              +--------------------------------------------------+
```

If nothing waits, what makes the system *appear* to wait? Three things, none of
them running code:

1. **Durable state on disk** — the whole pipeline state in a SQLite file.
2. **Wall-clock time**, which requires no participant.
3. **A future invocation** that knows how to find and rehydrate that state.

> **THE REFRAME**
> A pause is not the graph waiting. It is the graph **exiting cleanly, having
> written down enough to be rebuilt.** The process that called `interrupt()` may
> be long dead by the time someone answers.

### Why `interrupt()` is a `raise`, not a `sleep`

The mechanism is a control-flow exception. The last statement of LangGraph's
`interrupt()` is:

```python
raise GraphInterrupt(
    (Interrupt.from_ns(value=value, ns=conf[CONFIG_KEY_CHECKPOINT_NS]),)
)
```

`GraphInterrupt` inherits `GraphBubbleUp`, which inherits `Exception`. That is the
whole trick. LangGraph catches the throw, persists the checkpoint, and lets
`pipeline.invoke()` return *normally* with a marker in the output. From the
caller's perspective the pipeline simply finished early with a note attached.

### An objection you should be raising right now

If nothing waits, you might reasonably object: *"but the web app is a running
server — doesn't a server sit in a loop, listening? Isn't **that** the thing
waiting for the candidate?"*

It's a fair suspicion, because this project does contain exactly one `while
True`, in `web_app/server.py`. So does it contradict everything just said? No —
and seeing precisely *why not* is what locks the concept in place. Here it is:

```python
async def _event_stream(events: Queue) -> AsyncIterator[str]:
    while True:
        try:
            event = await asyncio.to_thread(events.get, True, 15)
        except Empty:
            yield ": keep-alive\n\n"
            continue
```

It is easy to mistake this for the wait. It streams *progress events* to the
browser and exits on a terminal event. It knows nothing about clarifications.
When the candidate finally answers, that arrives as a brand-new HTTP request
starting a brand-new background task.

### Why not just block?

Every blocking alternative fails in production in the same three ways: it holds a
worker for days; it loses everything on crash, deploy, or reboot; and it cannot
survive being moved to another machine. The durable approach survives all of
that and costs nothing while idle.

This is not exotic. AWS Step Functions calls it `.waitForTaskToken`. Temporal
calls it durable timers and signals. AutoGen documents terminating the run and
saving team state for later resumption. In every case the defining property is
identical: **the workflow is not resident in memory while it waits.**

---

## 5. Where a human can enter the system

Chapters 1–4 showed you one real pipeline, node by node. This chapter zooms
back out to compare *that* pipeline's shape against other systems — so what
follows is **four generic categories a gate can fall into**, not four sequential
stages every system must have, and not a rival to the six-verb lifecycle
(STOP→FREEZE→ASK→STORE→SHOW→GO) from earlier. The six verbs describe *how* a
pause works, once you've decided to have one. This chapter is about *where in
the architecture* that decision gets made — a question you answer once, before
any of the six verbs apply.

A human can be inserted at four different places, and they are not
interchangeable. Choosing wrongly is how teams end up with a gate that feels
thorough and protects nothing.

```
   User request
        |
        v
   +----------------+
   | Supervisor     |  <-- (1) PLAN REVIEW
   | / Router       |      "is this plan acceptable?"
   +----------------+
        |
        v
   +----------------+
   | Worker agents  |  <-- (2) OUTPUT REVIEW
   | (rewrite, ...) |      "is this generated content acceptable?"
   +----------------+
        |
        v
   +----------------+
   | Tool boundary  |  <== (3) CAPABILITY GATE   *** the enforcement point ***
   +----------------+      "may this concrete action execute?"
        |
        v
   +----------------+
   | Publish/return |  <-- (4) RELEASE REVIEW
   +----------------+      "may this leave the system?"
```

| Entry point | Human decides | Best for | The risk | This repo's equivalent |
|---|---|---|---|---|
| (1) Plan / supervisor | Whether the plan is acceptable | High-impact plans, regulated workflows | Too coarse — a subagent may still call a risky tool later | *None.* There is no routing decision to review — every resume goes through the same fixed stages. |
| **(2) Agent output** | Whether generated content is acceptable | Editing, QA, expert validation | May be too late; earlier side effects already happened | **Here.** `optimize_experience` produces the rewritten bullets; `audit_experience_candidate_fact_gaps` (inside `clarifications.py`) reviews that output and decides whether a bullet is too thin; `await_candidate_clarifications` is where the pause actually sits. |
| (3) Tool / capability | Whether a concrete action may execute | Refunds, deletes, sends, deploys, writes | *(recommended enforcement point for anything irreversible)* | *None.* There is no tool call here that sends money, deletes data, or messages anyone — see chapter 1's comparison table. |
| (4) Final release | Whether the result may leave the system | Reports, customer comms | Does not protect internal mutations already made | *None.* `render_final_resume` runs unattended once the quality gate passes. |

Read that last column carefully: **three of the four rows are empty for this
project, and that is not a gap — it is the correct shape for a Supply ask with
no irreversible action.** A system only needs a gate at the rows where it
actually has something to protect at that point.

**Problem B (refunds)** fills in differently. Its mandatory gate sits at (3),
wrapped around `issue_refund`, precisely so no routing change or delegated
subagent can reach the money without passing through it. Plan review at (1) is
optional garnish there, not the enforcement point.

**Problem A (this repo)** sits at (2) — output review — and that is *correct here*
because there is no capability to protect. The bullet has already been rewritten;
nothing has been sent, charged, or deleted. The pause exists to improve content,
not to guard an action.

> **How to choose:** ask "what is the worst thing that happens if the gate is
> bypassed?" If the answer involves money, data loss, or an external message,
> your gate belongs at the capability. If the answer is "the output is worse,"
> output review is enough.

---

## 6. The four kinds of ask

Before designing a schema or a screen, decide which of these you are building.
Getting this wrong produces UIs that ask for the wrong shape of answer.

| Kind | The human does | Example | Response shape |
|---|---|---|---|
| **Approve** | Accepts or rejects an unchanged proposal | "Approve this $120 refund?" | enum: approve / reject |
| **Edit** | Modifies the proposal before it proceeds | "Refund $80 instead of $120" | decision + edited arguments |
| **Choose** | Selects among generated options | "Refund, replace, or credit?" | option id |
| **Supply** | Provides information the system cannot obtain | "What was the measurable outcome?" | validated free-form or typed data |

> **MNEMONIC — Approve · Edit · Choose · Supply.**

**This project implements only Supply**, and that single fact explains several of
its design decisions that would otherwise look like omissions:

- There is **no `allowed_decisions` enum**, because a Supply answer is inherently
  open text — there is no fixed set to choose from.
- There is **no agent recommendation** in the request, because there is no
  proposal to accept or reject. The candidate is the sole source; a suggested
  answer would be the invention we are trying to avoid.
- **Rejecting is not a concept.** Leaving a question blank simply means "no
  answer", and the pipeline proceeds with the thin-but-truthful bullet.

If you are building Approve or Edit, you need the opposite: a constrained
decision enum, the exact arguments that would execute, and a visible diff of what
the human changed.

---
---

# Part II — The six components

These are not an unordered checklist. They are the runtime lifecycle in order:
component 1 fires, causing 2, which requires 3 and 4, which feeds 5, which
triggers 6.

```
   1. STOP           2. FREEZE          3. ASK
   Decide a human    Halt execution     Define the shape of the
   is required       and unwind         request and its answer
   |                 |                  |
   +--------+--------+---------+--------+
            |                  |
            v                  v
   4. STORE                5. SHOW
   Persist everything      Put the request in front
   needed to rebuild       of an authorized human
            |                  |
            +--------+---------+
                     |
                     v
              6. GO
              Validate, inject, and continue
              exactly once
```

| # | Component | The question it answers | In this repo |
|---|---|---|---|
| 1 | Trigger / authority policy | When must the machine stop? | `clarifications.py` |
| 2 | Interrupt / pause | How does execution return control without blocking? | `orchestration/nodes/experience.py` |
| 3 | Question/answer contract | What is being asked, and what answers are valid? | `models.py` |
| 4 | Durable checkpoint | What must survive process death and deployment? | `orchestration/checkpointing.py` + `persistence.py` |
| 5 | Human surface and routing | Who sees the request, with what evidence? | `web_app/server.py` + `persistence.py` |
| 6 | Resume mechanism | How is a validated answer injected into the exact paused run? | `orchestration/runner.py` |

---

## 7. Component 1 — Trigger: when must the machine stop?

### The concept

A **trigger policy** is the rule deciding whether execution may continue
autonomously. It expresses the boundary of machine authority. The model may
*recommend*; the policy *decides*.

There are exactly three reasons to escalate, and naming yours is the first
design decision:

| Family | Meaning | Refund example | Resume example |
|---|---|---|---|
| **Costly** | High-impact, irreversible, expensive, security-sensitive | Refund > $50; delete account; production deploy | *(none — nothing here is irreversible)* |
| **Confusing** | Insufficient evidence, ambiguity, low confidence | Damage claim conflicts with carrier evidence | **A bullet is truthful but missing a candidate-owned fact** |
| **Compulsory** | Policy, regulation, or contract requires a human | Enterprise account always requires sign-off | *(none)* |

> **MNEMONIC — the three C's: Costly · Confusing · Compulsory.**

### The decisions you must make

**Decision 1 — Deterministic or model-judged?**

```
        Is the rule expressible as data you already have?
                        |
            +-----------+-----------+
            | yes                   | no
            v                       v
   DETERMINISTIC CODE          MODEL JUDGEMENT
   amount > 50                 "is this bullet thin?"
   tier == enterprise          "is this evidence contradictory?"
   region in restricted        |
            |                  v
            |          Is the decision Compulsory or Costly?
            |                  |
            |          +-------+--------+
            |          | yes            | no
            |          v                v
            |     NOT ACCEPTABLE     acceptable, but
            |     alone -- wrap it   measure its calibration
            |     in a deterministic
            |     backstop
            v
        preferred wherever possible
```

**A Compulsory gate must never depend on model confidence.** If a rule says a
human signature is mandatory, evaluate that rule in ordinary application code. A
model that confidently skips a required approval is still wrong, and "the model
decided not to ask" is not a defence anyone will accept.

For Confusing triggers, avoid raw "model confidence" unless you have calibrated
it against real outcomes. Better signals: missing required evidence, disagreement
between independent evaluators, policy-lookup failure, tool errors, or
out-of-distribution detection.

**Decision 2 — What bounds the volume?** A trigger that fires constantly produces
reviewer fatigue, which produces rubber-stamping, which means your gate now
provides *the appearance* of oversight with none of the substance. Volume needs a
deterministic bound even when the judgment is semantic.

### This project's implementation, step by step

Two words before any code. A **role**, in this codebase, just means one entry
in the candidate's work history — one job at one company. A candidate with
three past jobs has three roles, and this trigger runs once per role, not once
per candidate and not once per bullet.

And **"structured output"** — the mechanism the trigger is actually built on —
means forcing an LLM's normally free-text answer into a schema you define, so
the response comes back as a validated object instead of a paragraph you'd have
to parse. That is the single most load-bearing idea in this component, so it is
worth seeing the general-purpose function before the specific call that uses it:

```python
# src/tools/llm_gateway/structured_output.py

def request_structured_output(
    output_model: type[OutputModel],   # the Pydantic schema the reply must match
    system_prompt: str,                # the instructions -- here, the rubric
    user_content: str,                 # the data to judge -- here, the role's bullets
    temperature: float | None = None,
) -> OutputModel:                      # a validated INSTANCE of output_model, not text
    ...
```

Four arguments in, one validated object out. Keep that shape in mind — it is
exactly what the trigger calls.

**Step 1 — one semantic call per role, at temperature zero.**

```python
# src/hitl/professional_experience/clarifications.py

def audit_experience_candidate_fact_gaps(
    source_experience: Experience,
    shipped_bullets: list[str],
    rewritten_bullets: list[ExperienceBulletRewrite],
) -> ExperienceBulletFactGapReview:
    review_input = _build_fact_gap_review_input(
        source_experience, shipped_bullets, rewritten_bullets
    )
    return request_structured_output(
        ExperienceBulletFactGapReview,          # <- output_model:  the schema
        EXPERIENCE_CANDIDATE_FACT_GAP_RUBRIC,   # <- system_prompt: the rubric text
        review_input,                            # <- user_content: this role's bullets
        temperature=0.0,
    )
```

Reading it against the generic signature above: this call tells the gateway
*"send the model this role's bullets (`review_input`), instructed by this rubric
(`EXPERIENCE_CANDIDATE_FACT_GAP_RUBRIC`), and don't hand me back free text — hand
me back a validated `ExperienceBulletFactGapReview`."* You will meet that return
type's actual shape in Step 2; for now, treat it as "whatever schema this
function promises to return, `request_structured_output` guarantees you get."

Two pieces still need naming:

- **`review_input`** is not the raw resume. It's built by
  `_build_fact_gap_review_input` (shown later in this chapter), which renders the
  role's job title, description, and every shipped bullet into one plain-text
  block — the actual words the model reads.
- **`EXPERIENCE_CANDIDATE_FACT_GAP_RUBRIC`** is not inline prompt text either. It
  is loaded at import time from a file: `src/config/tool_prompts/hitl/experience_candidate_fact_gap.md`.
  That separation is deliberate — the decision logic is reviewable prose under
  version control, editable by anyone who can read English, rather than a string
  buried inside Python.

`temperature=0.0` is the other production-relevant choice here: the same bullet
must not flip between "fine" and "needs a question" across identical runs. A
temperature above zero would let the model's phrasing (and therefore its
judgment) vary run to run on identical input — intolerable for a trigger, where
a candidate answering the same resume twice should get the same verdict both
times.

**Step 2 — the trigger's output shape is the design.** One finding per bullet:

```json
{
  "bullet_id": "acme-corp::backend-engineer::2023-01-01::bullet::2",
  "current_bullet": "Built and maintained microservices for the payments platform.",
  "gap": {
    "gap_category": "result",
    "missing_fact_summary": "No measurable outcome of the microservices work is stated.",
    "why_flagged": "The LLM cannot invent a metric the resume never gave.",
    "question": "What was the measurable outcome of the microservices work?"
  }
}
```

Note where the optionality lives. There is no `requires_candidate_input` boolean.
`gap` itself is nullable: **null means ship as written; present means stop.**

This is not cosmetic. The earlier shape had a boolean *plus* four individually
optional sibling fields, which permitted a nonsensical state — a finding claiming
input is needed while carrying no question — and so the joining code had to
defend against it:

```python
# the OLD shape, now deleted
if bullet_rewrite is None or finding.gap_category is None or not question:
    logger.warning("experience_clarification_finding_dropped", ...)
    continue
```

Read what that does in production: the model says a bullet needs a question, the
question is missing, and the system **silently drops it**. The candidate is never
asked. Nothing surfaces but a log line nobody reads.

```
   OLD SHAPE                              NEW SHAPE
   ---------                              ---------
   requires_candidate_input: bool         gap: CandidateFactGap | None
   gap_category:    X | None                        |
   missing_fact:    str  ""                         +-- either absent entirely
   why_flagged:     str  ""                         |
   question:        str | None                      +-- or complete, all 4 fields
                                                        (Pydantic enforces it)
   5 fields that can disagree
   -> runtime checks                      1 field that cannot
   -> silent drops                        -> no checks needed, nothing to drop
```

> **THE PRINCIPLE**
> Prefer invariants the type system enforces over invariants your code
> re-checks. A check can be forgotten at a new call site. A schema cannot.

**Step 3 — the join is pure code**, and only one failure remains possible:

```python
def clarifications_from_findings(experience, shipped_bullets, rewritten_bullets, findings):
    rewrite_by_bullet_id = {r.bullet_id: r for r in rewritten_bullets}
    clarifications = []
    for finding in findings:
        if finding.gap is None:
            continue                                  # ship as written
        bullet_rewrite = rewrite_by_bullet_id.get(finding.bullet_id)
        if bullet_rewrite is None:                    # hallucinated id
            logger.warning("experience_clarification_finding_dropped",
                           reason="unknown_bullet_id", bullet_id=finding.bullet_id)
            continue
        clarifications.append(
            ExperienceBulletClarification(
                **finding.gap.model_dump(),           # no copy-rename: same fields
                bullet_id=finding.bullet_id,
                company_name=experience.company_name,
                job_title=experience.job_title,
                start_date=experience.start_date.isoformat(),
                bullet=_shipped_bullet_text(bullet_rewrite, shipped_bullets),
            )
        )
    return clarifications
```

**Step 4 — the deterministic bound on volume.** The semantic judgment is
per-role and roles run in parallel, so the cap is applied after the merge:

```python
# src/orchestration/nodes/experience.py

def _cap_clarifications(clarifications, run_id):
    limit = get_config().workflow.max_clarifications_per_run   # default 12
    if len(clarifications) <= limit:
        return clarifications
    logger.warning("experience_clarifications_capped",
                   run_id=run_id, requested=len(clarifications), kept=limit)
    return clarifications[:limit]
```

Roles are processed in resume order, so truncating keeps whole recent roles rather
than a scattering across all of them, and what was dropped is logged rather than
silently discarded.

**Step 5 — degrade, don't destroy.** Asking is an *enhancement* on top of an
already-truthful rewrite, not a safety gate:

```python
try:
    fact_gap_review = audit_experience_candidate_fact_gaps(...)
except Exception:
    logger.exception("experience_fact_gap_review_failed", run_id=run_id, ...)
    return []          # ship the good rewrite; ask nothing
```

That decision follows directly from the risk profile. In Problem B the equivalent
failure must **fail closed** — if you cannot evaluate the refund policy, you
absolutely do not refund.

### The honest caveat

Because this trigger is entirely LLM-judged, it inherits LLM judgment variance.
There is no deterministic backstop forcing a question for, say, every bullet
containing no numbers. A rule like *"no digits therefore must ask"* was
considered and rejected: it would suppress legitimate questions about artifacts
and scope, and the "risk" being managed is a thin résumé line, not a privileged
operation. **Volume** is bounded deterministically; **judgment** is not. For a
Costly or Compulsory trigger that trade would be unacceptable.

---

## 8. Component 2 — Pause: how does execution halt?

### The concept

A durable pause means execution reaches a safe boundary, state is persisted,
control returns to the application, and **no compute stays alive**. Chapter 4
established the mechanics; this chapter is about placing the boundary correctly.

### The decisions you must make

**Decision 1 — where in the graph does the interrupt node sit?** The rule that
matters:

> Put the interrupt **between** the work that produced the proposal and the work
> that acts on it — and keep the protected side effect in a **separate later
> node**.

```
   BAD                                 GOOD
   ---                                 ----
   +---------------------------+       +---------------+   +---------------+
   | approval_node             |       | approval_node |-->| execute_node  |
   |   send_email(...)  <-- replays!|   |  interrupt()  |   |  send_email() |
   |   interrupt(...)          |       +---------------+   +---------------+
   |   charge_card(...)        |         no side effects     side effect,
   +---------------------------+         here                idempotent
```

**Decision 2 — is any code above `interrupt()` replay-safe?** This is
[the double-execution trap](#12-component-6--resume-how-does-the-answer-get-back-in),
covered in full in component 6. Preview: on resume, LangGraph re-runs the
interrupted node **from its top**, so everything above the `interrupt()` line
executes a second time.

**Decision 3 — one interrupt per node invocation.** Do not build a validation
loop containing `interrupt()`:

```python
# DO NOT DO THIS
def collect_amount(state):
    while True:                       # <-- each node restart replays earlier
        answer = interrupt("Enter amount")   #     iterations of this loop
        if valid(answer):
            return {"amount": answer}
```

Instead, store the next question in state and route back to the node with a
conditional edge:

```python
def collect_amount(state: State):
    question = state.get("pending_question") or "Enter approved refund amount"
    answer = interrupt(question)                     # exactly once per invocation
    if isinstance(answer, (int, float)) and 0 <= answer <= state["requested_amount"]:
        return {"approved_amount": float(answer), "pending_question": None}
    return {"pending_question": f"Invalid amount {answer!r}. Enter 0..{state['requested_amount']}"}
# a conditional edge routes back here while approved_amount is missing
```

Use graph routing for re-prompting. This is the second sense in which HITL "has
no while loop" — not only does nothing wait, but you must not *write* a loop
around the interrupt either.

**Decision 4 — never swallow the interrupt.** `interrupt()` works by raising, so
a broad `try/except Exception` around it will silently convert a pause into a
caught error. Catch narrowly, or not at all, in any function on the interrupt path.

### This project's implementation, step by step

**Step 1 — the pause is its own node**, placed after all of stage three's
parallel work and before ATS assembly, so a paused run never leaves
half-assembled output behind:

```python
# src/orchestration/graph.py
graph.add_edge("write_professional_summary", "await_candidate_clarifications")
graph.add_edge("optimize_experience",        "await_candidate_clarifications")
graph.add_edge("optimize_skills",            "await_candidate_clarifications")

graph.add_conditional_edges(
    "await_candidate_clarifications",
    _route_after_candidate_clarifications,
    {"rewrite_experience": "optimize_experience",   # resumed: rewrite with answers
     "assemble_resume":    "assemble_ats_resume"},  # normal: carry on
)
```

**Step 2 — the node itself, in three cases:**

```python
# src/orchestration/nodes/experience.py

def await_candidate_clarifications(state: ResumeEnhancementPipelineState):
    clarifications = state.get("experience_clarifications") or []
    clarification_answers = state.get("clarification_answers") or []
    if not clarifications:
        return {}                                   # case 1: nothing to ask
    if clarification_answers:
        logger.info("candidate_clarifications_received", ...)
        return {}                                   # case 2: resumed -- see ch.12
    logger.info("candidate_clarifications_requested", ...)
    interrupt({                                     # case 3: pause
        "type": "candidate_clarifications_required",
        "questions": [c.model_dump(mode="json") for c in clarifications],
    })
    return {}
```

Note there is **no code above `interrupt()` with side effects** — only two state
reads and a log line. That is what makes it replay-safe.

**Step 3 — what actually unwinds:**

```
   pipeline.invoke(...)                          <-- called by tailor_resume()
     |
     +-> LangGraph executes nodes
           |
           +-> await_candidate_clarifications(state)
                 |
                 +-> interrupt({...})
                       |
                       +-> raise GraphInterrupt   ......... the throw
                 |
           <-----+   caught by LangGraph's executor
           |
           +-> checkpoint written to SQLite       ......... state made durable
           |
     <-----+
     |
   returns {"__interrupt__": (...), ...}          ......... normal return
     |
   the runner writes the paused-run directory, returns a
   result, and the process is free to exit.
```

That ordering carries a guarantee worth naming: LangGraph writes the checkpoint
**as part of executing `interrupt()`**, before control returns. Only afterwards
does the runner write the human-facing files. A kill between those two steps
leaves the checkpoint correct — you lose convenience files, not state.

**Step 4 — detection is trivial by design:**

```python
def _pipeline_interrupted(output: dict) -> bool:
    return "__interrupt__" in output
```

### The seven interrupt rules to memorise

| # | Rule | Why |
|---|---|---|
| 1 | Resume with the same `thread_id` | The checkpointer uses it to load the paused thread |
| 2 | The node restarts from its beginning | Pre-interrupt code re-executes |
| 3 | Side effects before `interrupt()` must be idempotent | Otherwise replay duplicates them |
| 4 | Prefer side effects in *later* nodes | Separates approval from execution |
| 5 | Never catch `interrupt` with a broad `try/except` | It works via a special exception |
| 6 | Keep multiple interrupts in a stable order | Resume matching within a node depends on it |
| 7 | Use simple JSON-serializable payloads | Interrupt values are persisted |

### A caveat for multi-agent systems: subgraphs replay too

If a parent node invokes a subgraph as a function and the subgraph interrupts,
resuming can restart **both** the parent node containing the invocation *and* the
interrupted node inside the subgraph. In multi-agent designs where each
specialist is packaged as a subgraph, this is easy to miss.

> **DESIGN CONSEQUENCE**
> Do not put irreversible side effects in a parent node immediately before
> calling a subagent that may interrupt. That parent code can replay.

The robust structure is therefore always: **pure planning/routing nodes →
interrupt node → side-effect node**, with external mutations isolated in small
nodes carrying stable idempotency keys.

### Elsewhere: gating the capability instead

For Problem B, put the interrupt inside the tool so the policy travels with the
capability and every caller inherits it:

```python
@tool
def issue_refund(order_id: str, amount: float) -> str:
    decision = interrupt({
        "type": "tool_approval",
        "tool": "issue_refund",
        "args": {"order_id": order_id, "amount": amount},
        "allowed": ["approve", "reject", "edit"],
    })
    if decision["action"] == "reject":
        return "Refund rejected by reviewer"
    final_amount = decision.get("amount", amount)
    return payment_api.refund_once(                       # mutation AFTER the gate
        order_id=order_id, amount=final_amount,
        idempotency_key=f"refund:{order_id}:{final_amount}",
    )
```

Use tool-level gating when the risk is intrinsic to the capability. Use a graph
node when the decision needs broader workflow context. Many production systems do
both: classify policy in the graph, enforce it in the tool wrapper.

---

## 9. Component 3 — Contract: what exactly is being asked?

### The concept

An approval request is a **business object, not a prompt string**. It must be
independently renderable in a UI, auditable months later, and validatable before
a decision is applied.

### The decisions you must make

Here is the full field surface of a production contract. Not every problem needs
every field — the right-hand columns show why this project omits several.

| Field | Why it exists | Problem B | This repo |
|---|---|---|---|
| `approval_id` | Binds a response to one pending decision | Required | Uses `bullet_id` — questions are content, not decisions |
| `thread_id` / `workflow_id` | Binds the request to one workflow instance | Required | In the manifest, not the payload |
| `proposed_args` | Shows exactly what would execute | Required | **N/A** — nothing executes |
| `evidence_refs` | Lets the reviewer inspect facts without embedding documents | Required | The bullet text itself |
| `allowed_decisions` | Constrains the interaction to meaningful choices | Required | **N/A** — Supply is open text |
| `agent_recommendation` | Speeds review | Useful | **Deliberately absent** — a suggestion would be the invention we avoid |
| `expires_at` | Prevents stale decisions living forever | Required | **Present**, on the run |
| `state_version` | Optimistic concurrency, stale-answer rejection | Required | **Absent** — see ch.16 |
| `workflow_version` | Survives deployments during long pauses | Required | **Absent** — see ch.20 |

A realistic Problem B request:

```json
{
  "approval_id": "apr_72d9",
  "workflow_id": "refund_48812",
  "thread_id": "refund_48812",
  "type": "APPROVE_OR_EDIT",
  "action": "issue_refund",
  "resource": {"order_id": "48812", "customer_id": "cust_5512"},
  "proposed_args": {"amount": 120.00, "currency": "USD"},
  "reason": "Item arrived damaged",
  "evidence_refs": ["delivery_photo_1", "carrier_event_381"],
  "policy": {"policy_id": "returns-v4", "rule": "refund_over_50"},
  "agent_recommendation": "approve",
  "allowed_decisions": ["approve", "reject", "edit"],
  "expires_at": "2026-09-13T02:14:00Z",
  "workflow_version": "refund-mas-3.2",
  "state_version": 17
}
```

The **response** must be structured too: the decision, any edited arguments, a
reviewer note, the reviewer identity taken from the authenticated session (never
from the request body), and the version of the request the reviewer actually saw.

### This project's implementation, step by step

**Step 1 — one class, both directions.** The defining property of this contract
is that it is the *same class* on both sides of the pause:

```
   BEFORE THE PAUSE                       AFTER THE RESUME
   (written by the pipeline)              (read back by the pipeline)

   ExperienceBulletClarification          ExperienceBulletClarification
   +-----------------------------+        +------------------------------+
   | bullet_id   "acme::...::2"  |        | bullet_id   "acme::...::2"   |
   | company_name "Acme Corp"    |        | company_name "Acme Corp"     |
   | job_title   "Backend Eng."  |        | job_title   "Backend Eng."   |
   | start_date  "2023-01-01"    |  ===>  | start_date  "2023-01-01"     |
   | bullet      "Built and ..." |  disk  | bullet      "Built and ..."  |
   | question    "What was the..."|       | question    "What was the..."|
   | answer      ""              |        | answer      "Cut p95 by 40%" |
   | answered_at  null           |        | answered_at "2026-09-10T..." |
   | answered_by  null           |        | answered_by "web-ui (unauth)"|
   +-----------------------------+        +------------------------------+

        one class -- one file -- nothing to reconcile on the way back
```

Most systems drift into a `QuestionDTO` going out and an `AnswerDTO` coming back,
with glue matching them. Every such pairing is an opportunity to disagree. One
class removes the reconciliation step because there is nothing to reconcile.

**Step 2 — the shared base makes drift impossible:**

```python
# src/hitl/professional_experience/models.py

class CandidateFactGap(BaseModel):
    """What one bullet is missing, plus the question that asks for it.

    Every field is required: a gap that cannot name its category or phrase its
    question is not a gap. Optionality lives at the *nesting* point, never here.
    """
    gap_category: ExperienceBulletMissingFactCategory
    missing_fact_summary: str
    why_flagged: str
    question: str


class ExperienceBulletFactGapFinding(BaseModel):
    """The LLM's verdict on one bullet. Null gap == ship as written."""
    bullet_id: str
    current_bullet: str
    gap: CandidateFactGap | None = None


class ExperienceBulletClarification(CandidateFactGap):   # <-- inherits the 4 fields
    """The persisted question/answer the candidate sees."""
    bullet_id: str
    company_name: str
    job_title: str
    start_date: str = ""
    bullet: str
    answer: str = ""
    answered_at: datetime | None = None
    answered_by: str | None = None

    @property
    def is_answered(self) -> bool:
        return bool(self.answer.strip())
```

Because `ExperienceBulletClarification` **inherits** `CandidateFactGap`, the four
content fields are literally the same fields — no code copies them across, so
they cannot be renamed on one side only.

**Step 3 — stable identity that survives the gap.** Four fields exist purely so an
answer can find its way home. Their necessity is obvious once you picture a real
resume: several roles, several thin bullets each, all paused together.

```python
def build_experience_bullet_id(experience: Experience, bullet_index: int) -> str:
    role_id = experience.experience_id or (
        f"{experience.company_name.strip().lower()}::"
        f"{experience.job_title.strip().lower()}::"
        f"{experience.start_date.isoformat()}"
    )
    return f"{role_id}::bullet::{bullet_index}"
```

Note what it is **not**: a bare list index. An index would silently point at a
different bullet if roles or bullets shifted between pause and resume — and a
re-run does shift them. `start_date` is the tiebreaker for the genuinely awkward
case of the same title at the same company twice.

> **THE GENERAL RULE**
> Never correlate a human decision by list position. Bind it to a durable
> business identifier. This applies equally to parallel approvals, where two
> branches interrupt simultaneously and responses must map back by interrupt id
> and business id — never by arrival order.

---

## 10. Component 4 — Durable state: what must survive?

### The concept

The checkpoint is the durable snapshot that lets *another process* continue. Two
distinct things must survive, and keeping them separate in your head prevents a
lot of confusion:

1. **Execution state** — everything the workflow needs to continue. Framework-owned.
2. **Control-plane state** — the approval record, deadline, audit, authorization.
   Application-owned.

### The decisions you must make

**Decision 1 — what to persist.**

| Persist | Why |
|---|---|
| Business state | Order, amount, evidence refs, selected plan, risk classification |
| Conversation / messages | Model-visible interaction needed for coherent continuation |
| Tool and subagent results | Avoid recomputing expensive or external work |
| Routing state | Which node executes next |
| Pending request metadata | What external input is currently required |
| Versions | Workflow, policy, state, schema compatibility |
| Audit correlation IDs | Trace and approval linkage across systems |

> **DO NOT** make resumability depend on a model's hidden reasoning. Persist
> application-visible facts. If your run can only continue because some
> chain-of-thought happened to be in context, you do not have a resumable
> workflow.

**Decision 2 — which checkpointer.** In-memory savers are lost on restart and are
for local learning only. Production LangGraph deployments use `PostgresSaver` /
`AsyncPostgresSaver`; SQLite is reasonable for single-node or embedded use. The
Postgres checkpointer requires a one-time `setup()` to create its tables.

**Decision 3 — keep your approval ledger separate from framework tables.** Do not
treat the checkpointer's schema as your domain API. Your application should own
approval IDs, authorization rules, deadlines, policy snapshots, audit events, and
human-readable context:

```sql
CREATE TABLE approval_request (
    approval_id      uuid PRIMARY KEY,
    workflow_id      text NOT NULL,
    thread_id        text NOT NULL,
    interrupt_id     text,
    action_type      text NOT NULL,
    action_payload   jsonb NOT NULL,
    policy_snapshot  jsonb NOT NULL,
    status           text NOT NULL,   -- PENDING/APPROVED/REJECTED/EXPIRED/CANCELLED
    state_version    bigint NOT NULL,
    workflow_version text NOT NULL,
    expires_at       timestamptz NOT NULL,
    created_at       timestamptz NOT NULL DEFAULT now(),
    decided_at       timestamptz,
    decided_by       text,
    decision_payload jsonb
);

CREATE UNIQUE INDEX one_open_action_per_version
ON approval_request(workflow_id, action_type, state_version)
WHERE status = 'PENDING';
```

That partial unique index is doing real work — it makes "two open approvals for
the same action at the same version" impossible at the database level.

### This project's implementation, step by step

This project's control plane is a **directory on disk** rather than a table. That
is a legitimate choice for a single-user local tool, and an inadequate one for a
multi-tenant service — the shape is what matters, not the storage engine.

**Step 1 — the layout, defined exactly once:**

```
tailored_resumes/SHUBHAM_SINGH/Senior_Backend_Engineer/
|
+-- paused_run_7c606d5e18db4db3.../
    |
    +-- checkpoints.sqlite3         <-- ENTIRE pipeline state (framework-owned)
    |                                   opaque msgpack, keyed by thread_id
    |
    +-- clarifications_sheet.json   <-- the questions, human-editable
    |                                   MUTABLE: the candidate writes answers here
    |
    +-- answers_audit.jsonl         <-- every answer ever accepted
    |                                   APPEND-ONLY: never rewritten
    |
    +-- paused_run_manifest.json    <-- run_id, resume_path, jd_path,
                                        paused_at, expires_at
```

```python
# src/hitl/professional_experience/persistence.py

@dataclass(frozen=True)
class PausedRunLayout:
    """The file layout of one paused-run directory.

    Names live here once so a resume never has to guess the directory's shape,
    and so no caller can drift from it by hardcoding a literal.
    """
    root: Path

    @classmethod
    def at(cls, paused_run_path: str | Path) -> "PausedRunLayout":
        return cls(root=Path(paused_run_path))

    @property
    def sheet(self) -> Path:         return self.root / SHEET_FILENAME
    @property
    def manifest(self) -> Path:      return self.root / MANIFEST_FILENAME
    @property
    def audit_log(self) -> Path:     return self.root / AUDIT_FILENAME
    @property
    def checkpoint_db(self) -> Path: return self.root / CHECKPOINT_DB_FILENAME
```

That single-source rule is load-bearing; [chapter 11](#11-component-5--surface-who-sees-it-and-how)
shows what happened the one time it was broken.

**Step 2 — the manifest is the control record:**

```python
class ExperienceClarificationPausedRunManifest(BaseModel):
    run_id: str            # doubles as the LangGraph thread_id on resume
    resume_path: str
    jd_path: str
    paused_at: datetime
    expires_at: datetime

    @property
    def is_expired(self) -> bool:
        """Computed, never stored: a stored status can disagree with the clock."""
        return datetime.now(UTC) >= self.expires_at
```

**Step 3 — the checkpointer, with a security control.**

```python
# src/orchestration/checkpointing.py

def open_checkpoint_database(db_path: Path) -> SqliteSaver:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(db_path), check_same_thread=False)
    serde = JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_ALLOWED_MSGPACK_MODULES)
    return SqliteSaver(connection, serde=serde)
```

The serializer refuses to reconstruct a custom type on load unless explicitly
allowed. This blocks a tampered checkpoint file from instantiating arbitrary
Python classes — the class of vulnerability tracked as CVE-2026-28277. Passing an
explicit list *is* strict enforcement: anything unlisted is rejected regardless
of environment variables.

Keeping that list correct by hand would be a tax, so
`tests/unit/orchestration/test_checkpoint_allowlist.py` walks the real compiled
graph with LangGraph's own schema walker and fails the moment a new model becomes
reachable from state without being listed. The private-API dependency that walker
needs is confined to the test — a private API on the hot path of every production
run would be a poor trade.

**Step 4 — why the checkpointer does not live in this package.** The allowlist
names types from `data_models`, `agents/*`, and `tools/contracts`: it describes
the **whole pipeline's** state, not the human loop. And every run opens a
checkpointer whether or not it ever asks anything. Both lived here only because
this feature needed them first — history, not ownership. Moving them beside the
graph they serialize left this package with **no LangGraph import at all**: pure
domain plus file I/O.

**Step 5 — the file's lifecycle:**

```
   fresh run starts
   tailor_resume() opens .../checkpoints/<run_id>.sqlite3
          |
          +-- run PAUSES ----> archive_checkpoint_database() moves it into
          |                     paused_run_<id>/checkpoints.sqlite3   (kept)
          +-- run COMPLETES -> deleted (nothing left to resume)
          +-- run FAILS -----> deleted

   resumed run reopens paused_run_<id>/checkpoints.sqlite3
          +-- COMPLETES -----> deleted
          +-- PAUSES AGAIN --> kept, same directory reused
```

A run that pauses twice reuses one directory, so the candidate always has exactly
one folder to work in.

---

## 11. Component 5 — Surface: who sees it, and how?

### The concept

A review screen exists to **optimise decision quality**, not to expose an Approve
button. A reviewer who cannot tell why something was escalated will either
rubber-stamp it or block on it — both are failures.

### The decisions you must make

The reviewer should see the proposed action, why it was escalated, the relevant
evidence, the policy rationale, the agent's recommendation where one exists, and
the consequences of each choice. Beyond content:

- **Push into a surface reviewers already monitor**, or provide strong
  notifications. A queue nobody opens is a queue that expires.
- **Structured buttons for structured decisions**; free text supplements, never
  replaces, a constrained choice.
- **Show the deadline and the escalation path.**
- **Show a diff** for Edit requests — changed fields, clearly marked.
- **Do not expose hidden chain-of-thought.** Show evidence, rationale, policy
  citations, tool arguments, and uncertainty signals that are safe and useful.
- **Minimise personal and sensitive data** to what this reviewer actually needs.

> **HUMAN FACTORS RULE**
> If 99% of approvals are accepted without modification, your gate may be too
> noisy rather than well-calibrated. Monitor the approve-without-edit rate and
> involve real operators in testing.

### This project's implementation, step by step

There are two surfaces and both work.

**The CLI flow:** the candidate opens `clarifications_sheet.json` in an editor,
types answers into `answer` fields, and runs `--resume-from <paused_run_dir>`.
This is why the sheet is readable JSON carrying its own `_instructions` rather
than a bare machine array.

**The web flow:** questions render in the browser, answers post to
`/api/runs/{run_id}/resume`, and the resume runs as a background task.

**Step 1 — one module owns the format.** Everything else asks it:

```python
def record_clarification_answers(
    layout: PausedRunLayout,
    answers: Mapping[str, str],          # keyed by bullet_id, never by position
    *,
    answered_by: str,
) -> list[ExperienceBulletClarification]:
    clarifications = read_clarification_sheet(layout)
    known = {c.bullet_id for c in clarifications}
    unknown = sorted(set(answers) - known)
    if unknown:
        raise ValueError(f"No such question(s) in this paused run: {unknown}.")

    recorded_at = datetime.now(UTC)
    updated, new_records = [], []
    for c in clarifications:
        answer = (answers.get(c.bullet_id) or "").strip()
        if not answer:
            updated.append(c)                      # blank != answered
            continue
        updated.append(c.model_copy(update={
            "answer": answer, "answered_at": recorded_at, "answered_by": answered_by,
        }))
        new_records.append(ClarificationAnswerRecord(
            bullet_id=c.bullet_id, answer=answer,
            answered_by=answered_by, answered_at=recorded_at,
        ))

    if not any(c.is_answered for c in updated):
        raise ValueError("Answer at least one clarification before continuing.")

    # Audit first: the record of what was submitted must survive a failure to
    # persist the working copy, never the other way round.
    append_answer_records(layout, new_records)
    write_clarification_sheet(layout, updated)
    return updated
```

Three production behaviours are visible there: **previously answered questions are
preserved** (a candidate may answer across several sittings); **blank answers are
not answers**; and **the audit is written before the mutable copy**.

**Step 2 — the web layer knows nothing about the format:**

```python
# web_app/server.py
def _save_clarification_answers(paused_path: str, answers: list[dict[str, Any]]) -> None:
    submitted = {
        str(item["bullet_id"]): str(item.get("answer", ""))
        for item in answers if item.get("bullet_id")
    }
    if not submitted:
        raise ValueError("Answer at least one clarification before continuing.")
    record_clarification_answers(
        PausedRunLayout.at(paused_path), submitted,
        answered_by="web-ui (unauthenticated)",
    )
```

### The bug that lived here, and why it generalises

For a period the web flow was completely broken, and the failure is worth
studying because the lesson transfers everywhere.

`persistence.py` wrote the sheet as an object with two keys. `web_app/server.py`
read it back — with its own `json.loads` — as if it were a list.

```
        THE WRITER                              THE READER
        persistence.py                          web_app/server.py
            |                                        |
            v                                        v
   {                                     questions = json.loads(path.read_text())
     "_instructions": "Answer the...",   questions[0]["answer"] = "..."
     "clarifications": [ {...}, {...} ]                ^
   }                                                   |
            ^                                          |
     an OBJECT with 2 keys                  code expecting a LIST
            |                                          |
            +------------------ vs --------------------+
                                |
                                v
                     len(questions) == 2   (the two KEYS!)
                     questions[0]  ------> KeyError: 0
                                |
                                v
                  not caught by the endpoint's
                  except (ValueError, JSONDecodeError)
                                |
                                v
                        HTTP 500, every time
```

Every browser submission returned a 500. The CLI path worked perfectly, because
it went through the module that knew the real shape. That asymmetry is why it
survived: the surface tested by hand was fine; the one that wasn't, wasn't.

> **The lesson is not "someone made a typo."** The sheet's format was known in
> *two places* with nothing forcing agreement. Two copies of one fact will
> diverge; it is only a question of when. The fix was structural — one module
> owns the format and the other never opens the file — so there is no longer a
> second copy to disagree.

**Addressing by identity, not position.** The submission is keyed by `bullet_id`
rather than index. The old positional scheme coupled the browser's render order
to the file's write order with nothing keeping them in step — the same family of
assumption that produced the bug above.

### What is not checked here

`answered_by` records `"web-ui (unauthenticated)"`, and that string is doing
something important: telling the truth. There is no authentication on that
endpoint. Anyone with the `run_id` can answer; anyone with filesystem access can
edit the sheet. Recording the *surface* rather than inventing an identity keeps
the audit honest and marks exactly where a real identity plugs in. **It is a seam,
not a control.** See [chapter 17](#17-authorization-approve-the-action-not-the-person).

---

## 12. Component 6 — Resume: how does the answer get back in?

### The concept

Resume means: validate the answer, load the exact paused workflow, inject the
value, and continue — **exactly once**.

That last phrase is where most systems fail, because "exactly once" is not
something distributed systems give you for free.

### The at-most-once illusion

```
   WHAT YOU HOPE FOR              WHAT ACTUALLY HAPPENS
   -----------------              ---------------------
   human clicks Approve           human clicks Approve
        |                              |
        v                         (double-click / retry / redelivery /
   resume runs once                worker restart / browser refresh)
        |                              |
        v                              v
   refund issued once             resume job delivered 2-3 times
                                       |
                                       v
                                  refund issued 2-3 times  <-- unless you designed
                                                                for this
```

HTTP clients retry. Queues redeliver — Azure Service Bus, for instance, uses
at-least-once delivery with Peek Lock and explicitly recommends idempotent
consumers. Workers restart mid-job. Users double-click.

> **Do not design HITL around the assumption that a resume happens exactly once.
> Design the protected side effect so duplicates are harmless or rejected.**

### This project's implementation, step by step

**Step 1 — refuse cheaply, before opening anything:**

```python
# src/orchestration/runner.py

def resume_paused_run(paused_run_path: str, progress_callback=None) -> OrchestrationResult:
    layout, manifest = load_paused_run(paused_run_path)

    if manifest.is_expired:                                    # (a) deadline
        raise ValueError(
            f"This paused run expired on {manifest.expires_at.isoformat()} and can no "
            "longer be resumed. Start a fresh run to tailor this resume again."
        )
    answered_clarifications = read_answered_clarifications(layout)
    if not answered_clarifications:                            # (b) nothing to apply
        raise ValueError(
            f"{SHEET_FILENAME} has no answered questions yet; answer at least "
            "one clarification before resuming the paused run."
        )

    checkpointer = open_checkpoint_database(layout.checkpoint_db)   # only now
    pipeline = _build_pipeline(checkpointer)
    config = {"configurable": {"thread_id": manifest.run_id}}       # (c) same thread

    command = Command(
        resume={"status": "candidate_answers_submitted"},           # (d) returned by interrupt()
        update={"clarification_answers": answered_clarifications},  # (e) merged into state
    )
    output = _invoke_pipeline(pipeline, command, config, progress_callback)
```

Everything that can refuse the resume happens **before** a database connection is
opened or a graph is compiled. Failures are then fast and legible, which is
exactly when you want them to be.

**Step 2 — the three mechanisms that make continuation work:**

| Mechanism | Role |
|---|---|
| `thread_id = manifest.run_id` | How LangGraph finds the right checkpoint history. This is what "matches" an answer to its run. |
| `Command(resume=...)` | Becomes the return value of the `interrupt()` call that paused the graph. |
| `Command(update=...)` | Merges new values into state **before** the interrupted node re-runs — this is what puts the answers where the node can see them. |
| Reopening the *same* SQLite file | Without it, there is no history to find. |

### The double-execution trap

> **On resume, LangGraph re-runs the interrupted node from its top — not from the
> `interrupt()` line.** Every statement above that line executes a second time.

The classic bug follows immediately:

```python
def bad_example(state):
    send_email(...)              # <-- runs AGAIN on every resume
    answer = interrupt(question)
    return apply(state, answer)
```

Now look at what makes this codebase's node safe:

```
   FIRST PASS                              ON RESUME
   ----------                              ---------
   clarifications present?  YES            clarifications present?  YES
   answers present?         NO             answers present?         YES
        |                                       |
        |  falls through                        |  returns here
        v                                       v
   interrupt()  --> PAUSE                  return {}   <-- interrupt() is never
                                                            reached a second time
```

`Command(update={"clarification_answers": [...]})` populates state *before* the
node re-runs, so the `if clarification_answers:` branch is taken and the function
returns before it can pause again. Verify it in the logs: a resumed run emits
`candidate_clarifications_received`, never a second
`candidate_clarifications_requested`.

Two independent safeguards are in play. LangGraph's own mechanism would also make
`interrupt()` *return* rather than raise on the second pass, because the resume
value is present in its scratchpad. But this node never gets that far. The state
check does the work — and it is worth knowing that it works because this
particular node happens to have a natural precondition, **not** because the
framework guarantees it for you.

> **THE RULE — rewind, don't restart.** Check state before you ask again. Any
> side effect above an `interrupt()` must be idempotent, or moved below it.

**Step 3 — what happens after the answers land.** The graph does not proceed
straight to assembly:

```python
# src/orchestration/graph.py
def _route_after_candidate_clarifications(state) -> str:
    return "rewrite_experience" if state.get("clarification_answers") else "assemble_resume"
```

A resumed run goes **back** to `optimize_experience`, so bullets are rewritten
once more with the candidate's facts counted as role evidence. Those facts reach
the writer as structured clarification evidence and reach the truthfulness checker
folded into the role description — so a number the candidate supplied is treated
as source truth rather than flagged as an invention.

That second pass can legitimately surface *new* questions, and the graph loops
back to the pause node. **A run pausing twice is expected behaviour, not a bug.**

---

## 13. The whole lifecycle in one picture

```
 CANDIDATE            PIPELINE (process #1)          DISK               PROCESS #2
    |                        |                        |                     |
    |  upload resume + JD    |                        |                     |
    |----------------------->|                        |                     |
    |                        | extract / analyze      |                     |
    |                        | gap analysis           |                     |
    |                        | rewrite bullets        |                     |
    |                        |                        |                     |
    |                        | TRIGGER: one LLM call  |                     |
    |                        | finds a thin bullet    |                     |
    |                        | cap applied            |                     |
    |                        |                        |                     |
    |                        | interrupt() -> raise   |                     |
    |                        |----- checkpoint ------>| checkpoints.sqlite3 |
    |                        |----- sheet ----------->| clarifications_...  |
    |                        |----- manifest -------->| paused_run_manifest |
    |                        |        (expires_at)    |                     |
    |                        X  PROCESS EXITS         |                     |
    |                                                 |                     |
    |         . . . hours or days pass, nothing running . . .               |
    |                                                 |                     |
    |  sees questions (web UI or opens the JSON)      |                     |
    |<------------------------------------------------|                     |
    |  submits answers, keyed by bullet_id            |                     |
    |------------------------------------------------>| audit.jsonl (append)|
    |                                                 | sheet (updated)     |
    |                                                 |                     |
    |                                                 |  resume_paused_run()|
    |                                                 |-------------------->|
    |                                                 | check expiry        |
    |                                                 | check answers exist |
    |                                                 | reopen sqlite       |
    |                                                 |                     |
    |                        Command(resume=, update=) --------------------->|
    |                                                 | node re-runs, takes |
    |                                                 |   the OTHER branch  |
    |                                                 | rewrite w/ answers  |
    |                                                 | assemble ATS        |
    |                                                 | quality gate        |
    |                                                 | render              |
    |  finished resume                                |                     |
    |<--------------------------------------------------------------------- |
```

---
---

# Part III — Production hardening

The six components make HITL *work*. This part makes it *trustworthy*. Each
chapter names a control, explains the failure it prevents, and states honestly
whether this repository implements it.

| Control | Failure it prevents |
|---|---|
| Authorization | An unauthorized person approves a protected action |
| Timeout / expiry | The workflow waits forever, or executes from stale context |
| Escalation & routing | The decision reaches the wrong or unavailable person |
| Optimistic concurrency | Two reviewers both believe they won |
| Idempotent execution | A retry creates a duplicate side effect |
| Audit log | You cannot answer who approved what, when, on what evidence |
| Observability | The approval backlog degrades silently |
| Version compatibility | Paused state cannot be deserialized after a deploy |
| Revalidation | Approved facts changed while the run waited |

---

## 14. The approval state machine

A pending decision is **not a boolean**. Modelling it as one is how duplicate and
stale decisions get in.

```
                          approve
   +---------+ gate  +----------+ --------> +----------+ claim  +-----------+
   | RUNNING |-----> | PENDING  |           | APPROVED |------> | RESUMING  |
   +---------+       +----------+ reject    +----------+        +-----------+
                       |  |  | ----------> +----------+            |     |
                       |  |  |             | REJECTED |     success|     |error
                       |  |  | deadline    +----------+            v     v
                       |  |  +-----------> +----------+      +---------+ +--------+
                       |  |                | EXPIRED  |      |COMPLETED| | FAILED |
                       |  | cancel         +----------+      +---------+ +--------+
                       |  +--------------> +-----------+
                       |                   | CANCELLED |
                       |   re-escalate     +-----------+
                       +<------------------------+
```

| Invariant | Meaning |
|---|---|
| One decision wins | Only one valid transition out of `PENDING` |
| Stale decisions fail | A decision for `state_version` 16 cannot mutate version 17 |
| Authorization is current | Permission is checked **at submission**, not at request time |
| Preconditions revalidated | Mutable facts re-checked before the action |
| Side effect idempotent | Retries create no additional effect |

**This repo:** implements a much smaller machine — a run is either resumable or
expired, computed from `expires_at`. There is no `PENDING → APPROVED` transition
because there is no decision to win, and no `REJECTED` because declining to
answer simply means the thin bullet ships. That is proportionate here, and would
be dangerously thin for Problem B.

---

## 15. Idempotency: the most important reliability pattern

**Idempotency** means repeating the same logical command produces no additional
effect after the first success. For any protected action, derive the key from
**stable business identity**, never from a random retry id:

```python
idempotency_key = f"refund:{order_id}:{approval_id}"

result = payments.refund(
    order_id=order_id,
    amount=approved_amount,
    idempotency_key=idempotency_key,
)
# If the worker retries after a timeout, the provider returns the ORIGINAL
# result instead of issuing another refund.
```

```
   WRONG                                RIGHT
   -----                                -----
   key = uuid4()      <-- new key       key = f"refund:{order_id}:{approval_id}"
   on every retry                            |
        |                                    +-- same key on every retry,
        v                                        so the provider dedupes
   provider sees 3 distinct
   requests -> 3 refunds
```

If the downstream tool has no idempotency support, implement a durable action
ledger with a unique constraint and a `CLAIMED → EXECUTED` transition — but be
honest that a database marker alone cannot be atomic with an external side
effect. Prefer an API that natively supports idempotency, or design a
reconciliation process.

**This repo:** there is **no protected side effect at all**, so there is nothing
to make idempotent. Resuming twice would re-run a rewrite and produce another
document — wasteful, not harmful. This is the single biggest reason this
implementation is simpler than Problem B, and the first thing you would have to
add if you adapted it to gate a real action.

---

## 16. Concurrency: exactly one decision wins

Two reviewers open the queue at the same moment. Both click Approve. Without
control, both succeed and the workflow may resume twice.

The fix is a **compare-and-set** transition, not an application-level check:

```sql
UPDATE approval_request
   SET status = 'APPROVED', decided_by = :subject, decided_at = now()
 WHERE approval_id   = :id
   AND status        = 'PENDING'          -- still open
   AND state_version = :expected          -- not stale
   AND expires_at    > now();             -- not expired

-- Require row_count == 1. If 0 -> 409 Conflict or 410 Gone.
-- Only then enqueue the resume job.
```

Authorization happens *before* that update, using current identity and current
policy. The database transition guarantees only one decision wins even if two
authorized reviewers click simultaneously.

**This repo:** **not implemented.** Nothing prevents `resume_paused_run()` being
invoked twice concurrently on the same directory — no lock file, no version
check. Two concurrent resumes would both read the sheet, both open the same
SQLite file, and race. The consequence here is wasted work and possibly a
corrupted checkpoint, not a double refund, which is why it has not been
prioritised — but it is a real gap and it is named as one.

---

## 17. Authorization: approve the action, not the person

"The user is logged in" is not authorization. Permission must be scoped to the
**concrete pending action**, evaluated at submission time.

| Rule | Example |
|---|---|
| Role threshold | Support manager may approve refunds ≤ $1,000 |
| Domain ownership | Finance approver required above $1,000 |
| Separation of duties | Whoever proposed the adjustment may not approve it |
| Resource scope | Reviewer may approve only accounts in their business unit |
| Step-up authentication | Very high-risk approvals require stronger auth |

> **SECURITY BOUNDARY**
> The reviewer response is **external input**. Validate its schema, authenticate
> the submitter, authorize the submitter *for this action*, verify the request is
> still open, and verify the version is current. Never take reviewer identity
> from the request body — take it from the authenticated session.

**This repo:** **not implemented, and labelled as such.** Anyone holding the
`run_id` can answer; anyone with filesystem access can edit the sheet.
`answered_by="web-ui (unauthenticated)"` is deliberately honest so the audit does
not imply an identity that was never verified. The seam is in place: threading a
real principal through `record_clarification_answers(..., answered_by=...)` is a
small change. The *policy* around it is the real work.

---

## 18. Freshness: revalidate before the side effect

A human may approve based on facts that were true six hours ago. Before executing
a protected action, re-check mutable preconditions: account status, balance,
order state, resource version, policy version, and whether a newer workflow
already acted.

If material facts changed, **do not silently execute the old approval.** Either
reject it as stale or generate a fresh request showing the updated facts.

> An approval is consent for a *particular action under a particular context*,
> not a permanent authorization token.

**This repo:** partially relevant. There is no action to revalidate, but the
analogous staleness exists: if the resume file changed between pause and resume,
`bullet_id`s may no longer match, and `answers_for_role` will simply not route
those answers. That fails safe — the answer is ignored rather than applied to the
wrong bullet — but it fails *silently*, which is worth knowing when debugging.

---

## 19. Timeouts and escalation

Every pending request needs a deadline and a deterministic timeout policy.
**"Wait forever" is not a policy.**

| Timeout action | Use when |
|---|---|
| Auto-reject | Default-safe for destructive or financial actions |
| Escalate to another reviewer | The business tolerates waiting but not abandonment |
| Cancel the workflow | The request becomes meaningless after the deadline |
| Re-evaluate, create fresh request | Facts or policy may have changed materially |
| Auto-approve | Rare — only where policy explicitly allows and risk is bounded |

Avoid silent auto-approval as a convenience fallback. If human review was
required because the action exceeded machine authority, a timeout should
normally **fail closed**.

**This repo:** implemented, minimally and appropriately.

```python
paused_at = datetime.now(UTC)
manifest = ExperienceClarificationPausedRunManifest(
    run_id=run_id, resume_path=resume_path, jd_path=jd_path,
    paused_at=paused_at,
    expires_at=paused_at + timedelta(hours=get_config().workflow.clarification_ttl_hours),
)
```

Default TTL is 168 hours (7 days). The timeout action is **cancel**: an expired
run refuses to resume and the candidate starts fresh. There is no escalation
because there is exactly one possible responder — a routing decision that does
not exist.

> **Why `is_expired` is computed, never stored.** There was once a `status` enum
> with a single value nothing ever read. Adding `EXPIRED` to it would have been
> the obvious move and the wrong one: a *stored* status must be updated by
> something, and that something can fail, lag, or never run for a directory
> nobody touches. A *computed* status cannot disagree with reality.
> **Derive what you can derive; store only what you cannot.**

---

## 20. Versioning the pause

Long pauses cross software versions. A run can pause under v3.2 and resume after
v3.3 ships. If state schema, node names, or tool contracts changed, a naive
resume fails or — worse — behaves differently.

- Persist `workflow_version`, `policy_version`, and important schema versions
  with the request.
- Keep old workflow code available until paused runs for that version drain, or
  implement an explicit migration.
- Do not reinterpret an old approval under materially different policy without
  re-review.
- Version API response schemas so an old reviewer UI cannot submit a malformed
  decision.
- **Test deserialization and resume across N-1 → N before every release.**

**This repo:** **not implemented.** No version is persisted with a paused run. In
practice the 7-day TTL bounds the exposure — a paused run cannot outlive many
deploys — but that is mitigation by accident, not by design. If you extend the
TTL, add a version field first.

---

## 21. Privacy, secrets, and what not to persist

Checkpoints and request payloads are durable application data that operators and
reviewers may see. Treat them accordingly.

- Store **secret references**, never raw credentials or tokens.
- Minimise personal data in the payload; fetch detailed evidence only for
  authorized reviewers.
- Encrypt at rest and in transit; consider application-level encryption for
  particularly sensitive checkpoint fields.
- Set retention periods for checkpoints, payloads, and audit records.
- Log identifiers and hashes rather than copying raw documents into logs.

**This repo:** a paused run contains a real résumé — inherently personal data —
sitting on local disk indefinitely until resumed or expired. The pipeline has a
PII redaction path with a run-scoped mapping that is *deliberately kept alive*
while a run is paused, because rehydration needs it after resume. That is the
right behaviour and also a retention consideration: a paused run holds both the
redacted state and the mapping needed to reverse it.

---

## 22. Observability: age beats count

HITL adds a queue whose latency can dominate end-to-end time. The system is not
healthy merely because agent traces look healthy.

| Metric | What it tells you |
|---|---|
| **Oldest pending age** | Whether anything is being abandoned |
| Pending count by queue/risk | Capacity and routing pressure |
| p50/p95/p99 decision latency | Human-loop service level |
| Approve / reject / edit ratio | Whether the trigger is calibrated |
| Approve-without-edit rate | Possible rubber-stamping or over-triggering |
| Expiry / escalation rate | Routing or staffing weakness |
| Resume failure rate | Serialization, version, or runtime issues |
| Duplicate decision conflicts | Concurrency and UI retry behaviour |
| Human override reasons | The best signal for improving the policy |

> **THE METRIC TO PAGE ON**
> **Oldest pending age**, not queue size. A queue of 200 five-minute approvals is
> healthy. Three approvals with one stuck for two days is an incident.

**This repo:** **not implemented.** Nothing counts paused runs or measures how
long the oldest has waited; discovering a stale one means listing
`tailored_resumes/**/paused_run_*` by hand. If this became multi-user, this is
the first thing to build.

---

## 23. This project's honest scorecard

| Control | State | Note |
|---|---|---|
| Durable pause/resume | **Solid** | `interrupt()` + SqliteSaver + `thread_id`; the process genuinely exits |
| Double-execution safety | **Solid** | State-checked branch; verified by which log event a resumed run emits |
| Trigger auditability | **Solid** | Rubric in version control, `temperature=0.0`, reasoning persisted with the question |
| Timeout / expiry | **Solid** | `expires_at` + refusal before any machinery opens |
| Audit trail | **Solid** | Append-only `answers_audit.jsonl`, written before the mutable sheet |
| Escalation fatigue | **Solid** | `max_clarifications_per_run` cap, with logging |
| Contract integrity | **Solid** | One class both directions; partial gaps unrepresentable |
| Escalation & routing | **N/A** | Exactly one possible responder |
| Idempotent side effects | **N/A** | No protected side effect exists |
| Authorization | **Missing** | Honest label only; the seam exists |
| Concurrency control | **Missing** | No lock or version guard on double resume |
| Version compatibility | **Missing** | Mitigated only by the short TTL |
| Queue observability | **Missing** | No age or backlog metric |

Read that table as a worked example of the real skill: **deciding which controls
your risk profile requires.** Three of the "Missing" rows would be release
blockers for Problem B. Here they are known, bounded, and written down — which is
a defensible engineering position, and very different from not having considered
them.

---
---

# Part IV — Building your own

## 24. Choosing a framework

| Framework | Pause primitive | Durable resume object | Multi-agent behaviour | Best fit |
|---|---|---|---|---|
| **LangGraph** | `interrupt()` + `Command(resume=...)` | Checkpointer keyed by `thread_id` | Interrupts propagate through graph/subgraph; node replay is explicit | Stateful graphs, explicit orchestration, complex approval logic |
| **OpenAI Agents SDK** | `needs_approval` / `interruptions` | Serializable `RunState` | Nested and handoff approvals surface on the outer run | Tool-centric agents with built-in approval flow |
| **Microsoft Agent Framework** | `approval_mode="always_require"` | Session + app-managed workflow state | Approval-marked tools handled through the agent/harness loop | Azure-oriented agent apps |
| **AutoGen** | `UserProxyAgent`, or terminate-and-resume | Saved team state | Human participates as a team member or between runs | Conversational teams; use the persisted between-run pattern for long waits |

> **DO NOT CHOOSE BY THE NAME OF THE FEATURE.** Evaluate four semantics:
> **What is persisted? What re-executes on resume? How are pending requests
> identified? Can the pause survive process death and deployment?**

A note on AutoGen worth generalising: its in-run `UserProxyAgent` **blocks**
until feedback arrives, and its own documentation points to a different pattern
for asynchronous workflows — terminate, save state, resume later. Conversational
HITL and durable workflow HITL are not the same operational problem. A terminal
prompt is fine for a demo; an approval that might wait overnight needs persisted
state and a restartable runtime.

---

## 25. The implementation path, in order

Build in this order. Each step is verifiable before the next adds risk.

```
   LEARN            1. One deterministic trigger (amount > 50)
    |               2. One interrupt node, in-memory checkpointer
    |               3. Resume with same thread_id -- OBSERVE the node restart
    v
   CORRECT          4. Move the side effect to a separate post-approval node
    |                  and give it an idempotency key
    |               5. Switch to a persistent checkpointer; kill the process
    |                  mid-pause and prove it resumes
    v
   CONTROL          6. Durable approval table + a simple review page
    |               7. Authenticated reviewer identity + action-level authorization
    |               8. Compare-and-set decision transition + duplicate-resume tests
    v
   OPERATE          9. expires_at, timeout behaviour, escalation, stale-fact revalidation
    |              10. Audit events and operational metrics
    |              11. Test a deployment while a run is paused
    v
   EXTEND         12. Only now: Edit / Choose / Supply interactions, parallel approvals
```

Step 3 deserves emphasis: **actually observe the node restarting.** Put a print
statement above the `interrupt()` call and watch it fire twice. Everything about
replay safety becomes obvious once you have seen it with your own eyes.

---

## 26. Six anti-patterns

Named as pharmacy failures, because the image makes them memorable.

| Name | Software failure | Correction |
|---|---|---|
| **Sticky Note** | State lives only in RAM and disappears | Durable checkpoint |
| **Blocking Technician** | A thread or process waits for hours | Persist and exit; resume later |
| **Blank Signature Line** | Reviewer sees "Approve?" with no evidence | Structured, context-rich contract |
| **Wrong Prescription** | Decision applied to the wrong or stale run | Strong IDs + state version + open-status check |
| **Double Dispense** | Protected side effect executes twice | Idempotent execution and node placement |
| **Rubber Stamp** | So many approvals that acceptance becomes reflexive | Tighten triggers; measure override and edit rates |

> **MNEMONIC** — Sticky Note · Blocking Technician · Blank Signature ·
> Wrong Prescription · Double Dispense · Rubber Stamp.

This repository has defences against four of these (durable checkpoint;
persist-and-exit; a context-rich question carrying `why_flagged` and
`missing_fact_summary`; and a volume cap against rubber-stamping). It is exposed
to *Wrong Prescription* in the narrow sense that a resume edited mid-pause can
desync `bullet_id`s, and it is immune to *Double Dispense* only because it
dispenses nothing.

---

## 27. The test plan that proves it

A demo shows the happy path. These tests show the architecture.

| Test | Expected result |
|---|---|
| Kill the runtime after interrupt, before the answer | Checkpoint survives; request still answerable |
| Restart on another machine or container | Same thread resumes successfully |
| Double-click Approve | Exactly one decision transition and one protected effect |
| Deliver the same resume job twice | Second processing is a no-op or returns the existing result |
| Reviewer loses authorization after the request was created | Decision rejected at submission time |
| Underlying record changes before the answer arrives | Precondition check blocks stale execution |
| Expire the request, then submit an answer | Rejected as expired; no side effect |
| Deploy a new workflow version with a paused old run | Pinned version resumes, or migration is explicit |
| Prompt-inject a subagent to bypass the gate | Capability policy still blocks the action |
| Parallel approvals answered in reverse order | Each response maps to the correct request |
| Malformed human response | Schema validation fails; workflow does not resume |
| Broad `try/except` around `interrupt` | Test should demonstrate why this is forbidden |

**What this repository actually tests today** (`tests/unit/hitl/professional_experience/`):
sheet round-trip through the single accessor; the on-disk shape is an object, not
a list (the regression that caused the 500); answers recorded by `bullet_id`, not
position; answering across multiple sittings preserves earlier answers; unknown
`bullet_id` rejected; whitespace-only answers rejected; the audit log is
append-only while the sheet holds only the latest; expiry computed from the
clock; and the join behaviour — gap present, gap null, unknown bullet, and the
impossibility of constructing a partial gap.

Absent, and honestly so: crash-mid-pause, duplicate-resume, and
deployment-across-pause tests.

---

## 28. Design review invariants and checklist

> **SIX PRODUCTION INVARIANTS**
> 1. No protected action executes before authorization.
> 2. No request exists without durable resumable state.
> 3. One pending decision transitions exactly once.
> 4. Resume is bound to the exact workflow and version.
> 5. Protected side effects are idempotent.
> 6. The entire human decision is auditable.

If an architecture cannot demonstrate all six under crash, retry, concurrency,
and deployment, it does not yet have production-grade HITL.

**Checklist**

```
[ ] Trigger is explicit and classified Costly / Confusing / Compulsory
[ ] Compulsory gates are deterministic, not model-confidence decisions
[ ] Trigger volume is deterministically bounded
[ ] Protected action cannot execute before the approval boundary
[ ] Checkpoint is durable and survives process death and deployment
[ ] thread_id / workflow id is stable; resume uses the same identity
[ ] Request is structured and carries decision-relevant context
[ ] Human response is schema-validated and action-authorized
[ ] PENDING -> terminal transition is atomic and version-guarded
[ ] Expired / cancelled / already-decided responses are rejected
[ ] Preconditions revalidated immediately before the side effect
[ ] Protected side effect has stable idempotency semantics
[ ] Code before interrupt() is replay-safe; no broad try/except swallows it
[ ] Nested subgraph replay behaviour has been considered
[ ] Parallel requests correlate by durable id, never by position
[ ] Every request has timeout and escalation behaviour
[ ] Workflow / policy / schema versions persisted for long waits
[ ] Audit record includes context shown, actor, decision, note, timestamps
[ ] Oldest pending age, latency percentiles, expiry and override rates monitored
[ ] Crash, duplicate, stale, concurrency, and N-1 -> N deploy tests pass
```

---

## 29. The 90-second explanation

> **SAY THIS IN A DESIGN REVIEW**
>
> HITL in a multi-agent system is a durable control-plane boundary. A policy —
> deterministic wherever the rule permits it — decides when machine authority
> ends. The runtime interrupts *before* the protected capability, persists graph
> state under a stable workflow/thread identifier, and creates a structured
> request. An authenticated, authorized reviewer approves, rejects, edits, or
> supplies data. The application atomically claims that pending decision, rejects
> stale or duplicate answers, loads the same checkpoint, and resumes. In
> LangGraph the interrupted node restarts from its beginning, so pre-interrupt
> code must be replay-safe and protected side effects belong after the gate in
> separate idempotent nodes. Production adds deadlines, escalation, per-action
> authorization, revalidation, audit, metrics, and version compatibility. For
> irreversible actions the gate belongs at the capability boundary, so no
> subagent or routing change can bypass it.

And for this repository specifically:

> This is a **Supply** ask with no protected side effect, so it correctly skips
> idempotency and approval semantics, and correctly implements durability,
> expiry, audit, and a volume cap. Its known gaps are authorization, concurrency
> control, and version pinning — all bounded by a 7-day TTL and a single-user
> deployment model.

---

## 30. Mnemonic sheet

| Memory hook | Recall |
|---|---|
| STOP · FREEZE · ASK · STORE · SHOW · GO | The minimum runtime lifecycle |
| Costly · Confusing · Compulsory | The three reasons to escalate |
| Approve · Edit · Choose · Supply | The four kinds of ask |
| A pause is an exit, not a wait | Durable HITL separates execution time from human time |
| Rewind the node, not the run | The interrupted node re-executes from its start |
| Gate the capability, not the prompt | Enforcement belongs before the irreversible tool |
| One decision, one effect | Concurrency control plus idempotency |
| Fresh facts before effects | Revalidate mutable preconditions after long waits |
| Version the pause | Persist workflow / policy / schema versions |
| Age beats count | Oldest pending age is the queue-health metric |
| Derive, don't store | A computed status cannot disagree with reality |
| One fact, one place | Two copies of a format will diverge |

---
---

## Appendix A — file map

```
src/hitl/professional_experience/
|
+-- README.md            <- you are here
|
+-- models.py            <- CandidateFactGap (what's missing + the question),
|                           ExperienceBulletClarification (the persisted
|                           question/answer, which inherits it), the audit
|                           record, and the paused-run manifest
|
+-- clarifications.py    <- the trigger (one LLM call) plus the pure-code join
|                           that attaches role identity to each gap
|
+-- answers.py           <- routes answered questions back to their exact role
|                           and folds them into evidence for the next rewrite
|
+-- persistence.py       <- PausedRunLayout (the only place file names live) and
                            the only code that reads or writes the sheet and audit

Elsewhere:

src/orchestration/checkpointing.py    <- the SqliteSaver and its msgpack allowlist
src/orchestration/nodes/experience.py <- the pause node + the volume cap
src/orchestration/graph.py            <- wires the pause node into the topology
src/orchestration/state.py            <- the TypedDict that IS the checkpointed state
src/orchestration/runner.py           <- resume_paused_run(), expiry refusal
src/orchestration/human_review_policy.py <- a DIFFERENT, non-resumable escalation
web_app/server.py                     <- the web half of the human surface
src/main.py                           <- the CLI half (--resume-from)
src/config/tool_prompts/hitl/experience_candidate_fact_gap.md  <- the trigger's rubric
tests/unit/orchestration/test_checkpoint_allowlist.py <- allowlist drift guard
tests/unit/hitl/professional_experience/  <- sheet, audit, expiry, and join tests
```

### A different escalation path — do not confuse the two

`src/orchestration/human_review_policy.py` also has a notion of escalating to a
human, via `RunDisposition.NEEDS_HUMAN_REVIEW`. **It is not this mechanism and it
pauses nothing.**

| | This module | `human_review_policy.py` |
|---|---|---|
| Trigger | An LLM judges a bullet thin | ATS render check is `INCONCLUSIVE` or unrecoverably `FAIL` |
| Effect | Graph **pauses**, waits for an answer | Run **terminates** with `human_review_required=True` |
| Resumable | Yes | No — a human intervenes outside this codebase |
| Who acts | The candidate | An internal reviewer (no queue or notification exists) |

Debugging "why did this run stop": a `paused_run_path` on the result means this
module; `NEEDS_HUMAN_REVIEW` with no paused path means the other.

---

## Appendix B — glossary

| Term | Meaning |
|---|---|
| **Approval gate** | A control point where execution cannot proceed until a required decision is satisfied |
| **Approval request** | Durable business object describing the proposed action, context, allowed responses, deadline, and version |
| **Audit trail** | Append-only record of request, evidence, actor, decision, and timestamps |
| **Checkpoint** | Persisted workflow state sufficient to continue later |
| **Clarification** | *(this repo)* One question about one bullet, and later its answer — the same object throughout |
| **Control plane** | Logic governing whether and how work may proceed |
| **Fact gap** | *(this repo)* The LLM's judgment that a bullet is missing a candidate-owned fact; its presence **is** the trigger |
| **Idempotency** | Repeating the same logical operation produces no additional effect after the first success |
| **Interrupt** | Framework mechanism suspending execution and returning control for external input; in LangGraph it raises |
| **Optimistic concurrency** | An update succeeds only if the record still has the expected version |
| **Paused run directory** | *(this repo)* `paused_run_<run_id>/`, holding sheet, manifest, audit log, and checkpoint |
| **Replay** | Re-execution of code after retry, resume, or recovery |
| **Resume** | Loading persisted state, injecting external input, continuing execution |
| **State version** | Comparable version used to detect stale decisions |
| **Supply ask** | A human provides information the system could not obtain, as opposed to approving, editing, or choosing |
| **Thread ID** | The checkpointer's key for one workflow's state; always the `run_id` here |
| **Trigger policy** | The rule defining when machine authority ends |
| **Workflow version** | The code/schema/policy compatibility boundary of a paused run |

---

## Appendix C — references

The general patterns in Parts I, III, and IV follow current official framework and
cloud documentation plus NIST guidance. **Framework APIs change quickly — verify
resume semantics, serialization constraints, and approval APIs against current
docs before implementing.** Claims about *this repository* were verified directly
against the code in this tree.

1. **LangGraph — Interrupts** · https://docs.langchain.com/oss/python/langgraph/interrupts
   `interrupt()`, `Command(resume=...)`, node restart semantics, interrupt rules, parallel interrupts, tools, subgraphs.
2. **LangGraph — Checkpointers** · https://docs.langchain.com/oss/python/langgraph/checkpointers
   `thread_id`, checkpoints, fault tolerance, state snapshots, namespaces.
3. **LangGraph — Persistence** · https://docs.langchain.com/oss/python/langgraph/persistence
   Checkpointer vs store, production persistence guidance, in-memory limitations.
4. **LangGraph Reference — PostgreSQL checkpointers** · https://reference.langchain.com/python/langgraph/checkpoints
   `PostgresSaver` / `AsyncPostgresSaver`, `setup()`, serialization and encryption guidance.
5. **OpenAI Agents SDK — Human-in-the-loop** · https://openai.github.io/openai-agents-python/human_in_the_loop/
   `needs_approval`, interruptions, `RunState` approve/reject, nested approvals, versioning pending tasks.
6. **OpenAI Agents SDK — RunState reference** · https://openai.github.io/openai-agents-python/ref/run_state/
7. **Microsoft Agent Framework — Tool approvals** · https://learn.microsoft.com/en-us/agent-framework/agents/tools/tool-approval
   `approval_mode`, approval request/response flow, Harness Agent middleware.
8. **AutoGen — Human-in-the-loop** · https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/human-in-the-loop.html
   `UserProxyAgent` in-run feedback; persisted terminate/resume pattern for async feedback.
9. **NIST AI 600-1 — Generative AI Profile** · https://doi.org/10.6028/NIST.AI.600-1
   Human-AI configuration, operator proficiency, monitoring outcomes, operator involvement in testing.
10. **Azure Service Bus — message loss and duplicates** · https://learn.microsoft.com/en-us/azure/service-bus-messaging/service-bus-message-loss-and-duplicates
    At-least-once delivery with Peek Lock, redelivery causes, idempotent consumers.
11. **Azure Database for PostgreSQL — High availability** · https://learn.microsoft.com/en-us/azure/postgresql/high-availability/concepts-high-availability

---

> **A note on scope.** This document describes the code as it is today. If you
> change `await_candidate_clarifications`, `resume_paused_run`, or the shape of
> the state, re-verify [chapter 12](#12-component-6--resume-how-does-the-answer-get-back-in)
> and [chapter 23](#23-this-projects-honest-scorecard) in particular — they are
> the two most likely to go quietly stale.
