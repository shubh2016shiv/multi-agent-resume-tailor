# Memory Boundaries
## What an LLM Is Allowed to See, Hold Onto, and Forget

> **Scope:** `src/formatters/` (the context-building layer), `src/formatters/llm_context_rendering.py`
> (the TOON encoding), `src/core/llm_cache.py`, and `src/core/llm_token_tracker/`.
> **Audience:** Contributors writing a new formatter; anyone asking "why does this
> agent get to see X but not Y," or "does anything in this system actually remember
> previous runs?"

---

## Table of Contents

1. [What Problem Memory Boundaries Solve](#1-what-problem-memory-boundaries-solve)
2. [The Reframing — This System Has No Agent Memory At All](#2-the-reframing--this-system-has-no-agent-memory-at-all)
3. [The Formatter Layer — Where Memory Boundaries Are Actually Drawn](#3-the-formatter-layer--where-memory-boundaries-are-actually-drawn)
4. [TOON — The Textual Shape Memory Takes](#4-toon--the-textual-shape-memory-takes)
5. [Case Study: Three Formatters, Three Different Truncation Policies](#5-case-study-three-formatters-three-different-truncation-policies)
6. [The Blackboard Slice — One Strategy Object, Many Narrow Readers](#6-the-blackboard-slice--one-strategy-object-many-narrow-readers)
7. [What an Agent Never Sees of Its Own or Another Agent's Work](#7-what-an-agent-never-sees-of-its-own-or-another-agents-work)
8. [The PII Boundary — Recap and Cross-Reference](#8-the-pii-boundary--recap-and-cross-reference)
9. [Persistence That Looks Like Memory But Isn't — The Response Cache](#9-persistence-that-looks-like-memory-but-isnt--the-response-cache)
10. [The Token Budget — A Hard Ceiling, Enforced Before the Call](#10-the-token-budget--a-hard-ceiling-enforced-before-the-call)
11. [Unwired Infrastructure — track_agent_tokens and the Real Source of Truth](#11-unwired-infrastructure--track_agent_tokens-and-the-real-source-of-truth)
12. [Design Rule — Deciding What Crosses a Memory Boundary](#12-design-rule--deciding-what-crosses-a-memory-boundary)
13. [Future Considerations](#13-future-considerations)

---

## 1. What Problem Memory Boundaries Solve

Every stage of this pipeline has access, in principle, to the entire typed
pipeline state described in [State Management](state-management.md) — the full
original resume, the full job description, every upstream agent's complete
output. Handing all of that to every agent on every call would be the easiest
thing to implement and one of the worst things to actually do. It would blur
each agent's job (a summary writer that can see the ATS assembler's keyword
scoring starts trying to do ATS optimization too), it would let one section's
drafting bias leak into another's, it would waste tokens on data an agent has
no task-relevant use for, and — for personally identifiable information
specifically — it would mean sending a candidate's real name and contact
details into a model call that never needed them.

Memory boundaries are the deliberate, per-agent answer to "of everything this
system knows at this point in the run, what is this one LLM call actually
allowed to see?" In this codebase that answer is not a vague policy — it is a
concrete Python module per agent, and this document is a tour of exactly what
each one keeps, what it drops, and why.

---

## 2. The Reframing — This System Has No Agent Memory At All

Before looking at what crosses a memory boundary, it's worth being precise
about what does not exist in this codebase in the first place: there is no
persistent agent memory, conversational history, or long-term recall of any
kind. Every single CrewAI task in this pipeline runs as
`Crew(agents=[agent], tasks=[task], process=Process.sequential).kickoff()` —
one agent, one task, one call (`src/orchestration/crew_task_execution.py::run_agent_task`,
documented in [Orchestration Graph §6](orchestration-graph.md#6-run_agent_task--the-one-seam-between-the-graph-and-crewai)).
Every node that needs an agent calls that agent's `create_*_agent()` factory
fresh (documented in [Agent Roles §3](agent-roles.md#3-anatomy-of-an-agent-factory--the-common-recipe)),
builds a brand-new context string, and gets one bounded response back. There
is no chat history object anywhere in this pipeline, no vector store of past
interactions, no agent that "remembers" the previous stage's reasoning except
through what a human — or in this case, a formatter — explicitly wrote into
its prompt.

```text
   WHAT THIS SYSTEM DOES NOT HAVE:              WHAT IT HAS INSTEAD:
   ---------------------------------             ------------------------------
   conversational memory across turns            a fresh Task, every single call
   an agent "recalling" a prior stage             a formatter re-projecting exactly
                                                    the fields that stage produced
   implicit continuity between agents             explicit typed state (State
                                                    Management) + explicit formatted
                                                    context (this document)
```

This reframing matters because it means "memory boundaries," in this system,
are really "context construction boundaries" — the question is never "what
does the agent remember," it is "what does this one formatter choose to put
in front of the model this one time." That is a fully deterministic, fully
auditable question, which is the entire point.

---

## 3. The Formatter Layer — Where Memory Boundaries Are Actually Drawn

`src/formatters/` has one module per agent that needs constructed context, and
every one of them follows the identical shape: a handful of small
`select_*_context` functions, each responsible for filtering exactly one
upstream data source down to the fields one agent's task actually needs, then
one `build_*_payload` function assembling those slices, then one
`format_*_context` entry point that renders the assembled payload to a string.

```text
   node calls format_<agent>_context(typed objects from state)
                          |
                          v
   build_<agent>_payload(...)
     calls several select_*_context(...) functions, ONE PER SOURCE OBJECT
       each keeps only the fields this agent's task needs
       each drops everything else, explicitly, often with a documented reason
                          |
                          v
   render_context_data(payload, format_type="toon")
                          |
                          v
   a compact string -- this, and ONLY this, is what the agent's task ever sees
```

Every formatter module's own docstring states, in plain prose, exactly what it
keeps and what it drops — this is not left to be inferred from reading the
select functions. That documentation-as-contract habit is itself part of the
memory-boundary discipline: a reviewer can read one docstring and know an
agent's entire information diet without tracing code.

---

## 4. TOON — The Textual Shape Memory Takes

Once a formatter has decided which fields survive, `src/formatters/llm_context_rendering.py`
renders that filtered payload into the literal text an LLM reads. The default
format, TOON, is a compact key-value notation — not JSON — purpose-built for
LLM context:

```text
   candidate_profile:
     professional_summary: "Backend engineer with 6 years..."
     skills:
       - skill_name: Python
         category: Backend
     work_experience:
       - job_title: Senior Engineer
         company_name: Acme Corp
   target_job:
     job_title: Staff Engineer
```

Quoting is applied only when a string actually needs it — empty strings,
strings containing whitespace, strings starting with a digit, or strings
containing structural characters (`{}[]()<>:,-"'`) — so the common case (a
short, unambiguous token like a skill name) renders with no quote-character
overhead at all. A second renderer, `render_markdown`, exists for the same
filtered payload and is explicitly for **optional human review and debugging
output**, never for what actually ships to a model in production. The two
renderers sharing one input contract (`render_context_data(data, format_type=...)`)
means a contributor debugging what an agent saw can render the exact same
payload as readable Markdown without touching the formatter logic that
decided the payload's contents in the first place.

---

## 5. Case Study: Three Formatters, Three Different Truncation Policies

The clearest evidence that memory boundaries are a deliberate, per-agent
decision — not a system-wide default — is that three formatters looking at
similar data (a candidate's achievements, a job's requirements) make three
different, and at one point contradictory, choices about how much to keep.

**`gap_analysis_formatter`** truncates each work-experience entry's
achievements to `achievements[:3]` before the Gap Analysis Specialist ever
sees them. This agent's job is breadth: compare the whole career against the
whole job, across every role, so a handful of representative achievements per
role is the right amount of evidence for a comparison that spans the entire
resume at once.

**`experience_optimizer_formatter`** caps job requirements to
`MAX_PRIORITY_REQUIREMENTS_FOR_EXPERIENCE_REWRITE = 6`, ranked so must-have
requirements sort before should-have and nice-to-have ones. This agent's job
is the opposite of breadth: it rewrites **one role's** bullets at a time (see
[Agent Roles §4.5](agent-roles.md#45-experience-section-optimizer)), so it
needs a short, prioritized signal of what to emphasize, not the JD's entire
requirement list repeated on every one of the pipeline's parallel per-role calls.

**`professional_summary_formatter`** does neither — it keeps **every**
achievement, in every role, completely uncapped and unranked, and its own
docstring explains why in unusual detail: an earlier version of this same
formatter *did* cap and rank achievements by keyword match, and separately
injected a "use this vocabulary" sentence into the writer's context. Both
mechanisms shipped, and both were later identified, via live smoke-test runs,
as the direct cause of the writer's own reported failure mode — generic,
checklist-style summaries that read like a capability list rather than a
narrative. The ranking-by-vocabulary-match cap on achievements "silently hid a
candidate's most JD-relevant evidence" in at least one real run. The fix was
not a better ranking heuristic; it was to stop ranking and capping at all,
because "the writer's own evidence_used step already does the picking, and it
does a better job of it when it can see everything."

```text
   gap_analysis_formatter:            achievements[:3]     -- breadth over depth
   experience_optimizer_formatter:    top 6 requirements    -- focus for one role
   professional_summary_formatter:    UNCAPPED, UNRANKED    -- corrected in production;
                                                                capping caused the bug
```

This is the single strongest argument in this codebase against a uniform
"summarize everything down to save tokens" policy: for at least one agent,
that exact instinct was tried, shipped, caused a measured quality regression,
and was reversed. Memory boundaries are decided per agent, against that
agent's actual task, and are themselves subject to being wrong and needing
correction — not a rule to apply mechanically everywhere.

---

## 6. The Blackboard Slice — One Strategy Object, Many Narrow Readers

`AlignmentStrategy` (produced once, by the Gap Analysis Specialist) is read by
three different downstream agents, and `professional_summary_formatter`'s own
docstring names the pattern explicitly: *"AlignmentStrategy documents a
Blackboard Pattern (one writer... many readers) specifically so downstream
agents read that agent's own analysis instead of re-deriving it."*

```text
                         AlignmentStrategy
                    (one object, five+ fields)
                              |
        +----------------------+----------------------+
        |                      |                        |
        v                      v                        v
professional_summary_    experience_optimizer_    skills_optimizer_
formatter reads ONLY:    formatter reads ONLY:     formatter reads ONLY:
  summary_of_strategy      experience_guidance        skills_guidance
  professional_summary_
  guidance
```

Each formatter takes exactly its own slice of the shared strategy object and
passes it through **unmodified** — never re-summarized, never reinterpreted.
The professional summary formatter's docstring records a real regression here
too: an earlier version discarded the Gap Analysis agent's actual guidance and
substituted a hardcoded paragraph on every run, which quietly duplicated (and
diverged from) the one agent whose entire job was to produce that guidance.
The fix was to trust the blackboard's writer and stop re-deriving its output
downstream — the same "don't let two places decide the same thing" principle
that shows up as `AlignmentStrategy` itself in
[Agent Roles §4.3](agent-roles.md#43-gap-analysis-specialist).

---

## 7. What an Agent Never Sees of Its Own or Another Agent's Work

Formatters don't only filter *source* documents — they also filter what one
agent's own finished output is allowed to carry forward into the next agent's
memory. `ats_optimization_formatter` is the clearest example: the Professional
Summary Writer produces **four** full drafts plus a recommended version and
self-critique notes (`ProfessionalSummary`, see
[Agent Roles §4.4](agent-roles.md#44-professional-summary-writer)), but
`choose_summary_text` resolves that down to **one string** — the chosen
draft's content — before the ATS Optimization Specialist ever sees any of it.
The other three drafts, and every reason the writer gave for preferring one
over another, never leave the summary node's own scope. Likewise,
`OptimizedExperienceSection.optimization_notes` and `.keywords_integrated`
bookkeeping never cross into the ATS assembler's context at all — only
`optimized_experiences`, the actual rewritten entries, do. A downstream agent
sees a prior agent's **decision**, never its deliberation.

The `quality_feedback_formatter` is the deliberate exception that proves this
rule has a reason, not just a habit: because this agent's entire job is a
truthfulness comparison between two full resumes, it is handed both the
complete original resume and the complete final resume via unfiltered
`model_dump(mode="json")` calls — no capping, no slicing. When an agent's task
is fundamentally "compare two whole things," the formatter's job is to make
sure both whole things are actually present, not to guess which parts of each
might matter.

---

## 8. The PII Boundary — Recap and Cross-Reference

The most safety-critical memory boundary in the system — real personal data
never reaching an LLM call unless the feature flag is off — is owned by the
redaction/rehydration mechanism and the Redis-backed mapping store, not by any
formatter in this directory. That boundary is documented in full in
[State Management §9](state-management.md#9-pii-mapping--state-deliberately-kept-out-of-the-typed-pipeline-state):
placeholders like `[PERSON_1]` stand in for real values everywhere in typed
state, `assert_extraction_input_redacted` actively re-verifies no raw PII
value survives in text about to reach the extraction agent, and real values
are restored only after every quality gate has run, immediately before
rendering. It is mentioned here only to place it correctly on the map: this is
the one memory boundary in the system that exists for legal and safety
reasons rather than task-scoping reasons, and it is enforced by a separate,
narrower, fail-closed store precisely because a formatter's job is to decide
*what* an agent needs, not to be the last line of defense against a *sensitive
value* an agent must never see at all.

---

## 9. Persistence That Looks Like Memory But Isn't — The Response Cache

`src/core/llm_cache.py` configures LiteLLM's disk-backed response cache, and
it is worth being precise about why this is not a form of agent memory despite
persisting across process runs on disk. The cache key is the entire request —
model name, messages, and generation parameters — hashed together; a cache hit
means "this exact request was made before and here is the exact response that
came back," never "the model recalls something related." Two requests that
differ by so much as a single word in the rendered TOON context are different
cache entries. The project's own internal guide is emphatic on this
distinction: it is response caching, explicitly **not** token counting,
prompt compression, vector search, retrieval, or "storing conversation
memory."

The cache is wired at exactly the two real LLM entry points — `run_agent_task`
and `request_structured_output` — and nowhere else, toggled by
`feature_flags.enable_cache`. Its actual purpose is developer cost and
iteration speed: rerunning the same test, retrying after an unrelated bug fix,
or reloading a tool with identical inputs all skip a redundant, identical
provider call. It has no bearing on what any agent is allowed to see on a
*new* request — it only ever short-circuits a request that is byte-for-byte
identical to one already answered.

---

## 10. The Token Budget — A Hard Ceiling, Enforced Before the Call

`ensure_token_budget` (`src/core/llm_token_tracker/budget_guard.py`) is the
one place a memory boundary is enforced not by curating *which* fields are
included, but by measuring the *total size* of what a formatter assembled and
refusing to proceed if it's too large. It is called from
`request_structured_output` — the tool-layer LLM gateway documented in
[Tool Contracts §7](tool-contracts.md#7-the-llm-gateway--where-judgment-engines-actually-call-a-model)
— before any provider call, and raises `TokenBudgetExceeded` rather than
silently truncating or sending an oversized request. This is the backstop
behind every formatter's own judgment: formatters decide what *should*
reasonably fit; the token budget is the hard ceiling that catches anything
that doesn't, loudly, before it ever reaches a provider.

---

## 11. Unwired Infrastructure — track_agent_tokens and the Real Source of Truth

Not every piece of memory-adjacent infrastructure in this codebase is actually
in use, and it's worth being honest about which one isn't. `track_agent_tokens`
(`src/core/llm_token_tracker/tracking_context.py`) is a context manager that
estimates and logs input tokens around an execution boundary — but its own
docstring states plainly: *"this helper is not wired into any production
`src/` call site today... it is therefore optional infrastructure, not an
active part of the pipeline."* The same docstring names where the real answer
to "how many tokens did this call actually use" lives instead: `src/observability`,
where a LiteLLM-to-LangSmith callback captures the provider's own reported
prompt/completion tokens, cost, and latency for real calls — not a local
estimate computed before the call is even made. This distinction — estimated,
unused local tooling versus the actual, wired observability pipeline — is
worth knowing before reaching for `track_agent_tokens` expecting it to reflect
production reality; see [Observability](observability.md) for the system that
actually does.

---

## 12. Design Rule — Deciding What Crosses a Memory Boundary

```text
Does this agent's task need this field to do ITS job specifically?
  NO  -> drop it in the formatter's select_*_context function, and say so
         in the module docstring (Section 3).

Is the source object something another agent already produced?
  YES -> pass through only the FINAL decision (Section 7), never the
         deliberation, drafts, or bookkeeping behind it -- unless the
         task IS comparing full artifacts (quality_feedback_formatter
         is the deliberate exception).

Does the field come from a shared upstream analysis (a "blackboard"
object like AlignmentStrategy) that this agent should trust rather
than re-derive?
  YES -> pass that agent's own slice through UNMODIFIED (Section 6).
         Do not re-summarize or re-interpret another agent's analysis.

Are you tempted to cap, rank, or inject a vocabulary hint to "reduce
noise" or "save tokens"?
  STOP -> that exact instinct caused a measured quality regression in
          this codebase once already (Section 5). Prove the cap helps
          with a real run before shipping it, the same way its removal
          was justified by one.

Is the value personally identifiable information?
  YES -> it does not belong in formatter logic at all -- it is handled
         by redaction/rehydration and the PII mapping store (Section 8),
         a stricter boundary than any formatter enforces on its own.
```

---

## 13. Future Considerations

**Whether formatter truncation policies should be revisited as a set, not one
at a time.** The three-formatter case study in Section 5 shows one cap that
was proven wrong by a live run. The other two caps (`achievements[:3]`,
top-6 requirements) have not been shown to be wrong, but they also haven't
been re-examined with the same scrutiny that caught the summary-writer
regression — worth an explicit pass asking the same question of every
existing cap: is this backed by an observed failure mode, or is it an
untested assumption that happens not to have caused visible harm yet?

**Whether the Blackboard Pattern should extend to more shared analysis
objects.** `AlignmentStrategy` is the one clear blackboard object in the
system today — one writer, several narrow readers, each trusting the writer's
own analysis. As the pipeline grows, whether other shared upstream artifacts
(the code-computed match report, for instance) deserve the same explicit
"don't re-derive, only slice and pass through" discipline is worth deciding
deliberately, rather than leaving each new formatter to rediscover the
principle independently.

**Whether unwired infrastructure like `track_agent_tokens` should be removed
or completed.** Optional, disconnected tooling that estimates something the
real observability pipeline already measures more accurately is a
maintenance cost with no corresponding benefit today. Whether it should be
finished and wired in (if local pre-call estimation turns out to serve a
purpose LangSmith's post-call reporting cannot) or removed entirely is an
open architectural question this document surfaces but does not resolve.
