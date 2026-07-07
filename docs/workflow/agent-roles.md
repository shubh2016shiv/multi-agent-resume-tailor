# Agent Roles
## The Bounded-Specialist Model Behind Resume Tailor's Multi-Agent Pipeline

> **Scope:** `src/agents/` — 8 active agent factories, their config surface, and the
> role-boundary contract that connects them to `src/orchestration/`.
> **Audience:** Contributors adding or modifying an agent; reviewers auditing why a
> resume was tailored the way it was.

---

## Table of Contents

1. [What Problem the Role Model Solves](#1-what-problem-the-role-model-solves)
2. [The Core Design Principle — Agents Reason, Code Decides](#2-the-core-design-principle--agents-reason-code-decides)
3. [Anatomy of an Agent Factory — The Common Recipe](#3-anatomy-of-an-agent-factory--the-common-recipe)
4. [The Eight Active Roles — A Field Guide](#4-the-eight-active-roles--a-field-guide)
5. [Tool Access as a Design Lever — Three Tiers of Trust](#5-tool-access-as-a-design-lever--three-tiers-of-trust)
6. [The Code-Computed-Context Pattern — Why Gap Analysis Has No Tools](#6-the-code-computed-context-pattern--why-gap-analysis-has-no-tools)
7. [Post-Hoc Engines — Never Let an Agent Grade Its Own Homework](#7-post-hoc-engines--never-let-an-agent-grade-its-own-homework)
8. [allow_delegation=False — Why the Graph Is the Only Manager](#8-allow_delegationfalse--why-the-graph-is-the-only-manager)
9. [Per-Role Temperature — Tuning Determinism vs. Creativity](#9-per-role-temperature--tuning-determinism-vs-creativity)
10. [The Role Boundary Contract — From Node to State](#10-the-role-boundary-contract--from-node-to-state)
11. [Case Study: How Experience Optimization Actually Runs](#11-case-study-how-experience-optimization-actually-runs)
12. [Configuration Surface — agents.yaml and tasks.yaml](#12-configuration-surface--agentsyaml-and-tasksyaml)
13. [Adding a New Role — The Checklist](#13-adding-a-new-role--the-checklist)
14. [Future Considerations](#14-future-considerations)

---

## 1. What Problem the Role Model Solves

A resume-tailoring system carries a truthfulness risk that most agentic applications
do not. If a code-review agent hallucinates, the compiler catches it. If a customer
support agent overreaches, a human reads the transcript. If a resume-tailoring agent
invents a metric, fabricates a tool the candidate never used, or upgrades "supported
migration" to "led migration," the artifact that ships is a lie told on the
candidate's behalf, submitted to an employer, and difficult to walk back.

The obvious shortcut — one general-purpose "resume agent" with a big system prompt
and a bag of tools — is tempting precisely because it is easy to build. It is also
the wrong shape for this problem. A single agent that extracts, analyzes, rewrites,
assembles, and grades its own resume gives you no seam at which to intervene. When
the final resume overclaims, omits a section, or fails to render, a monolithic agent
gives you one crime scene with every fingerprint smeared into every other one. You
cannot say which step produced the defect because there was only one step.

Resume Tailor answers this by refusing to build a manager agent at all. Eight narrow
agents each do one bounded job, invoked one at a time by LangGraph nodes that never
run concurrently against the same state field. Each agent takes a formatted context,
returns a typed Pydantic object, and hands control back. Nothing about that agent's
reasoning is trusted to also validate itself, decide what runs next, or delegate to
another agent. The result is a system where every field in the final resume can be
traced to exactly one producing stage — because exactly one stage was allowed to
write it.

---

## 2. The Core Design Principle — Agents Reason, Code Decides

Every agent in this codebase drafts. No agent decides. The distinction sounds like a
platitude until you look at what "deciding" means in practice: choosing whether a
draft is good enough to ship, choosing whether a resume passes ATS compliance,
choosing whether a bullet's claim is supported by the source resume, choosing
whether the pipeline should pause for a human. All four of those decisions are made
by plain Python — deterministic engines, Pydantic validators, and LangGraph
conditional edges — never by an LLM call sitting inside an agent's own turn.

```text
   +----------------------+
   |  orchestration node  |   formats typed state into a prompt-ready context
   +----------------------+
              |
              v
   +----------------------+
   |     CrewAI agent      |   runs ONE task, may consult read-only tools
   +----------------------+
              |
              v
   +----------------------+
   |  Pydantic validation  |   output_pydantic=<Model> — malformed output never
   +----------------------+   reaches the next node
              |
              v
   +----------------------+
   |  code-owned engines   |   deterministic checks: truthfulness floor,
   |  (never inside agent) |   ATS compliance, coverage stats, quality scoring
   +----------------------+
              |
              v
   +----------------------+
   |  node writes state    |   one or more typed fields merged into
   +----------------------+   ResumeEnhancementPipelineState
```

The agent's turn ends the moment it returns its typed draft. Everything that decides
whether that draft is trustworthy, complete, or ready to move forward happens after
the agent has left the building. This is the single idea that every other section in
this document elaborates on.

---

## 3. Anatomy of an Agent Factory — The Common Recipe

All eight files under `src/agents/<role>/agent.py` are structurally identical. Once
you have read one, you have read the skeleton of all of them — only the config key,
the tool list, and the temperature differ. Every factory follows the same four steps:

```text
STEP 1: LOAD CONFIG AND BUILD THE LLM INSTANCE
        config = load_agent_config("<agent_config_key>")
        llm_instance = LLM(model=config["llm"], temperature=config.get("temperature", <default>))

STEP 2: ASSEMBLE TOOLS AND RUNTIME DEFAULTS
        defaults = get_config().llm.agent_defaults
        tools = [...]   # empty, read-only audit tools, or document-I/O tools

STEP 3: BUILD THE AGENT
        Agent(
            role=..., goal=..., backstory=...,     # from YAML, never hardcoded
            llm=llm_instance,
            allow_delegation=False,                 # true for all 8 -- see Section 8
            tools=tools,
            max_retry_limit=defaults.max_retry_limit,
            max_rpm=defaults.max_rpm,
            max_iter=defaults.max_iter,
            max_execution_time=defaults.max_execution_time,
            respect_context_window=defaults.respect_context_window,
        )

STEP 4: LOG AND RETURN
```

Two things about this recipe are worth dwelling on.

**Config validation happens before the LLM is ever constructed.**
`src/agents/agent_config.py` defines `load_agent_config(name)`, which pulls the
matching block out of `agents.yaml` and checks it against a fixed
`_REQUIRED_FIELDS = ["role", "goal", "backstory", "llm"]`. If a field is missing,
the factory raises `RuntimeError` immediately, with a message naming the exact
missing keys and the file to fix. An agent is never built halfway — either every
required field resolves, or construction fails loudly at startup rather than
producing an agent with an empty backstory that quietly degrades output quality.

**Every agent shares one resilience envelope.**
`agent_defaults` lives in `src/config/settings.yaml`, not per-agent:

```yaml
agent_defaults:
  max_retry_limit: 3           # retries when the agent encounters an error
  max_rpm: 60                  # requests per minute, rate limiting
  max_iter: 15                 # max iterations before forcing a best answer
  max_execution_time: 300      # seconds per task (5 minutes)
  respect_context_window: true # auto-summarize to avoid context overflow
```

Whether an agent is the zero-tool Gap Analysis Specialist or the four-tool Resume
Extractor, it retries the same number of times, is capped at the same iteration
count, and respects the same wall-clock budget. Resilience is a property of the
platform, not a per-role tuning knob — the only per-role knobs are the persona
(YAML), the tools, and the temperature.

---

## 4. The Eight Active Roles — A Field Guide

```text
                       +----------------------------+
                       |   LangGraph orchestration   |
                       +----------------------------+
                                    |
        +--------------+-----------+-----------+---------------+
        |              |                       |               |
        v              v                       v               v
  resume_content   job_description       gap_analysis      (fan-out to 3
   _extractor        _analyst            _specialist        content agents)
        |              |                       |
        +------+-------+                       |
               |                                v
               +------------------->  professional_summary_writer
               |                      experience_section_optimizer
               |                      skills_section_strategist
               |                                |
               |                                v
               +------------------->  ats_optimization_specialist
                                                 |
                                                 v
                                     quality_feedback_reviewer
                                        (advisory, non-gating)
```

### 4.1 Resume Content Extractor

- **Factory:** `create_resume_extractor_agent` (`src/agents/resume_parser/agent.py`)
- **Config key:** `resume_content_extractor` · **Temperature:** `0.0`
- **Output contract:** `Resume`
- **Tools:** up to 4, the richest tool set of any role — `convert_resume_document_to_markdown`,
  `check_resume_markdown_quality`, `redact_pii_from_resume_markdown` (conditional),
  `extract_structured_resume_from_markdown`.

This is the only agent in the system that touches a raw document. Its tool
sequence — convert, audit, optionally redact, extract — is fixed by the order the
tools are listed in, not left to the agent to improvise. The PII redaction tool is
the most interesting design detail: `build_resume_ingestion_tools(enable_pii_redaction)`
does not pass a flag *into* the tool, it decides whether the tool exists in the
agent's tool list at all. When the feature flag is off, the agent has no
redaction affordance to reach for — the capability is absent, not merely unused.
Temperature `0.0` reflects that extraction is a transcription task: there is a
right structured answer, and the agent should not editorialize while producing it.

### 4.2 Job Description Analyst

- **Factory:** `create_job_analyzer_agent` (`src/agents/job_description_analyser/agent.py`)
- **Config key:** `job_description_analyst` · **Temperature:** `0.0`
- **Output contract:** `JobDescription`
- **Tools:** none.

The simplest agent in the system. The job posting is converted to Markdown *before*
the task is created, so the text simply arrives in the agent's context — there is
no tool call for the agent to make, no ambiguity about which document to parse. Its
entire job is disciplined structured extraction, shaped by the persona and task
instructions in `agents.yaml`/`tasks.yaml` rather than by any runtime tool.

### 4.3 Gap Analysis Specialist

- **Factory:** `create_gap_analysis_agent` (`src/agents/gap_analysis/agent.py`)
- **Config key:** `gap_analysis_specialist` · **Temperature:** `0.3`
- **Output contract:** `AlignmentStrategy`
- **Tools:** none — deliberately. See [Section 6](#6-the-code-computed-context-pattern--why-gap-analysis-has-no-tools).

This agent writes the single plan of record that the three downstream
content-generation agents will follow without re-analyzing anything themselves. Its
goal statement in `agents.yaml` says this outright: *"You do the analysis ONCE; the
Summary Writer, Experience Optimizer, and Skills Optimizer each read your guidance
and never re-analyze."* Its backstory encodes three testable habits — ground every
match/gap/keyword in the supplied match report, anchor the fit score to that
report's score rather than recomputing it, and write guidance that respects each
downstream agent's real limits (the Experience Optimizer may only reorder bullets
verbatim; the Skills Optimizer may add only evidence-backed skills). Temperature
`0.3` is a deliberate step up from the `0.0` extraction roles: synthesizing
executable guidance from facts requires more latitude than transcribing facts, but
still far less than drafting prose.

### 4.4 Professional Summary Writer

- **Factory:** `create_professional_summary_agent` (`src/agents/professional_summary/agent.py`)
- **Config key:** `professional_summary_writer` · **Temperature:** `0.7`
- **Output contract:** `ProfessionalSummary`
- **Tools:** `audit_summary` — a hybrid tool blending mechanical checks (length,
  first-person voice, boilerplate detection) with LLM judgment.

The highest temperature of any agent in the system, and intentionally so: this
agent's job is to generate four summary drafts using different narrative
frameworks, self-critique each one, and recommend the strongest. That is a
creative-search task, not a fact-transcription task, and the temperature reflects
it. The `audit_summary` tool is consulted *during* the agent's own reasoning loop —
it is a quality mirror the agent can hold up to its own drafts, not a gate that
blocks output.

### 4.5 Experience Section Optimizer

- **Factory:** `create_professional_experience_agent` (`src/agents/professional_experience/agent.py`)
- **Config key:** `experience_section_optimizer` · **Temperature:** `0.0`
- **Output contract:** `ExperienceRewriteProposal` internally, merged into `OptimizedExperienceSection`
- **Tools:** none. Audit, repair-decisioning, ID restoration, and merge are all
  orchestration-owned, not agent-owned.

This is the strictest-contract agent in the system, and the only one invoked
**per work-experience entry** rather than once per resume — `src/orchestration/nodes/experience.py`
fans out one CrewAI task per role in parallel (`ThreadPoolExecutor`, capped at 4
workers) and merges the per-role results afterward. Its rewrite contract is
1-to-1: same bullet count, same bullet order, verbatim identity per bullet, no
invented figure. Temperature `0.0` matches its purpose — it rewrites language, it
does not invent content. The full mechanics of this role's truth floor and repair
loop are detailed in [Section 11](#11-case-study-how-experience-optimization-actually-runs),
because they are the richest illustration of "agents reason, code decides" anywhere
in the codebase.

### 4.6 Skills Section Strategist

- **Factory:** `create_skill_optimizer_agent` (`src/agents/skill_optimizer/agent.py`)
- **Config key:** `skills_section_strategist` · **Temperature:** `0.4`
- **Output contract:** `OptimizedSkillsSection`
- **Tools:** none — the agent's own docstring is explicit that "all quality checks
  run code-owned on typed output after the agent finishes, not through agent tool
  calls."

The agent reorders, categorizes, and prioritizes skills according to the Gap
Analysis Specialist's `skills_guidance`. Whether every listed skill is actually
evidenced in the resume is checked afterward by the `check_skills_evidence` engine
in `src/tools/truthfulness/` — the agent is not trusted to certify its own
evidence trail while assembling the list. Temperature `0.4` sits between the
extraction roles and the summary writer: reordering and grouping require judgment,
but not narrative creativity.

### 4.7 ATS Optimization Specialist

- **Factory:** `create_ats_optimizer_agent` (`src/agents/ats_optimizer/agent.py`)
- **Config key:** `ats_optimization_specialist` · **Temperature:** `0.1`
- **Output contract:** `AtsOptimizedResume`
- **Tools:** `validate_ats_compliance`, `analyze_jd_keyword_coverage` — both
  read-only, consulted *while reasoning*, never mutating.

This agent assembles the optimized summary, experience, and skills sections (plus
contact/education carried over from the original resume) into one coherent,
ATS-aligned `Resume`. It is deliberately not the same thing as a validator: its
output is the assembled resume plus decision notes, not a self-scored compliance
report. The two tools let it check its own assembly work as it goes, but the
authoritative measurement happens afterward in `ats_optimizer/engines.py` — see
[Section 7](#7-post-hoc-engines--never-let-an-agent-grade-its-own-homework).
Temperature `0.1` is near-deterministic because assembly is closer to a mechanical
merge than to composition.

### 4.8 Quality Feedback Reviewer

- **Factory:** `create_quality_feedback_agent` (`src/agents/quality_feedback/agent.py`)
- **Config key:** `quality_feedback_reviewer` · **Temperature:** `0.2`
- **Output contract:** `QualityFeedback`
- **Tools:** `audit_truthfulness`, `analyze_jd_keyword_coverage` — and, pointedly,
  **not** `validate_ats_compliance`.

This agent audits the tailored resume and writes advisory feedback — narrative
only. Its module docstring states the omission of the ATS tool as a design
decision, not an oversight: ATS compatibility can only be measured meaningfully on
the *real rendered artifact* (the actual `.tex` file), which this agent never sees.
It only has the lossy in-context text representation. Giving it the ATS tool would
let it produce a self-certified compliance opinion that the code-owned
`resume_quality_evaluation.evaluate_rendered_structure` — which inspects the
rendered file directly — would discard anyway. Rather than let that false
confidence sit in the agent's own reasoning trace, the tool was removed outright.
This agent cannot write scores and cannot make the release decision; that gate is
entirely code-owned, downstream of this agent's output.

---

## 5. Tool Access as a Design Lever — Three Tiers of Trust

Reading the eight roles side by side reveals that tool access is not incidental —
it is a deliberate signal of how much the system trusts an agent to touch anything
beyond its own reasoning.

```text
TIER 0 — Reason-only, zero tools
  job_description_analyst · gap_analysis_specialist ·
  experience_section_optimizer · skills_section_strategist
  --------------------------------------------------------
  These agents receive everything they need already assembled in context by a
  node/formatter. They cannot call out to anything mid-turn. Their entire
  contribution is the reasoning step itself.

TIER 1 — Read-only audit tools, consulted while reasoning, never mutate
  professional_summary_writer (audit_summary) ·
  ats_optimization_specialist (validate_ats_compliance, analyze_jd_keyword_coverage) ·
  quality_feedback_reviewer (audit_truthfulness, analyze_jd_keyword_coverage)
  --------------------------------------------------------
  These agents can hold a mirror up to their own draft mid-turn. The tools report
  findings; they never rewrite anything on the agent's behalf.

TIER 2 — Document I/O, the only role that touches raw files
  resume_content_extractor (convert, quality-check, redact, extract)
  --------------------------------------------------------
  The single agent trusted with file-format conversion and PII handling. No other
  role is wired with anything resembling a mutation-adjacent tool.
```

Nowhere in the system does an agent hold a tool that writes the final artifact,
grades a hard pass/fail gate, or calls another agent. The tiers rise only as high
as "read and report" — the write path, the gate, and the routing decision are
reserved for code and for the graph.

---

## 6. The Code-Computed-Context Pattern — Why Gap Analysis Has No Tools

The Gap Analysis Specialist is the clearest example of a pattern used throughout
the pipeline: when a fact can be computed deterministically, compute it in Python
*before* the agent runs, and hand it to the agent as ground truth in its context —
rather than giving the agent a tool it could call to compute the same fact itself
mid-turn.

```text
   ALTERNATIVE REJECTED:                    ACTUAL DESIGN:
   agent calls a "match" tool                strategy node runs match_resume_to_job()
   mid-turn, must re-serialize                in code BEFORE the agent is invoked
   the resume to call it                              |
        |                                              v
        v                                    result rendered into context as
   risk: agent works from its                 `current_match_report` (score +
   own lossy TOON view, tool                   per-requirement findings)
   result may drift from what                          |
   the agent already believes                          v
                                              agent instructed: "treat as fact,
                                              do not recompute" (task description,
                                              gap_analysis tasks.yaml)
```

The task description for `create_alignment_strategy_task` is explicit about this:
*"current_match_report: the CODE-COMPUTED ground truth... Treat current_match_report
as fact. Do not contradict it or recompute it; build on it."* This removes an entire
class of failure — an LLM re-deriving a match score by ad hoc arithmetic and getting
a different, unreliable number from the one code already computed correctly.

The commented-out superseded version of this same YAML file, still preserved inline
in `src/config/tasks/gap_analysis.yaml`, is worth reading as a historical artifact:
the prior prompt asked the LLM to hand-compute `overall_fit_score` via a
40/30/20/10 weighted formula and demanded "at least 15 keywords." Both requirements
were removed in favor of anchoring to the code-computed score and integrating only
as many keywords as the match report actually supports — because arithmetic
performed by an LLM is unreliable, and a fixed keyword floor pressures the agent
into padding with unsupported terms. The file keeps this diff as a comment with a
dated rationale header rather than deleting it — prompts are treated as versioned,
reviewable artifacts with institutional memory, not disposable strings.

---

## 7. Post-Hoc Engines — Never Let an Agent Grade Its Own Homework

Several agent packages ship a sibling `engines.py` alongside `agent.py`. These
modules are never imported by the agent's factory and never called during the
agent's turn — their docstrings say so explicitly. They exist to measure an
agent's already-produced, already-validated Pydantic output *after the fact*, for
tests, monitoring, and QA gates.

```text
   agent.py                         engines.py
   ---------                        ----------
   builds the CrewAI Agent          check_strategy_quality(strategy) -> dict
   agent runs, returns               (score 0-100, issues, warnings — flags things
   typed Pydantic output             like "high fit score but many gaps")

                                     calculate_coverage_stats(strategy) -> dict
                                      (total_matches, total_gaps, coverage_ratio)

   [ never called here ]  <-- x -->  [ called by tests / monitoring / QA gate ]
```

Both `gap_analysis/engines.py` and `ats_optimizer/engines.py` follow the identical
contract: pure functions, Pydantic model in, plain `dict` out, no LLM call, no
framework dependency, no I/O beyond what's already been validated. `check_ats_quality`
in `ats_optimizer/engines.py` is the sharpest example — it does not trust the
agent's own read of its assembly. It **renders the assembled resume to actual
text** (`render_resume`) and re-runs three purely mechanical engines
(`audit_ats_formatting`, `audit_section_headers`, `analyze_keyword_coverage`)
against that real, rendered artifact, then collects every `BLOCKER`/`MAJOR`
finding into a `serious_findings` list. The agent's two Tier-1 tools
(`validate_ats_compliance`, `analyze_jd_keyword_coverage`) let it check its own
work mid-turn; `check_ats_quality` is the independent, code-owned re-measurement
that actually decides whether the output is safe to ship.

---

## 8. allow_delegation=False — Why the Graph Is the Only Manager

Every one of the eight `Agent(...)` constructions sets `allow_delegation=False`
explicitly. CrewAI supports agent-to-agent delegation — one agent instantiating a
task on another agent conversationally, mid-reasoning. This project switches that
capability off, project-wide, without exception.

The consequence is structural, not cosmetic. If delegation were allowed, an
Experience Optimizer agent could, in principle, decide mid-turn to hand a
sub-problem to the ATS Optimizer, or the Gap Analysis Specialist could delegate
part of its synthesis back to a content agent. Any of that would mean control flow
lived partly inside an agent's own non-deterministic reasoning trace — invisible to
the graph, unloggable as a routing decision, and impossible to checkpoint cleanly.

With delegation off, the **only** place control transfers from one role to another
is `src/orchestration/graph.py` — a `StateGraph` whose edges (including the
conditional routing functions `_route_after_ats_check`, `_route_after_quality`, and
`_route_after_candidate_clarifications`) are the sole legal mechanism for moving
from one agent's territory to another's. This is precisely what makes checkpoints,
human-in-the-loop pauses, and partial reruns possible without touching a single
agent's internals: because the graph is the only manager, resuming or rerouting a
run is a graph operation, not an agent-prompting problem.

---

## 9. Per-Role Temperature — Tuning Determinism vs. Creativity

Reading the eight `temperature` values together as a set reveals a spectrum, not
arbitrary per-agent tuning:

| Role | Temperature | Why |
|---|---|---|
| Resume Content Extractor | `0.0` | Transcription — a right structured answer exists |
| Job Description Analyst | `0.0` | Transcription — same reasoning |
| Experience Section Optimizer | `0.0` | Rewrites language only; must not invent content |
| ATS Optimization Specialist | `0.1` | Assembly is closer to a mechanical merge than composition |
| Quality Feedback Reviewer | `0.2` | Narrative audit, but grounded in tool findings |
| Gap Analysis Specialist | `0.3` | Synthesizing executable guidance from fixed facts |
| Skills Section Strategist | `0.4` | Reordering/grouping requires judgment, not narrative creativity |
| Professional Summary Writer | `0.7` | Drafts 4 narrative variations and self-critiques — a creative search task |

The pattern: temperature rises in step with how much of a role's job is
**generating options** versus **transforming already-decided facts**. Roles that
sit closest to the truthfulness boundary (extraction, verbatim rewriting) sit at
`0.0`. The one role whose entire job is creative variation (the summary writer)
sits highest, and even then is bounded by a code-owned `audit_summary` tool it
consults during its own turn.

---

## 10. The Role Boundary Contract — From Node to State

```text
   node formats context             (src/formatters/, src/orchestration/nodes/)
           |
           v
   CrewAI agent runs one task       (allow_delegation=False; tools per Section 5)
           |
           v
   Pydantic output is validated     (Task output_pydantic=<Model>)
           |
           v
   node writes field(s) into        (partial dict; LangGraph merges it into
   ResumeEnhancementPipelineState    ResumeEnhancementPipelineState — see state.py)
```

`src/orchestration/state.py` defines `ResumeEnhancementPipelineState` as a
`TypedDict` where every field starts `None`. A node sets its own output field(s)
and returns a partial dict; LangGraph merges that dict back into the shared state.
The comment at the top of `state.py` states the invariant plainly: *"Downstream
nodes must only read a field after the node that produces it has run."* The graph
topology in `graph.py` is what enforces this read order — it is not a convention
contributors have to remember, it is a structural guarantee of which nodes have
edges into which.

This boundary is why the system can support checkpoints, human clarification
pauses, quality gates, and partial reruns without changing its overall shape. A
checkpoint is just a snapshot of this typed dict. A human-in-the-loop pause is just
an `interrupt()` call inside one node, resumed by re-invoking the graph with new
state fields. None of that machinery has to know anything about what happens
*inside* any given agent's reasoning — it only has to know the state contract.

---

## 11. Case Study: How Experience Optimization Actually Runs

The Experience Section Optimizer is the single richest illustration of "agents
reason, code decides" in the entire codebase, because its orchestration node
(`src/orchestration/nodes/experience.py`) implements a full truth-floor-then-repair
loop around every agent call.

```text
   resume.work_experience  (N entries)
           |
           v
   fan out: one role-scoped CrewAI task per entry, in parallel
   (ThreadPoolExecutor, max_workers = min(N, 4))
           |
           v
   +-------------------------------------------------------------+
   |  PER ROLE:                                                   |
   |                                                               |
   |  1. agent proposes ExperienceRewriteProposal                  |
   |     (rewritten_bullets: 1 per source bullet, same bullet_id)  |
   |                                                               |
   |  2. TRUTH FLOOR (deterministic, no LLM, non-negotiable):      |
   |     - bullet count parity                                     |
   |     - bullet_id / order preserved                              |
   |     - detect_claim_inflation: any NEW figure vs. the source?  |
   |                                                                |
   |  3. QUALITY REVIEW (LLM-scored, best-effort):                 |
   |     audit_experience_rewrite_quality -- unsupported            |
   |     specificity, ownership inflation, vague accomplishments,  |
   |     brochure tone, JD-keyword decoration                      |
   |                                                                |
   |  4. if truth floor clears AND no BLOCKER/MAJOR quality issue: |
   |        SHIP the rewrite as-is                                |
   |     else:                                                     |
   |        spend the ONE allowed repair call, feeding back every  |
   |        finding at once                                        |
   |     if repair still fails the truth floor:                    |
   |        SHIP the untouched SOURCE bullets instead              |
   |        (never a truthfulness-failing rewrite)                 |
   |                                                                |
   |  5. any bullet that is truthful but too thin to be concrete   |
   |     without inventing facts -> clarifying_question captured   |
   +-------------------------------------------------------------+
           |
           v
   merge all N role results into one OptimizedExperienceSection
           |
           v
   if any role raised clarifying questions:
       graph pauses at await_candidate_clarifications (interrupt())
       -- resumed later with clarification_answers, re-running only
          the affected roles' rewrite once more
```

Several details here are worth calling out because they generalize the whole
philosophy of this document into one concrete flow:

- **The rewrite's writable surface is one field.** `_rebuild_rewritten_role_from_proposal`
  keeps only the `achievements` text the LLM produced; company, dates, description,
  and `skills_used` are always copied verbatim from the source `Experience` object,
  whatever the LLM's proposal contains. The agent cannot rewrite what it was never
  given the power to touch.
- **Truthfulness and quality are enforced with different force.** The truth floor
  (bullet count, ID parity, no new figures) is non-negotiable and requires zero LLM
  judgment to check — it is closed, countable fact. Substance quality is
  best-effort: a thin-but-truthful bullet earns one repair attempt, and if it is
  still thin afterward, the system surfaces it to the *candidate* as a question
  rather than inventing a number to make it read better.
- **The repair budget is exactly one call.** There is no retry loop searching for a
  passing draft. One repair, then ship the best truthful version available — which,
  in the worst case, is the original bullets, untouched.
- **The clarification loop is a real pause, not an offline note.** `interrupt()` is
  a genuine LangGraph mechanism that halts graph execution until the run is resumed
  with candidate answers; those answers are folded into the role's evidence before
  the next rewrite pass, so a fact the candidate supplies becomes admissible source
  evidence rather than something the truth floor would flag as invented.

This one node demonstrates every idea in this document at once: an agent that
drafts, code that owns the truth floor, a bounded and disclosed repair budget, and
a human-in-the-loop boundary that is a graph interrupt rather than a side channel.

---

## 12. Configuration Surface — agents.yaml and tasks.yaml

Every persona lives in `src/config/agents/<role>.yaml`, keyed by the config name
the factory passes to `load_agent_config`:

```yaml
gap_analysis_specialist:
  role: "Resume-to-Job Alignment Strategist (domain-agnostic)"
  goal: >
    Turn a code-computed match report into one truthful alignment strategy...
  backstory: >
    You read the supplied match report... and build on it rather than
    second-guessing it. You hold three habits: ...
  llm: openai/gpt-4o
  temperature: 0.3
  verbose: true
```

Every task lives in the matching `src/config/tasks/<role>.yaml`, associated back
to its agent by name (`agent: gap_analysis_specialist`), and carrying its own
`description` (the procedure, written as an ordered list the agent is meant to
follow field-by-field) and `expected_output` (the shape the Pydantic model
enforces regardless).

Both files in this project are notable for keeping **superseded prompt versions as
commented-out blocks with a dated rationale header**, rather than relying on git
history alone. `src/config/tasks/gap_analysis.yaml` documents, inline, exactly why
the previous prompt was replaced: it read context keys the formatter no longer
emits, hand-computed a fit score the code now provides authoritatively, forced an
arbitrary 15-keyword minimum that pressured padding, and used language that pushed
the reorder-only Experience Optimizer to violate its own verbatim contract. Treating
prompts this way — versioned, diffed, and reasoned about in-place — is itself part
of the architecture: a prompt regression is exactly as reviewable as a code
regression.

---

## 13. Adding a New Role — The Checklist

Codified in `.claude/rules/agents/RULES.md` and consistent with every existing
role:

1. Create `src/agents/<agent_name>/agent.py` following the four-step recipe in
   [Section 3](#3-anatomy-of-an-agent-factory--the-common-recipe).
2. Add the persona block to `src/config/agents/<agent_name>.yaml` — `role`, `goal`,
   `backstory`, `llm` are mandatory; `temperature` and `verbose` are optional with
   sane defaults.
3. Add the task block to `src/config/tasks/<agent_name>.yaml`, referencing the
   agent by config key.
4. Register the factory in `src/agents/__init__.py`'s `__all__` — this is the
   only import surface other modules should use.
5. Wire the new agent into a node in `src/orchestration/nodes/`, and add it (and
   its edges) to `build_resume_enhancement_graph` in `src/orchestration/graph.py`.
6. Decide its tool tier deliberately (Section 5) — default to Tier 0 (no tools)
   unless the role genuinely needs to consult a read-only audit mid-turn, or
   touches raw documents.
7. If the role's output benefits from independent, code-owned measurement, add a
   sibling `engines.py` with pure functions — never call it from inside the
   agent's own task.

---

## 14. Future Considerations

**Extending independent, code-owned measurement to every role.** Only Gap Analysis
and ATS Optimization currently have a sibling code-owned validation layer sitting
outside the agent's own turn. Professional Summary and Skills Optimizer already
have read-only audit tools the agent consults mid-turn, but no equivalent
after-the-fact, independent measurement of their output the way Gap Analysis and
ATS Optimization have. Closing that gap would make the "never let an agent grade
its own homework" guarantee uniform across all eight roles rather than concentrated
in two of them.

**Whether the three-tier tool-trust model needs a fourth tier.** Today no agent
holds a tool that writes the final artifact — the tiers rise only as high as
"read and report." If the system ever grows a role that legitimately needs to
produce a file or artifact directly (rather than a typed draft a node renders
downstream), that would be a new tier, and it deserves the same explicit,
documented trust boundary the current three tiers have — not a quiet exception
bolted onto an existing role.
