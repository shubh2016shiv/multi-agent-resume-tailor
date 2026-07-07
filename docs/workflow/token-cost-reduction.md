# Token Cost Reduction
## Where the Money Actually Goes, and the Discipline That Keeps It Down

> **Scope:** `src/formatters/` (the cost-shaping layer), per-role model selection
> in `src/config/agents/*.yaml`, the response cache and token budget guard, and
> a documented historical incident that explains why this architecture looks
> the way it does.
> **Audience:** Contributors adding a new field to an agent's context; anyone
> asking "why does this pipeline cost what it costs, and where would a new
> inefficiency most likely hide?"

---

## Table of Contents

1. [What Problem Token Cost Reduction Solves](#1-what-problem-token-cost-reduction-solves)
2. [Two Different Concerns That Must Not Be Conflated](#2-two-different-concerns-that-must-not-be-conflated)
3. [Origin Story: The Skills Optimizer Bottleneck](#3-origin-story-the-skills-optimizer-bottleneck)
4. [What Persists Today: The Formatter Discipline This Incident Produced](#4-what-persists-today-the-formatter-discipline-this-incident-produced)
5. [Per-Role Model Selection — A Real, Active Cost Lever](#5-per-role-model-selection--a-real-active-cost-lever)
6. [The Response Cache — Free Repetition, Not Compression](#6-the-response-cache--free-repetition-not-compression)
7. [The Token Budget Guard — a Ceiling, Not a Reduction Strategy](#7-the-token-budget-guard--a-ceiling-not-a-reduction-strategy)
8. [A Feature Flag That No Longer Gates Anything](#8-a-feature-flag-that-no-longer-gates-anything)
9. [The One Role With an Output Cap — an Asymmetry Worth Naming](#9-the-one-role-with-an-output-cap--an-asymmetry-worth-naming)
10. [Quality Link — Why Smaller Context Also Means Better Output](#10-quality-link--why-smaller-context-also-means-better-output)
11. [Design Rule — Where to Look Before Adding a New Field to Context](#11-design-rule--where-to-look-before-adding-a-new-field-to-context)
12. [Future Considerations](#12-future-considerations)

---

## 1. What Problem Token Cost Reduction Solves

A single resume-tailoring run is not one LLM call — it is at least eight,
often more once the per-role experience fan-out documented in
[Orchestration Graph §4](orchestration-graph.md#4-stage-by-stage-walkthrough)
is counted per work-experience entry. Every field that formatter code decides
to include in any one of those calls is paid for on every run, for every
candidate, indefinitely — an unnecessary field is not a one-time cost, it is
a recurring one that scales with usage. Worse, as
[Memory Boundaries §5](memory-boundaries.md#5-case-study-three-formatters-three-different-truncation-policies)
documents in its own case study, unnecessary context is not merely wasteful —
it can actively degrade output quality. Token cost reduction in this system
is therefore not a cost-center afterthought; it is the same discipline that
produces better resumes, arrived at from the economics side instead of the
quality side.

---

## 2. Two Different Concerns That Must Not Be Conflated

This codebase's own documentation set addresses token efficiency at two
entirely different layers, and it's worth being precise about which one this
document covers:

```text
   DEVELOPER / CODING-SESSION TOKEN COST         RUNTIME PIPELINE TOKEN COST
   (docs/graphify-guide.md,                       (THIS document)
    docs/token-efficiency-guide.md)
   -----------------------------------             -----------------------------
   how much it costs an AI coding                  how much it costs the
   assistant (Claude Code, Codex, etc.)             resume-tailoring PIPELINE
   to navigate and understand THIS                 ITSELF to run -- the eight-plus
   repository while a developer works on it         LLM calls every resume makes

   solved by: graphify's knowledge graph,           solved by: formatter-level
   session discipline, prompt caching for           context curation, per-role
   the CODING assistant's own context               model selection, response
                                                     caching, and token budgets
                                                     for the PIPELINE's own calls
```

Graphify reduces how many tokens a coding assistant burns exploring this
codebase's files — it has no effect whatsoever on what a candidate's resume
costs to tailor. Conflating the two would be a category error: one is about
the cost of *working on* this system, the other is about the cost of
*running* it. Everything from here on is about the second.

---

## 3. Origin Story: The Skills Optimizer Bottleneck

`docs/optimization_achievements.md` documents a dated (November 2024), fully
concrete incident that is worth understanding in detail, because the
formatter architecture this document's other sections describe as *today's*
behavior is the direct, lasting consequence of what this incident found. The
model roster in that document (Gemini 2.5 Flash / Flash Lite) predates the
current per-role model selection covered in
[Section 5](#5-per-role-model-selection--a-real-active-cost-lever), so its
dollar figures should be read as historical evidence of *why* this
discipline exists, not as today's live pricing.

### 3.1 The Shocking Discovery

With token tracking enabled, one agent stood out sharply from the rest:

```text
   Agent: Skills Optimization Specialist
   Input Tokens:  8,984  (63% of total)
   Output Tokens: 5,197  (37% of total)
   Total Tokens:  14,181
   Cost per Call: $18.87

   For comparison, a typical agent in the same system consumed
   4,000-5,000 tokens total. The Skills Optimizer was using
   nearly THREE TIMES that amount, for a task — reorganizing an
   already-known list of skills — that should have been one of
   the cheapest calls in the whole pipeline.
```

Extrapolated to even moderate development usage, this one agent alone
projected to nearly $2,000/month in cost — making it, by a wide margin, the
single highest-priority target for optimization.

### 3.2 Five Root Causes, Named Precisely

Investigation found the bottleneck was not one bug but five independent,
stacking inefficiencies, each worth naming because each is a distinct failure
mode any future formatter could reintroduce:

```text
1. UNNECESSARY METADATA -- the formatter sent every field on the Skill
   model, including justification, evidence quotes, and a confidence_score
   that only the Resume Extractor agent (not the Skills Optimizer) ever
   needed. ~1,200 tokens for a 45-skill resume, spent on fields the
   receiving agent had no use for at all.

2. REDUNDANT JOB-REQUIREMENT TEXT -- full requirement sentences (150+
   characters each) were repeated per matched skill, when the agent only
   needed to know WHICH skill matched and HOW STRONGLY. ~800 tokens for
   20 matches.

3. VERBOSE GAP SUGGESTIONS -- every skill gap carried a 200+ character
   human-readable suggestion meant for a person reading a report, not an
   agent deciding what to write. ~600 tokens for 10 gaps.

4. NO OUTPUT LENGTH CONSTRAINT -- the agent's own config set no cap on
   how much it could write, so a structured JSON task produced lengthy
   free-text justification for every decision. This single gap accounted
   for roughly HALF of the total waste on its own -- 2,500 of the
   eventual 5,100 tokens saved in the first optimization pass.

5. DUPLICATE KEYWORDS -- the same keyword (e.g. "Python", "Docker")
   frequently appeared in both the strategy's keyword list and the job's
   own ATS keyword list, so the agent read it twice for no reason.
   ~300 tokens.
```

### 3.3 The Zombie Parameter — a Second Bottleneck in the Experience Optimizer

A second, independently-discovered inefficiency lived in the Experience
Optimizer's evaluation tool: its signature accepted a `strategy_json`
parameter — roughly 1,500 tokens of full gap-analysis data — that the tool's
own logic parsed and then **never used**. Because this tool ran iteratively
(up to three times per experience entry), the waste compounded: roughly
13,500 wasted tokens per resume from one unused function parameter alone.
The fix was not a smarter formatter — it was deleting the parameter and the
70+ lines of now-pointless parsing logic that existed only to serve it. This
is a distinct lesson from the five above: sometimes the cheapest token isn't
one you compress, it's one you discover nobody was ever reading.

### 3.4 Combined Results

```text
   Original run:              ~33,000 tokens,  ~$25.00
   After all three phases:    ~12,500 tokens,  ~$9.50
   Net improvement:           -62% tokens,     -62% cost, 0% quality loss
```

---

## 4. What Persists Today: The Formatter Discipline This Incident Produced

The five root causes in Section 3.2 map directly, almost one-to-one, onto the
formatter architecture documented in full in
[Memory Boundaries §3](memory-boundaries.md#3-the-formatter-layer--where-memory-boundaries-are-actually-drawn):
every `select_*_context` function in `src/formatters/` exists specifically to
keep only the fields one agent's task needs and explicitly document what it
drops, which is precisely the discipline that was missing when the Skills
Optimizer's formatter sent every field on the `Skill` model regardless of
relevance. The TOON encoding documented in
[Memory Boundaries §4](memory-boundaries.md#4-toon--the-textual-shape-memory-takes)
is a more compact wire format than the verbose JSON this incident's numbers
were measured against, compounding the field-level savings with a
format-level one.

It is worth stating plainly, though, that not every lesson from Section 3.2
generalizes as a blanket rule. [Memory Boundaries §5](memory-boundaries.md#5-case-study-three-formatters-three-different-truncation-policies)
documents a formatter (`professional_summary_formatter`) that tried the exact
same instinct — capping and ranking content to save tokens — and found it
caused a measured *quality* regression, not just a cost saving, because the
writer's own evidence-selection step needed to see everything to do its job
well. The discipline this incident established is "keep only what the
receiving agent's specific task needs" — not "always minimize," which are not
the same rule. Section 3's "unnecessary metadata" and "redundant text" were
genuinely unnecessary for the Skills Optimizer's task; that is a claim about
*that task*, not a universal truncation policy.

---

## 5. Per-Role Model Selection — A Real, Active Cost Lever

Beyond context shaping, this system applies a second, entirely independent
cost lever that the historical incident's own three-layer architecture
foreshadowed: **not every role uses the same model.** The current
`src/config/agents/*.yaml` roster shows a deliberate tiering:

```text
   gpt-4o-mini   resume_content_extractor, job_description_analyst
                 -- the two purely TRANSCRIPTION roles (Agent Roles §4.1,
                    §4.2): a right structured answer exists, and a smaller
                    model handles disciplined extraction reliably

   gpt-4o        gap_analysis_specialist, quality_feedback_reviewer
                 -- synthesis and narrative judgment over already-decided
                    facts

   gpt-4.1       professional_summary_writer, experience_section_optimizer,
                 skills_section_strategist, ats_optimization_specialist
                 -- the four roles doing the heaviest creative or
                    compositional work: drafting, rewriting, assembling
```

This is the same underlying idea as the historical document's three-layer
"Analysis / Generation / QA" model assignment, carried into the current
architecture with different specific models. The lesson generalizes past any
particular model name: match the model's cost and capability to what the
role actually needs to do, rather than defaulting every agent to the most
capable (and most expensive) option available. This complements, rather than
duplicates, the temperature spectrum documented in
[Agent Roles §9](agent-roles.md#9-per-role-temperature--tuning-determinism-vs-creativity)
— temperature tunes *how* a model behaves for a given role; model selection
tunes *what it costs* to run that role at all.

---

## 6. The Response Cache — Free Repetition, Not Compression

`src/core/llm_cache.py`, documented in full in
[Memory Boundaries §9](memory-boundaries.md#9-persistence-that-looks-like-memory-but-isnt--the-response-cache),
is a cost mechanism worth placing precisely on this map: it does not reduce
the token count of any individual call, and it is not a compression
technique. What it does is make an **exact repeat** of a call — the same
model, same messages, same parameters — free on the second occurrence, by
serving the cached response instead of paying the provider again. This
matters most during development and iteration (rerunning the same test,
retrying after an unrelated fix), and, as
[Idempotency §7](idempotency.md#7-the-one-place-idempotency-and-the-response-cache-meet)
notes, it incidentally makes two independent runs on identical inputs cheap
the second time without making them the same run. It is a real cost
mechanism, but a categorically different one from everything in
Sections 3-5: those reduce what one call costs; this reduces how often a
*genuinely identical* call gets paid for twice.

---

## 7. The Token Budget Guard — a Ceiling, Not a Reduction Strategy

`ensure_token_budget` (documented in
[Memory Boundaries §10](memory-boundaries.md#10-the-token-budget--a-hard-ceiling-enforced-before-the-call)
and [Tool Contracts §7](tool-contracts.md#7-the-llm-gateway--where-judgment-engines-actually-call-a-model))
is the last line of defense in this picture, not a cost-reduction technique
in its own right. It does not make any call cheaper; it refuses to make a
call at all once a formatter's assembled context exceeds a configured
ceiling, raising `TokenBudgetExceeded` rather than silently truncating or
overspending. Every other mechanism in this document is about keeping
*normal* calls efficient; this one is about catching the case where
something has already gone wrong upstream — a formatter regression, an
unexpectedly large resume — before it becomes an expensive surprise instead
of a loud, early failure.

---

## 8. A Feature Flag That No Longer Gates Anything

`FeatureFlags.enable_condensed_formatting`, still present in
`src/core/settings/schema.py` today, carries a detailed docstring describing
exactly the behavior this document has been discussing: *"Condenses verbose
job requirements, deduplicates keywords, and uses ultra-compact TOON format
to reduce token usage by ~20-30%... Recommended: True for production (cost
savings), False for development (easier debugging)."* This flag is the
direct, named descendant of Section 3's Phase 2 optimizations.

Searching the active codebase for where this flag is actually read today
turns up nothing outside the schema file itself — no formatter in
`src/formatters/` branches on it. This is the same recurring pattern named
explicitly in
[Observability §7](observability.md#7-the-recurring-pattern--aspirational-instrumentation-from-before-the-refactor):
a mechanism built and documented for an earlier state of the architecture
that the current code has moved past without removing the mechanism's
description. In this specific case, the underlying behavior the flag was
meant to toggle didn't disappear — it graduated. TOON is no longer an
opt-in, feature-flagged format; it is the unconditional default
`format_type` every formatter in the system uses (Section 4). Requirement
condensing and keyword deduplication are similarly unconditional today (see
`experience_optimizer_formatter`'s fixed `MAX_PRIORITY_REQUIREMENTS_FOR_EXPERIENCE_REWRITE`
cap and `optimize_skills`'s keyword handling, both documented in
[Memory Boundaries](memory-boundaries.md) and
[Orchestration Graph](orchestration-graph.md)). What's missing is only the
flag's *other* branch — there is no code path left that sends the
"full, unfiltered data for easier debugging" the docstring describes, because
the condensed path became the only path. The flag itself is now vestigial:
flipping it in `settings.yaml` today changes nothing, for the same reason
documented at length in
[Evaluation §10](evaluation.md#10-a-config-knob-that-does-nothing--a-live-discoverability-hazard)
for a different, unrelated config value.

---

## 9. The One Role With an Output Cap — an Asymmetry Worth Naming

Of the eight active agent configs, exactly **one** —
`skill_optimizer.yaml` — sets an explicit `max_tokens: 2000` limit. None of
the other seven do. Read against Section 3.2's Problem 4 ("no output length
constraint... accounted for roughly half the total waste on its own"), this
is not an oversight so much as a scar: the one role that historically proved
it needed an explicit output cap still carries one, and the other seven,
which were never shown to need one, don't. Whether that remains the right
distribution under the current model roster (Section 5) — where the roles
now carrying the heaviest generative load (`gpt-4.1`: professional summary,
experience, skills, ATS assembly) overlap only partially with the one role
that has a cap — is a question worth revisiting deliberately rather than
assuming the historical fix still targets the current bottleneck.

---

## 10. Quality Link — Why Smaller Context Also Means Better Output

Token reduction in this system was never purely a cost exercise, and the
project's own history makes that explicit in both directions. Section 3
found that removing genuinely irrelevant fields (metadata one agent never
used, a parameter no code ever read) cost nothing in quality and saved
real money. [Memory Boundaries §5](memory-boundaries.md#5-case-study-three-formatters-three-different-truncation-policies)
found the opposite lesson from the same instinct applied somewhere it didn't
belong: capping and ranking a summary writer's evidence to save tokens
directly caused generic, checklist-style output. The throughline is not
"smaller is always better" — it's that a formatter's job is to match context
precisely to what one agent's specific task needs, and that discipline
happens to pay off on both axes at once when applied correctly, and to fail
on both axes at once when applied as a blanket policy instead of a
per-agent judgment.

---

## 11. Design Rule — Where to Look Before Adding a New Field to Context

```text
Does the RECEIVING agent's specific task actually use this field, or does
it just make the payload feel more complete?
  "feel complete" -> leave it out. Section 3.2's Problem 1 is exactly
                      this mistake, at a measured cost of ~1,200 tokens
                      for one resume.

Is this field already present elsewhere in the same payload (the same
keyword in two lists, the same fact under two names)?
  YES -> deduplicate before it reaches the formatter's output, the way
         Section 3.2's Problem 5 was fixed.

Does a tool or task signature accept a parameter you're not certain
anything downstream actually reads?
  CHECK -> Section 3.3's zombie parameter cost ~13,500 tokens per resume
           before anyone looked. Grep for where a parameter is actually
           used, not just where it's accepted, before assuming it earns
           its cost.

Are you about to add a cap, rank, or truncation to save tokens?
  STOP -> re-read Memory Boundaries §5 first. The identical intention
          caused a real quality regression in this exact codebase. Prove
          it with a real run before shipping it, not by assumption.

Are you about to flip enable_condensed_formatting expecting it to
change formatter behavior?
  IT WON'T -> Section 8. The behavior it once gated is now unconditional;
              the flag reads as documentation of a past state, not a
              live switch.
```

---

## 12. Future Considerations

**Whether `enable_condensed_formatting` should be removed from the config
schema.** Section 8 documents that this flag no longer gates any code path —
its "condensed" behavior is now the only behavior, and its "full,
unfiltered" alternative no longer exists anywhere in the formatters. Leaving
it in `settings.yaml` with a docstring describing a toggle that doesn't
happen is the same discoverability hazard named for a different value in
[Evaluation §10](evaluation.md#10-a-config-knob-that-does-nothing--a-live-discoverability-hazard) —
worth resolving the same way: remove it, or restore the branch it once
controlled.

**Whether the single `max_tokens` cap on `skill_optimizer` should be
re-evaluated against the current model roster.** Section 9 notes that the
historical justification for this one role's output cap predates the current
per-role model assignments in Section 5. Whether the four `gpt-4.1` roles —
which now carry the heaviest generative load in the system — would benefit
from the same explicit ceiling, or whether none of them need one under
today's task designs, is worth deciding from current evidence rather than
inheriting a scar from an incident involving a different model roster
entirely.

**Whether this system should re-run the Section 3 discovery process
periodically.** The original bottleneck was found by turning on token
tracking and looking directly at what came back — not by inspecting code for
suspicious patterns in advance. As new roles, tools, or formatter fields are
added over time, nothing currently guarantees that a new "Skills Optimizer
incident" would be noticed before it accumulated real cost. Whether periodic,
deliberate token-usage review belongs as a standing practice — distinct from
the one-time historical investigation documented here — is an open question
this document raises without answering.
