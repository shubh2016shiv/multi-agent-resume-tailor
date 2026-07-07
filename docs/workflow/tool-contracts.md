# Tool Contracts
## The Deterministic and Judgment Layer Beneath Every Agent Claim

> **Scope:** `src/tools/` — the shared `ReviewResult` contract, the mechanical and
> judgment engines built on it, the agent-facing tool wrappers, and the one LLM
> gateway every judgment engine calls through.
> **Audience:** Contributors adding a new check; reviewers asking "how do we know
> this finding is trustworthy, and how sure is the system that it's right?"

---

## Table of Contents

1. [What Problem the Contract Layer Solves](#1-what-problem-the-contract-layer-solves)
2. [The Core Design Principle — One Shape for Every Finding](#2-the-core-design-principle--one-shape-for-every-finding)
3. [The ReviewResult Contract — Anatomy](#3-the-reviewresult-contract--anatomy)
4. [Two Kinds of Tools — Mechanical Engines vs. Judgment Engines](#4-two-kinds-of-tools--mechanical-engines-vs-judgment-engines)
5. [The Confidence Axis — Why Severity Alone Isn't Enough](#5-the-confidence-axis--why-severity-alone-isnt-enough)
6. [Agent-Facing Tools vs. Code-Owned Engines — Two Layers, One Contract](#6-agent-facing-tools-vs-code-owned-engines--two-layers-one-contract)
7. [The LLM Gateway — Where Judgment Engines Actually Call a Model](#7-the-llm-gateway--where-judgment-engines-actually-call-a-model)
8. [Prompt Rubrics as Versioned Files, Not Inline Strings](#8-prompt-rubrics-as-versioned-files-not-inline-strings)
9. [Case Studies in Contract Design](#9-case-studies-in-contract-design)
10. [engine_id — Traceability From Finding Back to Source](#10-engine_id--traceability-from-finding-back-to-source)
11. [Composability — Merging Findings for Agents and for Code](#11-composability--merging-findings-for-agents-and-for-code)
12. [Why Contracts Are Stronger Than Prompts](#12-why-contracts-are-stronger-than-prompts)
13. [Future Considerations](#13-future-considerations)

---

## 1. What Problem the Contract Layer Solves

Every stage of the pipeline eventually asks the same question in a different
costume: is this piece of text actually trustworthy? Does this skill have
evidence behind it? Does this rewritten bullet still say something true? Will
an ATS parser choke on this formatting? An agent can *answer* any of these
questions in prose, but prose is exactly the wrong shape for a system that
needs to route, gate, block, or repair based on the answer. "Looks good to me"
from an LLM is not a value a `StateGraph` conditional edge can branch on.

`src/tools/` exists to turn every one of these questions into the same typed
shape: a list of structured findings, each with a location, a severity, and a
confidence, regardless of whether the check that produced it was a regular
expression or a language model. That uniformity is what lets
[Orchestration Graph](orchestration-graph.md)'s nodes make gating decisions by
inspecting fields on a Pydantic object instead of parsing an agent's
sentence for the word "acceptable."

---

## 2. The Core Design Principle — One Shape for Every Finding

```text
   ANY CHECK IN THE SYSTEM, WHATEVER IT USES INTERNALLY:

   regex / unicode heuristics  --\
   whole-token counting         --\
   embedding cosine similarity   --> all produce the SAME shape -->  ReviewResult
   one bounded LLM call         --/
   an LLM call with a rubric   --/

   the CALLER never needs to know which of these produced a finding
   to decide whether it matters -- severity + confidence already say so
```

Whether a tool is "mechanical" (a regex, a token count, a rendered-artifact
inspection) or a "judgment engine" (one bounded LLM call against a rubric) is
an internal implementation detail. From the outside, both return a
`ReviewResult`. This is what lets a routing function in `graph.py` or a gate
in a node treat a keyword-density check and a truthfulness judgment as the
same kind of input: a list of comments it can filter by severity and
confidence, never a string it has to interpret.

---

## 3. The ReviewResult Contract — Anatomy

`src/tools/contracts/review.py` defines the entire vocabulary every tool in the
system shares:

```text
ReviewResult
  comments: list[ReviewComment]
  summary: str
  score: float | None            -- optional headline metric, tool-specific

ReviewComment                     (one finding from one engine)
  engine_id: str                  -- stamped by the harness, never by the model
  message: str                    -- what was noticed
  quoted_text: str                -- the flagged text, when there is one
  location: Location              -- WHERE, structured
  severity: Severity               -- BLOCKER > MAJOR > MINOR > SUGGESTION
  confidence: Confidence            -- HIGH > MEDIUM > LOW
  advice: str                     -- what to change
  proposed_rewrite: str | None     -- optional concrete fix

Location                          (a structured anchor, not just a quote)
  section: Section                 -- SUMMARY / EXPERIENCE / SKILLS / EDUCATION / ...
  bullet_index: int | None
  item_id: str | None
  character_span: list[int] | None
```

Two design choices here are worth dwelling on. First, `Location` is preferred
over raw `quoted_text` for anchoring precisely because a quote is fragile — if
a model misquotes a source string by even one character, a naive
substring-match anchor breaks, whereas a structured `section` + `bullet_index`
or `character_span` does not. Second, `severity` and `confidence` are
deliberately two separate axes rather than one combined score — a finding can
be extremely serious (`BLOCKER`) but reported with honest uncertainty
(`LOW`), and the system needs both numbers to decide whether to act on it. That
distinction is the subject of [Section 5](#5-the-confidence-axis--why-severity-alone-isnt-enough).

---

## 4. Two Kinds of Tools — Mechanical Engines vs. Judgment Engines

Every engine under `src/tools/engines/` falls into one of two families, and a
handful are an explicit hybrid of both:

```text
MECHANICAL ENGINES                          JUDGMENT ENGINES
(no LLM call, deterministic)                (one bounded LLM call, via the gateway)
-----------------------------                -----------------------------------
claim_inflation_detector                     skills_evidence_validator
keyword_coverage_analyzer                    rewrite_drift_detector
formatting_validator                         requirements_matcher
section_header_validator                     requirement_entailment_judge
                                              experience_rewrite_quality_auditor

                    HYBRID (mechanical gate, then judgment)
                    ----------------------------------------
                    summary_quality_auditor
```

Mechanical engines are always deterministic — same input, same output, every
time — which is exactly the property a quality gate needs: it cannot be fooled
by a model marking its own homework, and it costs nothing to run. Judgment
engines exist only where a mechanical proxy genuinely cannot answer the
question — whether a skill is *semantically* evidenced by a resume, whether a
rewrite subtly drifted from the truth, whether prose reads as generic
boilerplate. The system's default is mechanical wherever possible; judgment
engines are reserved for questions that are, by their nature, judgment calls.

---

## 5. The Confidence Axis — Why Severity Alone Isn't Enough

The contract module states the rule plainly: *"Mechanical tools are always
high [confidence]."* A regex either found a match or it didn't — there's no
uncertainty about whether the number `"5,000"` appears in a resume, only about
whether its presence *matters* (that's what severity encodes). Judgment
engines, by contrast, must report confidence honestly, because their entire
value proposition depends on the caller being able to tell a highly-certain
model verdict from a hedge.

```text
mechanical finding:  confidence is ALWAYS HIGH
                     (presence of a value is a measurement, not a judgment)

judgment finding:    confidence reflects the model's OWN honest uncertainty
                     HIGH   -- concrete, verifiable, clearly supported/unsupported
                     MEDIUM -- likely, but not certain ("likely unsupported")
                     LOW    -- a hedge ("the field may implicitly cover it")
```

This is why several gates in the pipeline require **both** a serious severity
**and** HIGH confidence before acting — `is_confident_unsupported` in the
Skills Optimizer node (documented in [Orchestration Graph §4](orchestration-graph.md#4-stage-by-stage-walkthrough))
is the canonical example: a skill is only removed when a judgment engine is
both loud (BLOCKER/MAJOR) and certain (HIGH) about its absence of evidence.
A MEDIUM or LOW finding is treated as advisory, never as license to delete a
candidate's truthful claim on a hunch. The two axes exist precisely so that
"this is serious" and "I am sure about this" can be asked, and answered,
independently.

---

## 6. Agent-Facing Tools vs. Code-Owned Engines — Two Layers, One Contract

`src/tools/` is organized into two layers that share the same `ReviewResult`
contract but serve two different consumers:

```text
   src/tools/engines/*        -- the actual checks, return ReviewResult
          |
          | wrapped by
          v
   src/tools/agent_tools/*    -- @tool-decorated functions an Agent may call
          |                      mid-turn; return a PLAIN STRING (agents read text)
          |
          v
   render_review_result()     -- flattens ReviewResult into a readable block:
                                  "[severity/confidence] (section) message
                                     advice: ...
                                     rewrite: ..."
```

An agent tool such as `validate_ats_compliance` or `audit_truthfulness` never
returns a `ReviewResult` object to the agent — CrewAI tools return strings, and
an LLM reads prose, not Pydantic. But the *engine* underneath every one of
those tools returns the typed `ReviewResult` first, and it is that same typed
object — not the rendered string — that orchestration nodes and quality gates
consume directly when they need to filter by `severity` or `confidence`
programmatically. The same check therefore has two faces: a string an agent
can read mid-reasoning, and a structured object code can act on
mechanically — generated from the exact same underlying computation, never
duplicated logic.

---

## 7. The LLM Gateway — Where Judgment Engines Actually Call a Model

Every judgment engine in the system calls through exactly one choke point:
`src/tools/llm_gateway/`. No engine constructs its own `LLM` instance or
crafts its own retry logic — that would mean re-solving the same reliability
problem in five different files with five different bugs.

```text
   request_structured_output(output_model, system_prompt, user_content, temperature)
     -- the generic harness: ANY judgment engine that needs a typed Pydantic
        result back from the model goes through here
     -- STEP 1: enforce the input token budget BEFORE any provider call
                (raises TokenBudgetExceeded rather than sending an oversized request)
     -- STEP 2: call the model with response_format=output_model
     -- STEP 3: retry ONCE on malformed output, then raise RuntimeError
                (never returns malformed data to a caller -- fails loudly instead)

   request_review(engine_id, rubric_prompt, review_input)
     -- the ReviewResult-specific wrapper: calls request_structured_output(ReviewResult, ...)
     -- then stamps every comment.engine_id = engine_id AFTER validation,
        because "the model cannot reliably know which engine invoked it"

   embed_texts / cosine_similarity / max_similarity
     -- a separate, non-structured-output gateway for embedding-based similarity
     -- returns SCORES ONLY; the threshold/cutoff decision belongs to the caller,
        never to this module
```

Three details here recur as themes across this whole codebase. First, the
gateway enforces its own reliability guarantee (retry-once-then-fail-loudly) so
that no downstream engine has to. Second, `engine_id` is metadata the harness
asserts *about* the model's output, not information the model is trusted to
report about itself — the same "code decides the trust boundary" principle
documented for state and routing in [Orchestration Graph](orchestration-graph.md)
applies here at the level of a single API call. Third, `max_similarity`
deliberately stops short of making a match/no-match decision — it returns a
number and leaves the cutoff to whichever engine calls it, keeping the
judgment of "how similar is similar enough" visible in the caller's own code
rather than buried in a shared utility.

---

## 8. Prompt Rubrics as Versioned Files, Not Inline Strings

Every judgment engine loads its rubric through `load_tool_prompt(relative_path)`
in `src/core/prompt_catalog.py`, never as an inline Python string:

```text
SKILLS_EVIDENCE_RUBRIC = load_tool_prompt("truthfulness/skills_evidence.md")
REWRITE_DRIFT_RUBRIC   = load_tool_prompt("truthfulness/rewrite_drift.md")
ENTAILMENT_RUBRIC      = load_tool_prompt("requirement_entailment/entailment_judge.md")
```

`load_tool_prompt` resolves the requested path under one configured prompt
root, rejects any path that would escape that root (a directory-traversal
guard on what is otherwise just a file read), and caches the result — a rubric
file is read from disk exactly once per process. The practical effect is that
every judgment engine's prompt is a reviewable, diffable Markdown file living
in one catalog directory, exactly the same discipline documented for CrewAI
agent personas and task instructions in
[Agent Roles §13](agent-roles.md#13-configuration-surface--agentsyaml-and-tasksyaml).
Whether the prompt belongs to a full CrewAI `Agent` or to a single bounded
tool-layer LLM call, this codebase treats it as a versioned artifact, not a
disposable string embedded in logic.

---

## 9. Case Studies in Contract Design

### 9.1 `claim_inflation_detector` — Mechanical, HIGH Confidence by Construction

Detects whether a rewrite introduced a number the original resume never had, using
a regex over normalized numeric tokens (so `"5K"` and `"5,000"` compare equal).
Its own module docstring records a real design regression it moved past: an
earlier version also used spaCy NER to flag introduced organizations and
products, but NER trained on prose is unreliable on resume fragments — it
mislabelled common tech terms like `"Python"` and `"AWS"` as newly-introduced
organizations even when they were already present, tanking the accuracy score
with false positives. The fix was not to tune the NER model; it was to narrow
the engine's scope to exactly one measurable thing — numeric values — and hand
semantic entity fabrication to a separate, confidence-gated judgment engine
(`rewrite_drift_detector`) instead. Every finding here is `Confidence.HIGH`
because the presence of a numeric value is a measurement, and `Severity.MAJOR`
rather than `BLOCKER` specifically so a false positive here can never
unilaterally hard-block the pipeline on its own.

### 9.2 `skills_evidence_validator` / `rewrite_drift_detector` — Pure Judgment, Confidence-Gated

Both engines have no mechanical half at all, because the question they answer
— "is this claim actually supported?" — cannot be settled by string matching
in either direction. A skill can be backed without being named (a bullet about
"container orchestration" supports a listed skill of "Kubernetes") and named
without being backed ("familiar with Kubernetes through reading" contains the
word but is weaker than no claim at all). Only reading the context settles
either case, so, in the engine's own words, "the detection IS the judgment."
Two guardrails keep this honest: self-evident skills are never flagged, and
when a field might implicitly cover a skill, the engine sets confidence to
`LOW` rather than flagging it with false certainty — deferring to the
downstream gate's rule that only HIGH-confidence findings act.
`rewrite_drift_detector` adds one more discipline worth naming: it is
explicit about its own ceiling, describing itself as "a safety net, not a lie
detector" because it is a model diffing work another model produced — it
reliably catches an invented credential or a brand-new number, but not a
subtle reframing the optimizer already rationalized to itself. Its
implementation also short-circuits entirely — no LLM call at all — when the
rendered original and revised text are byte-identical, since there is nothing
to judge.

### 9.3 `requirement_entailment_judge` — A Bounded, Temperature-0 Closed-Label Judge That Verifies Itself

This engine answers one narrow question — does the resume's evidence entail
one specific job requirement? — with a closed three-value verdict
(`ENTAILED` / `NOT_SUPPORTED` / `INCONCLUSIVE`), never free text. Two things
distinguish it as a "bounded" judge rather than an open-ended one. First, it
overrides the configured agent temperature down to `0.0` explicitly, because
"a quality gate must be as reproducible as possible" — this is the same
principle behind the per-role temperature spectrum in
[Agent Roles §9](agent-roles.md#9-per-role-temperature--tuning-determinism-vs-creativity),
applied to a tool-layer call instead of an agent. Second, and more
interestingly, the engine does not simply trust a model's claim of
`ENTAILED` — when the model asserts entailment, it must also supply a
`supporting_quote`, and the engine's own code (`_quote_exists`) verifies that
quote is *actually present*, verbatim or whitespace-normalized, in the resume
text it was given. A model that claims a match but cannot produce a real quote
gets its verdict downgraded to `INCONCLUSIVE` in code, not accepted at face
value. Any gateway failure, blank input, or unverifiable quote fails closed to
`INCONCLUSIVE` rather than guessing — this is a judgment engine that checks its
own model's homework before returning a verdict anyone else is allowed to act on.

### 9.4 `match_resume_to_job` — Composing a Judgment Engine and a Mechanical Engine Into One Report

`match_requirements` (judgment: does the resume's evidence semantically satisfy
each stated job requirement?) and `analyze_keyword_coverage` (mechanical: which
literal JD keywords appear in the text?) are two independently useful engines.
`match_resume_to_job` composes them into exactly one `ReviewResult`, keeping
the requirements engine's score as the headline metric "because semantic
requirement fit matters more than raw keyword repetition." This is not an
academic composition exercise — it is, by its own docstring, "the single
computation the gap-analysis stage runs in code rather than via an LLM tool,"
the concrete mechanism behind the code-computed-context pattern documented in
[Agent Roles §6](agent-roles.md#6-the-code-computed-context-pattern--why-gap-analysis-has-no-tools)
and [Orchestration Graph §4](orchestration-graph.md#4-stage-by-stage-walkthrough).
`match_requirements`'s own docstring is candid about being "the single riskiest
engine in the system" for exactly the reason semantic matching is dangerous:
models confidently hallucinate equivalence ("SQL covers NoSQL" — it does not)
at roughly an 80% honesty ceiling the rubric explicitly acknowledges rather
than pretends away.

### 9.5 `formatting_validator` / `section_header_validator` — Two Independent Mechanical Engines, Merged at the Tool Layer

ATS compliance is checked by two engines that never talk to each other:
`audit_ats_formatting` looks for structural hazards (pipe tables, tab
characters, HTML fragments, exotic Unicode symbols, PDF-extraction artifacts
like stray `(cid:12)` tokens, multi-column layout heuristics, and Markdown
links whose visible label hides the real URL), while `audit_section_headers`
checks whether the resume's section titles match an ATS-recognizable alias
list, splitting missing sections into `MAJOR` (essential: experience,
education, skills) versus `SUGGESTION` (optional: certifications). The
`validate_ats_compliance` agent tool merges both results via
`merge_review_results` before rendering them to the agent — the agent sees one
coherent audit, but the two underlying checks remain independently
testable, independently tunable engines that happen never to need to know
about each other.

### 9.6 `summary_quality_auditor` — A Hybrid Engine, and a Documented Production Fix

This engine runs its mechanical half first — word-count band (80-110 words)
and first-person-pronoun detection via exact token matching (so `"academy"`
never false-matches `"my"`) — and then unconditionally spends one LLM call
judging what no mechanical proxy can read: generic boilerplate, a weak thesis,
brochure tone, a missing value proposition. Unlike some other engines in this
codebase, the LLM call here has no cheaper mechanical gate in front of it,
because the summary is often a recruiter's very first read and the highest-risk
place for generic AI tone to leak into the final resume — the engine's own
docstring accepts the fixed per-call cost as worth it for that reason alone.

The length check carries a documented production incident worth repeating in
full: the floor used to be `MINOR` severity while the ceiling was `MAJOR` — an
asymmetry that seemed reasonable ("too short" felt softer than "too long") —
until live runs actually shipped 67-word summaries that dropped a candidate's
strongest evidence, because a `MINOR` finding never blocked the run. The fix
was not a smarter length heuristic; it was to make the floor exactly as
serious as the ceiling, `MAJOR` on both sides, so the quality gate in
[Orchestration Graph §4](orchestration-graph.md#4-stage-by-stage-walkthrough)
(which blocks only on `MAJOR`+) actually catches both failure directions. This
is a clean example of a contract-design bug — not a prompt bug, not a model
bug — that only became visible once real output shipped, and was fixed at the
severity-assignment layer rather than by asking the model to try harder.

---

## 10. engine_id — Traceability From Finding Back to Source

Every `ReviewComment` in the system carries an `engine_id` — `"claim_inflation_detector"`,
`"skills_evidence_validator"`, `"requirement_entailment_judge"`, and so on —
and that field is set by the calling harness, never supplied by the model
itself. `request_review` states the reasoning directly: "the model cannot
reliably know which engine invoked it, so the gateway sets that boundary
metadata after validation." This is a small detail with an outsized payoff:
because every finding is traceable to the exact engine that produced it, any
downstream consumer — a log line, an observability trace, a developer staring
at a confusing quality-gate failure — can go straight to one file to
understand exactly what logic (and, for judgment engines, exactly which
rubric file) produced a given comment, without guessing from the message text
alone.

---

## 11. Composability — Merging Findings for Agents and for Code

Two small utility functions in `src/tools/agent_tools/resume_review_tools.py`
carry more architectural weight than their size suggests:

```text
merge_review_results(list[ReviewResult]) -> ReviewResult
  -- concatenates comments from several engines into ONE ReviewResult
  -- used whenever a single agent-facing tool wraps more than one engine
     (e.g. validate_ats_compliance = formatting + section headers;
           audit_truthfulness      = claim inflation + rewrite drift)

render_review_result(ReviewResult, title) -> str
  -- the ONE function that turns any ReviewResult into agent-readable text:
     "[severity/confidence] (section) message / advice: ... / rewrite: ..."
  -- every agent-facing tool in the system funnels through this single
     renderer, so an agent reading tool output always sees the same format
     no matter which engine, or how many merged engines, produced it
```

This is the same "one shape for every finding" principle from
[Section 2](#2-the-core-design-principle--one-shape-for-every-finding), applied
at the boundary where structured findings become prose an LLM reads. An agent
calling `validate_ats_compliance` never needs to know it triggered two
separate engines under one tool name; it sees one consistently-formatted
report, while the code building that tool remains free to add, remove, or
re-tune the underlying engines independently.

---

## 12. Why Contracts Are Stronger Than Prompts

A prompt describes desired behavior: "don't invent facts," "stay ATS-safe,"
"keep the tone professional." A contract defines *admissible* behavior — the
literal set of shapes a result is allowed to take, checked in code, not hoped
for in language. For a resume-tailoring system, that difference decides
whether a truthfulness guarantee is a request or a boundary. A prompt can ask
a model not to inflate a number; `claim_inflation_detector` can regex the
rendered text and answer with certainty whether it did. A prompt can ask a
model to judge whether a skill is evidenced; `skills_evidence_validator`
still routes that judgment through a typed contract with an honest confidence
value, so a hedge from the model is legible as a hedge to every downstream
consumer, not silently rounded up to a fact. The design bias throughout
`src/tools/` runs one direction: wherever a high-risk check *can* be expressed
as a deterministic, testable contract, it is moved out of prompt instructions
and into one.

---

## 13. Future Considerations

**Whether the confidence-gating threshold should be uniform across engines.**
Today, "act only on HIGH confidence + serious severity" is a rule each
consuming node re-derives locally (e.g. `is_confident_unsupported` in the
Skills Optimizer node). Since this pattern already repeats across at least two
engines (`skills_evidence_validator`, `rewrite_drift_detector`) and is likely
to repeat further, it may be worth deciding, at the contract level, whether
"confidently actionable" is itself part of `ReviewResult`'s vocabulary rather
than a predicate every consumer reimplements.

**Whether score semantics need a documented contract of their own.**
`ReviewResult.score` currently means different things in different engines — a
must-have requirement coverage fraction in `match_requirements`, a keyword
coverage fraction in `analyze_keyword_coverage`, and `None` in engines that
don't compute one at all. As more engines begin to populate it, the system may
need an explicit architectural decision about what `score` is allowed to mean,
rather than letting each engine's convention stand on its own.

**Whether mechanical and judgment engines should be visibly distinguished in
the contract itself.** Right now the mechanical/judgment split is knowable
only by reading an engine's implementation or its docstring — `Confidence.HIGH`
is a strong hint for mechanical engines, but it is not a declared, structural
guarantee. A downstream consumer (or a future auditor of the system) cannot
currently ask a `ReviewResult` "was this finding computed or inferred?"
without already knowing which engine produced it — worth deciding whether that
distinction belongs in the architecture's vocabulary directly.
