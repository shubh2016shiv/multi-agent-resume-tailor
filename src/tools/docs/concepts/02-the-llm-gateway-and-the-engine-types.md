# Concept 2 — The LLM Gateway, and the Three Engine Types

> Read after Concept 1. This answers, precisely: *which tools use the AI model, which
> are pure code, and which are both* — and how all the LLM traffic flows through one door.

## Part A — There is exactly one door to the LLM

A hard rule in this codebase:

> **No engine ever calls OpenAI (or any model SDK) directly. Every LLM call goes through
> the `llm_gateway/` package.**

This one-door design buys three things:

1. **A single place to look.** Want to know if a tool uses the model? Check whether its
   file imports from `llm_gateway`. That's the whole test.
2. **Traceability.** Because all calls funnel through one spot, every finding can be
   stamped with the `engine_id` of the engine that asked — so later you can trace any
   finding back to its engine, its prompt, and its token cost.
3. **One place to enforce correctness.** The gateway refuses to hand back malformed data
   (explained below), so no engine has to write that safety code itself.

The gateway has two functions. You will see engines import one or the other.

### `request_structured_output(output_model, system_prompt, user_content)`

This is the lower-level function (`llm_gateway/structured_output.py`). In plain words:
*"Ask the model a question, and force the answer to fit this exact Pydantic shape."*

```
request_structured_output(Resume, "extract a resume...", "<the markdown>")
        │
        ▼
   builds a CrewAI LLM   (model name + temperature come from settings.yaml — NEVER hardcoded)
        │
        ▼
   asks the model, demanding JSON shaped like `Resume`
        │
        ▼
   validates the reply against the Pydantic model
        │           │
        │ valid      │ malformed
        ▼            ▼
   returns the    retry ONCE; if it fails again, raise an error
   typed object   (it never returns half-broken data to the caller)
```

The model name is read from project config (`settings.yaml`), so you will **never** see
`"gpt-4o"` hardcoded inside an engine. That is deliberate — the model is configuration,
not code.

This function is used both for **extraction** (turn text into a `Resume` or a
`JobDescription`) and, indirectly, for **reviews** (next).

### `request_review(engine_id, rubric_prompt, review_input)`

This is the convenience wrapper most judgment *review* engines use
(`llm_gateway/review_requests.py`). It does two things:

1. Calls `request_structured_output(ReviewResult, rubric_prompt, review_input)` — i.e. it
   forces the model's answer into the `ReviewResult` shape from Concept 1.
2. **Stamps `engine_id` onto every comment** in the result. The model cannot know which
   engine called it, so the gateway sets the id for you.

```
request_review("language_quality_auditor", LANGUAGE_RUBRIC, "<bullets>")
        │
        ▼
   request_structured_output(ReviewResult, LANGUAGE_RUBRIC, "<bullets>")
        │
        ▼
   for each comment in the result:  comment.engine_id = "language_quality_auditor"
        │
        ▼
   returns a ReviewResult whose findings are all traceable to this engine
```

So a typical judgment review engine is *tiny*: it builds a text prompt (the "rubric")
describing what to look for, hands it to `request_review`, and returns the result. All
the heavy lifting — calling the model, enforcing the shape, retrying, stamping — is in
the gateway.

---

## Part B — The three engine types, and how to recognise each

### 1. Mechanical (pure code, no LLM)

A mechanical engine only *measures* with plain Python (counting, regex, math, or a
local NLP library like spaCy that runs offline). It cannot hallucinate; a bug is just a
bug. By the contract rule from Concept 1, every finding it makes is `confidence = high`.

**How to recognise one:** its file does **not** import anything from `llm_gateway`.

```
RESUME  ──►  [ pure Python: count / regex / spaCy / math ]  ──►  ReviewResult (all HIGH confidence)
                         no network, no model
```

Examples: `audit_bullet_structure` (counts bullets and words), `audit_ats_formatting`
(regex for tabs/tables), `detect_claim_inflation` (regex extracts and normalizes
numbers, then diffs two resumes — still pure computation).

### 2. Judgment (the model is the expert)

A judgment engine asks the model a well-framed question because the answer needs
professional taste that no rule can capture — e.g. "is this phrasing duty-language or a
real achievement, *for this person's field*?" There is no list of "bad phrases" that
works across software, nursing, and law; the model holds that cross-domain knowledge.

Because the model can be wrong, every finding carries an honest `confidence`, and the
system trusts high-confidence findings while treating low-confidence ones as advisory.

**How to recognise one:** its file imports `request_review` (or
`request_structured_output`) and does essentially no mechanical detection of its own —
it builds a prompt and returns what the model says.

```
RESUME ──► build a "rubric" prompt ──► request_review(...) ──► ReviewResult
                                          (the one LLM call)     (HIGH / MEDIUM / LOW)
```

Examples: `audit_language_quality`, `validate_skills_evidence`, `detect_rewrite_drift`,
`match_requirements`, plus the two **extractors** (`extract_resume`,
`extract_job_requirements`) which use `request_structured_output` to turn text into typed
objects.

### 3. Hybrid (mechanical *and* judgment together)

A hybrid does some pure-code work *and* a judgment call. There are **two distinct
shapes**, and telling them apart is worth your time:

**Shape 3a — mechanical gate (the cheap, common one).** The mechanical half decides
*whether the LLM runs at all*. If the mechanical pass finds nothing, **no LLM call is
made.**

```
RESUME ──► [ mechanical filter ] ──┬── nothing found ──► ReviewResult([])   (no LLM call, $0)
                                   │
                                   └── found N items ──► LLM only on those N ──► ReviewResult
```
Example: `audit_quantification`. The mechanical half finds bullets with no digit. Only
those bullets are sent to the model (to suggest a metric). A fully quantified resume
costs zero LLM calls.

**Shape 3b — mechanical AND always-on LLM (merged).** Both halves always run and their
findings are merged, because there is no mechanical proxy to gate on.

```
RESUME ──┬──► [ mechanical checks ] ──► HIGH-confidence findings ──┐
         │                                                          ├─► merge ─► ReviewResult
         └──► request_review(...)   ──► MEDIUM-confidence findings ─┘
```
Example: `audit_summary_quality`. Length and first-person pronouns are checked
mechanically (HIGH confidence); "is it generic / does it state a value proposition" is a
judgment call (MEDIUM) that always runs for any non-empty summary.

---

## Part C — The complete classification (memorise the shape, not the list)

| Engine | Type | Imports `llm_gateway`? |
|---|---|---|
| convert_document_to_markdown | Mechanical | No |
| audit_extraction_quality | Mechanical | No |
| redact_pii | Mechanical | No |
| extract_resume | Judgment | Yes (`request_structured_output`) |
| extract_job_requirements | Judgment | Yes (`request_structured_output`) |
| audit_bullet_structure | Mechanical | No |
| audit_consistency | Mechanical | No (uses spaCy, runs offline) |
| audit_quantification | **Hybrid 3a** (gate) | Yes (`request_review`) |
| audit_language_quality | Judgment | Yes (`request_review`) |
| audit_summary_quality | **Hybrid 3b** (always) | Yes (`request_review`) |
| analyze_keyword_coverage | Mechanical | No |
| match_requirements | Judgment | Yes (`request_review`) |
| audit_ats_formatting | Mechanical | No |
| audit_section_headers | Mechanical | No |
| validate_skills_evidence | Judgment | Yes (`request_review`) |
| detect_claim_inflation | Mechanical | No (uses spaCy) |
| detect_rewrite_drift | Judgment | Yes (`request_review`) |
| resume_renderer | Mechanical | No (LaTeX + tectonic) |

> Rule of thumb you can apply to any new file: **grep it for `llm_gateway`.** Present ⇒
> it uses the model (judgment or hybrid). Absent ⇒ mechanical.

---

## What to take away

1. All LLM traffic goes through `llm_gateway/`; that one door is how you tell which tools
   use the model and how findings stay traceable.
2. The gateway forces the model's reply into a Pydantic shape and retries-then-fails on
   bad output, so engines never handle broken model data.
3. Three types: **mechanical** (no gateway import, always high confidence), **judgment**
   (gateway import, model is the expert, honest confidence), **hybrid** (both — either a
   mechanical *gate* on the LLM, or mechanical *plus* an always-on LLM call).

Next: **[03-the-two-layers-and-the-two-modes.md](03-the-two-layers-and-the-two-modes.md)**
— how engines and agent-facing tools coordinate, and a full end-to-end trace.
