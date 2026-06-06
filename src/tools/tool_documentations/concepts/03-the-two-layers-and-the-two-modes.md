# Concept 3 — The Two Layers, and the Two Modes (how it all coordinates)

> Read after Concepts 1 and 2. This is the "how does it all fit together" document.

## Part A — Why two layers exist

Recall the two layers from the README:

- **Engines** (Layer 1): small, single-purpose units. Typed data in, a `ReviewResult`
  out. One per file.
- **Agent-facing tools** (Layer 2): the ~7 coarse instruments an agent is actually
  handed. Each bundles several engines.

Why not just hand the engines straight to the agents? Two reasons:

1. **Keep the agent's world small.** An agent that had to choose between ~19 micro-tools
   would have a bloated, confusing prompt and would pick wrong. Agents get only the
   coarse tools they can call with clean inputs. Professional experience audit is a
   code-facing exception: orchestration runs it after the writer returns typed output.

2. **Different shapes at each layer.** Engines speak in typed objects (`Resume`) and
   return `ReviewResult` objects — perfect for testing and for code. But a CrewAI tool
   must **return a string** to the LLM and receives its arguments **as text/JSON** from
   the LLM. Layer 2 is the translation boundary: it parses the incoming JSON into a
   `Resume`, runs the engines, and renders the merged `ReviewResult` back into a string.

```
   AGENT (an LLM)                         AGENT-FACING TOOL (Layer 2)         ENGINES (Layer 1)
   "audit this resume" ── JSON string ──► parse JSON -> Resume object  ──►   engine A (Resume) -> ReviewResult
                                          │                                  engine B (Resume) -> ReviewResult
                                          │                            ◄──   engine C (Resume) -> ReviewResult
                          string  ◄────── render( merge(A,B,C) )
```

## Part B — What an agent-facing tool actually does (the 4 steps)

Every agent-facing review tool in `agent_facing_tools.py` follows the same four steps:

```
1. PARSE      Resume.model_validate_json(resume_json)
              (if the JSON is bad, return a clear error string — never crash)
                       │
                       ▼
2. RUN        call each backing engine with the typed object
              e.g. match_requirements(resume, job), analyze_keyword_coverage(...)
                       │
                       ▼
3. MERGE      _merge([...])  ->  one ReviewResult holding ALL the comments
              (concatenates comments; keeps the lead engine's score)
                       │
                       ▼
4. RENDER     _render_review_result(merged, "<title>")  ->  a readable string
              the agent reads this string and decides what to do next
```

Two small shared helpers do steps 3 and 4 for all tools:

- `_merge(results)` — takes several `ReviewResult`s and concatenates their comments into
  one. It also carries forward the first engine's `score` (so a composite tool does not
  silently lose a meaningful number).
- `_render_review_result(result, title)` — turns a `ReviewResult` into the plain-text
  report the agent reads (a title, the summary, the score if any, then one block per
  finding: `[severity/confidence] (section) message` + advice).

That's the entire coordination mechanism. An agent-facing tool is "parse → run engines →
merge → render."

### Agent-facing tools and the engines behind them

```
audit_summary              ───► audit_summary_quality     (hybrid)

check_skills_evidence      ───► validate_skills_evidence  (judgment)

audit_truthfulness         ─┬─► detect_claim_inflation    (mechanical)
                            └─► detect_rewrite_drift       (judgment)

match_job_requirements     ─┬─► match_requirements        (judgment)
                            └─► analyze_keyword_coverage  (mechanical)

validate_ats_compliance    ─┬─► audit_ats_formatting      (mechanical)
                            └─► audit_section_headers      (mechanical)

analyze_jd_keyword_coverage ──► analyze_keyword_coverage  (mechanical)
```

Professional experience quality uses the same `ReviewResult` shape, but it is not
an agent-facing tool. Orchestration calls the code-facing helper after structured
writer output exists:

```
audit_experience_quality_for_experiences
                            ├─► audit_bullet_structure_for_experiences   (mechanical)
                            ├─► audit_consistency_for_experiences         (mechanical)
                            ├─► audit_quantification_for_experiences      (hybrid)
                            └─► audit_language_quality_for_experiences    (judgment)
```

Notice how a tool's *type* comes from what it bundles: `match_job_requirements` is
"hybrid" because it runs one judgment engine and one mechanical engine and merges them.

## Part C — The engines the orchestrator runs directly (not agent tools)

Some engines are **not** wrapped as agent-facing tools, because no reasoning is needed —
they are fixed pipeline steps the orchestrator just runs in order:

- `convert_document_to_markdown` — turn the uploaded file into Markdown.
- `audit_extraction_quality` — sanity-check that conversion didn't lose content.
- `redact_pii` — mask personal data **before** any text reaches the model.
- `extract_resume` / `extract_job_requirements` — turn Markdown into typed objects.
- `resume_renderer` — turn the final `Resume` back into a PDF.

These run once, in a fixed sequence, with no agent deciding anything. That is why they
are plain functions the orchestrator calls, not `@tool`s handed to an agent.

## Part D — The two modes, drawn out

### Mode A — Resume Excellence (no job description)

```
  uploaded file
       │
       ▼  convert_document_to_markdown        (mechanical)
   Markdown
       │
       ▼  audit_extraction_quality            (mechanical)  -> flags a bad conversion
       ▼  redact_pii                          (mechanical)  -> masks PII, keeps a map
   redacted Markdown
       │
       ▼  extract_resume                      (judgment, LLM)
   Resume (structured)
       │
       ├──► Experience Optimizer  → code-facing experience audit after structured output
       ├──► Summary Writer        → audit_summary              (length, person, value)
       ├──► Skills Optimizer      → check_skills_evidence      (every skill backed up?)
       ├──► ATS Optimization      → validate_ats_compliance    (formatting, headers)
       │
       ▼  (after any rewrite)
       └──► Quality Assurance     → audit_truthfulness         (did the rewrite stay honest?)
       │
       ▼  resume_renderer                     (mechanical)     -> final ATS-safe PDF
   finished PDF
```

### Mode B — Job Tailoring (a job description is supplied)

Everything in Mode A, **plus** the two job-matching steps. The only new ingestion step is
extracting the job description; the only new review step is matching against it.

```
   ... same ingestion as Mode A ...
   Resume (structured)            job-post text
       │                                │
       │                                ▼  extract_job_requirements   (judgment, LLM)
       │                            JobDescription (structured)
       │                                │
       ├──── all the Mode-A reviews ────┤
       │                                │
       └──► Gap Analysis  → match_job_requirements(Resume, JobDescription)
                            ├─ match_requirements   (judgment): does the resume EVIDENCE each requirement?
                            └─ analyze_keyword_coverage (mechanical): which JD keywords appear?
       │
       ▼  resume_renderer  -> final ATS-safe PDF
```

So Mode B is **additive**: the same Mode-A machinery, plus `job_matching/` and the
job-description extractor switch on. Sixteen of the engines need no job description at
all; only `job_matching/` (and the JD extractor) are Mode-B-only.

## Part E — A full end-to-end trace of one finding

Let's follow a single weak bullet, `"Responsible for building models."`, all the way
through, so you see every layer touch it.

```
1. INGESTION
   PDF ─► convert ─► "...Responsible for building models...." (Markdown)
       ─► redact (name/email masked) ─► extract_resume (LLM)
   ──► Resume object whose work_experience[0].achievements[0] == "Responsible for building models."

2. THE WRITER RETURNS STRUCTURED OUTPUT
   The Professional Experience agent reads TOON context and returns
   OptimizedExperienceSection through CrewAI output_pydantic.

3. CODE COORDINATES THE EXPERIENCE CHECKS
   run 4 engines on typed list[Experience]:
       audit_bullet_structure_for_experiences  -> (nothing for this bullet)
       audit_consistency_for_experiences       -> (nothing)
       audit_quantification_for_experiences    -> mechanical: bullet has no digit -> send to LLM ->
                                                  ReviewComment("lacks a quantified result", SUGGESTION, MEDIUM)
       audit_language_quality_for_experiences  -> LLM: "Responsible for" is duty language ->
                                                  ReviewComment("duty language, reframe as achievement", MINOR, HIGH)
   _merge(...) -> one ReviewResult with BOTH comments

4. ORCHESTRATION READS THE REVIEWRESULT
   [suggestion/medium] (experience) The bullet lacks a quantified result.
       advice: Add a metric such as number of models built or accuracy gained.
   [minor/high] (experience) Duty language: states a responsibility, not an achievement.
       advice: Reframe as a concrete achievement with an outcome.

5. ORCHESTRATION DECIDES whether one rewrite is needed, then later the Quality
   Assurance agent calls audit_truthfulness(original, revised) to confirm the rewrite
   did not invent anything.
```

That is the whole system in one picture: ingestion produces typed data, an agent calls a
coarse tool or code-facing helper drives several engines and merges their same-shaped
results, orchestration or the agent acts on the merged report, and a truthfulness check
guards any rewrite.

## What to take away

1. Two layers: **engines** do the work (typed in, `ReviewResult` out); **agent-facing
   tools** bundle engines only when an agent can provide clean tool inputs.
2. Every agent-facing tool is the same four steps: **parse → run engines → merge →
   render**.
3. Ingestion, rendering, and professional experience audit are run **directly by
   orchestration/code** when typed objects already exist.
4. **Mode B is Mode A plus** the job-matching engines and the JD extractor; everything
   else is shared.

Next, read a real engine end to end:
**[../engines/resume-diagnostics/quantification-auditor.md](../engines/resume-diagnostics/quantification-auditor.md)**,
then a real agent-facing tool:
**[../agent-tools/match-job-requirements.md](../agent-tools/match-job-requirements.md)**.
