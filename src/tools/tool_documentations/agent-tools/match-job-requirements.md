# Agent-Facing Tool — `match_job_requirements`

**File:** `src/tools/agent_facing_tools.py`
**Function:** `match_job_requirements(resume_json: str, job_json: str) -> str`
**CrewAI tool name:** "Match Job Requirements"
**Type:** Hybrid — it bundles one **judgment** engine and one **mechanical** engine
**Runs in:** Mode B only (a job description is required)
**Used by:** the Gap Analysis agent

> Read the three `concepts/` docs first. This page shows how a Layer-2 tool *coordinates*
> engines — the pattern is identical for all 7 agent-facing tools.

---

## 1. Purpose (one sentence)

Tell the Gap Analysis agent, for one resume against one job, **how well the resume
actually evidences each requirement** *and* **which job keywords are present** — as a
single readable report.

## 2. Why it exists

"How well does this resume answer *this* job?" is the core question of job tailoring, and
it really has two different sub-questions that need two different kinds of tool:

1. **Does the resume EVIDENCE each requirement?** ("The job needs 5 years of Kubernetes —
   does the resume actually demonstrate that?") This is a *semantic judgment*: Flask
   experience *partially* covers a FastAPI requirement; SQL does **not** cover NoSQL. Only
   the model can weigh that, so this half is a **judgment engine** (`match_requirements`).

2. **Do the job's exact keywords literally appear?** Applicant Tracking Systems scan for
   specific terms. This is a *mechanical* fact — a word is present or it isn't — so this
   half is a **mechanical engine** (`analyze_keyword_coverage`).

Bundling both behind one tool means the Gap Analysis agent asks one question and gets the
full picture — the nuanced evidence match *and* the literal keyword check — without
juggling two tools.

## 3. How it works — the 4 steps (parse → run → merge → render)

This is the universal Layer-2 pattern (see Concept 3). Here it is for this specific tool:

```
match_job_requirements(resume_json, job_json)
        │
        ▼
1. PARSE   Resume.model_validate_json(resume_json)
           JobDescription.model_validate_json(job_json)
           └─ if either is bad JSON: return a clear "could not parse" string (never crash)
        │
        ▼
2. RUN     ┌─► match_requirements(resume, job)                         (JUDGMENT, 1 LLM call)
           │     classifies each requirement: matched / partial / gap
           │     severity from the requirement's importance:
           │        must_have -> blocker, should_have -> major, nice_to_have -> minor
           │        a partial match -> suggestion
           │     sets score = fraction of MUST-HAVE requirements matched (0.0–1.0)
           │
           └─► analyze_keyword_coverage( render_resume(resume), job.ats_keywords )  (MECHANICAL)
                 renders the resume to text, then whole-token-matches each JD keyword
                 reports coverage %, which keywords are absent, and keyword density
        │
        ▼
3. MERGE   _merge([ <judgment result>, <mechanical result> ])
           one ReviewResult holding ALL comments; keeps the lead engine's score
           (here, the must-have-coverage score from match_requirements)
        │
        ▼
4. RENDER  _render_review_result(merged, "Job Requirements Match")  ->  a string
           the Gap Analysis agent reads this and plans how to close the gaps
```

Note the small but important detail in step 2: the mechanical engine needs the resume as
**text**, so the tool calls `render_resume(resume)` (the shared helper in
`shared/resume_rendering.py`) to flatten the structured `Resume` into a string first.

## 4. What the agent sees (real output)

```
=== Job Requirements Match ===
The resume partially evidences Python but lacks Kubernetes; Go is not evidenced.; 50% keyword coverage
Score: 0.50
[suggestion/medium] (skills) Partial match for Kubernetes: listed, but no experience shown.
    advice: Add a project or role where you used Kubernetes (clusters, deployments).
[minor/high]       (skills) Gap for Go: no evidence found in the resume.
    advice: Add Go experience if you have it.
[major/high]       (other)  2 of 4 JD keywords absent
    advice: Work these JD keywords into bullets where true: Go, MLOps.
```

Read it through the Concept-1 lens: each line is a `ReviewComment` with a
`[severity/confidence]`, a `(section)`, a `message`, and `advice`. The `Score: 0.50` is
the must-have coverage carried up from the judgment engine. The first two findings came
from the judgment engine; the third came from the mechanical engine — but they merged
into one uniform report, which is exactly why the shared result shape matters.

## 5. Inputs and outputs

| | |
|---|---|
| **Inputs** | `resume_json` — a `Resume` serialized to JSON. `job_json` — a `JobDescription` serialized to JSON. (The agent/orchestrator serializes the objects; the tool parses them back.) |
| **Output** | A single string: a merged, readable report from both engines. |
| **Cost** | One LLM call (the judgment engine). The keyword half is free. If the job has no requirements, the judgment engine short-circuits with no LLM call. |

## 6. Gotchas

- **This is the riskiest judgment in the system.** Semantic equivalence is exactly where
  models confidently hallucinate ("SQL covers NoSQL"). The `match_requirements` engine's
  prompt is built around that: it must only claim a match when the resume *clearly*
  evidences it, prefer "partial" when unsure, and set `low` confidence on shaky calls.
  Downstream, only **high-confidence** findings should drive decisions; medium/low are
  advisory. Treat the requirement-match output as ~80% reliable, not gospel.
- **Two scores could exist, but only one survives.** Both engines can set a `score`
  (`match_requirements` = must-have coverage; `analyze_keyword_coverage` = keyword
  coverage). `_merge` keeps the **first** one (the judgment engine's), because must-have
  coverage is the headline metric here. This was a real bug once (the score got dropped)
  — keep the lead engine first in the `_merge` list.
- **Keyword matching is whole-token, not substring.** "AI" will not match inside
  "Maintained" (a past bug), but it also means "API" matches "APIs" via a deliberate
  plural rule. See the `analyze_keyword_coverage` engine doc for the details.

## 7. The same idea, in one line

*Parse the resume and job, ask the model how well each requirement is evidenced (the
nuanced half) and mechanically check which JD keywords literally appear (the exact half),
then merge both into one report for the Gap Analysis agent.*
