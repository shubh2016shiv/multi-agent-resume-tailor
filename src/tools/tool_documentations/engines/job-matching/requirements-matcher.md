# Engine — `requirements_matcher`

**File:** `src/tools/job_matching/requirements_matcher.py`
**Main function:** `match_requirements(resume: Resume, job: JobDescription) -> ReviewResult`
**Type:** Judgment (one LLM call)
**Runs in:** **Mode B only**
**Used by:** the `match_job_requirements` agent-facing tool → the Gap Analysis agent

> The agent-facing tool that wraps this engine (`match_job_requirements`) has its own doc
> showing the *coordination*. This page is about the engine itself.

---

## 1. Purpose (one sentence)

For each requirement in the job, decide whether the resume **fully evidences it, partially
covers it, or has a gap** — and report each partial/gap as a finding, with a score for how
many must-have requirements are met.

## 2. Why it exists

This is the heart of job tailoring: "how much of this job does the resume actually answer?"
It is fundamentally a *semantic* question. "The job needs FastAPI; the resume shows Flask" —
that's a partial match, not a miss, because the two are adjacent. "The job needs NoSQL; the
resume shows SQL" — that is **not** a match, even though they sound similar. Only a model can
weigh that kind of equivalence, so this is a judgment engine.

It is also, by the project's own honest assessment, the **riskiest** engine — semantic
equivalence is exactly where models confidently hallucinate. The whole design is built to
contain that risk (see "How it works" and "Gotchas").

## 3. How it works

```
match_requirements(resume, job)
        │
        ▼
  job has no requirements?  ──► ReviewResult([], "no requirements to match")  (no LLM call)
        │
        ▼
  build a prompt payload containing:
     • the job requirements, each tagged [must_have | should_have | nice_to_have] and any
       "5+ yrs" figure                              (via a small formatter)
     • the resume rendered as evidence text         (via shared.render_resume)
        │
        ▼
  request_review("requirements_matcher", REQUIREMENTS_RUBRIC, <payload>)
        │     the ONE LLM call
        ▼
  ReviewResult where:
     • each GAP becomes a finding whose SEVERITY comes from the requirement's importance:
          must_have -> blocker,  should_have -> major,  nice_to_have -> minor
     • each PARTIAL match becomes a SUGGESTION ("strengthen this evidence")
     • a full match produces NO finding (nothing to fix)
     • score = fraction of MUST-HAVE requirements that are fully matched (0.0–1.0)
```

### How the risk is contained

The rubric (`REQUIREMENTS_RUBRIC`) is engineered around the model's known failure mode:

- Claim a **match only when the resume clearly evidences it**; treat adjacent-but-different
  tech as *partial*, not full ("SQL does NOT cover NoSQL").
- When unsure, set `confidence = low` and prefer "partial" over a confident "match."
- Accept that this judgment tops out around ~80% accuracy — so only **high-confidence**
  findings should drive downstream decisions; medium/low are advisory.

## 4. Inputs and outputs

| | |
|---|---|
| **Inputs** | A `Resume` and a `JobDescription`. |
| **Output** | A `ReviewResult` with one finding per partial/gap, severity scaled to importance, and a `score` = must-have coverage. |
| **Cost** | One LLM call (skipped if the job has no requirements). |

## 5. Who calls it

The `match_job_requirements` agent-facing tool, which pairs this engine's semantic verdict
with the mechanical `analyze_keyword_coverage`, then hands the merged report to the Gap
Analysis agent.

## 6. Gotchas

- **Treat the output as ~80% reliable, not gospel.** This is the one engine where the model
  is most likely to be confidently wrong. The confidence levels exist precisely so the system
  can lean on the sure findings and treat the shaky ones as suggestions.
- **The `score` is model-computed and can be internally inconsistent.** A planned caller-side
  check (noted in a `TODO`): distrust a `score == 1.0` if a `blocker` finding is also present —
  the two contradict each other.
- **`years_required` is judged, not math.** When a requirement says "5+ years", the model
  weighs it from the evidence; there is no mechanical year-counting (a deliberate v1 choice).

## 7. The same idea, in one line

*Ask the model to classify every job requirement as matched / partial / gap against the
resume's evidence — scaling each gap's severity to how important the requirement is, scoring
must-have coverage, and leaning hard on honest confidence because this is where models
hallucinate most.*
