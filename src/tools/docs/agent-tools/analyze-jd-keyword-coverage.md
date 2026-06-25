# Agent-Facing Tool — `analyze_jd_keyword_coverage`

**File:** `src/tools/agent_facing_tools.py`
**Function:** `analyze_jd_keyword_coverage(resume_text: str, keywords_csv: str) -> str`
**CrewAI tool name:** "Analyze Keyword Coverage"
**Type:** Mechanical (it wraps the mechanical `keyword_coverage_analyzer` engine)
**Runs in:** **Mode B only** (the keywords come from the job)
**Used by:** the ATS Optimization agent and the Quality Assurance agent

---

## 1. Purpose (one sentence)

Give the ATS Optimization (and QA) agent a single tool that reports how many of a job's keywords
literally appear in the resume, and at what density.

## 2. Why it exists

In Mode B, an agent often wants the *literal* keyword check on its own (separate from the deeper
requirement matching done by `match_job_requirements`). This tool is that standalone check: a thin
wrapper over the keyword engine. Its one extra job is **input shaping** — an agent naturally passes
a list of keywords as a comma-separated string, so this tool splits that string into a real list
before calling the engine.

## 3. How it works

```
analyze_jd_keyword_coverage(resume_text, keywords_csv)
        │
        ▼
1. SHAPE   split keywords_csv on commas -> ["Python", "Kubernetes", "Go", ...]
           (trim spaces; drop empties)
        │
        ▼
2. RUN     analyze_keyword_coverage(resume_text, keywords)   (mechanical, whole-token matching)
        │
        ▼
3. (single engine — no merge)
        │
        ▼
4. RENDER  _render_review_result(result, "Keyword Coverage") -> a readable string
```

No LLM call; every finding is `high` confidence. The engine's whole-token matching (so "AI" doesn't
match inside "Maintained" but "API" matches "APIs") is described in the keyword engine's doc.

## 4. Inputs and outputs

| | |
|---|---|
| **Inputs** | `resume_text` — the resume as text. `keywords_csv` — the job keywords as one comma-separated string. |
| **Output** | A string titled "Keyword Coverage", including the coverage `Score` (0–1), which keywords are absent, and a density note. |
| **Cost** | Zero LLM calls. |

## 5. What the agent sees (example)

```
=== Keyword Coverage ===
33% keyword coverage
Score: 0.33
[major/high] (other) 2 of 3 JD keywords absent
    advice: Work these JD keywords into bullets where true: Kubernetes, Go.
```

## 6. Gotchas

- **It's the *literal* check, not the *semantic* one.** "Does the word appear?" — not "does the
  resume evidence this requirement?". The latter is `match_job_requirements`. An agent doing gap
  analysis usually wants that richer tool; this one is for the pure keyword view.
- **Empty keyword list is handled gracefully** — the engine returns "No keywords provided to match"
  rather than a misleading 0% with a bogus density warning.

## 7. The same idea, in one line

*Split the comma-separated keyword string into a list, run the mechanical keyword engine, and return
the literal coverage report — the keyword-only view, distinct from the deeper requirement matching.*
