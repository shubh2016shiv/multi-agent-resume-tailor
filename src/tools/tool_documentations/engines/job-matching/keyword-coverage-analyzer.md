# Engine — `keyword_coverage_analyzer`

**File:** `src/tools/job_matching/keyword_coverage_analyzer.py`
**Main function:** `analyze_keyword_coverage(resume_text: str, required_keywords: list[str]) -> ReviewResult`
**Type:** Mechanical (string matching + math — no LLM)
**Runs in:** **Mode B only** (the keywords come from the job description)
**Used by:** the `match_job_requirements` and `analyze_jd_keyword_coverage` agent-facing tools

---

## 1. Purpose (one sentence)

Measure, literally, how many of the job's keywords appear in the resume, and whether they
appear at a healthy density (not absent, not stuffed).

## 2. Why it exists

Applicant Tracking Systems and recruiters scan for specific terms from the job. This engine
answers the exact, mechanical question: "of the keywords this job cares about, which actually
appear in the resume, and how often?" The keywords always come **from the job description**
(via the extractor) — never from a hardcoded list — so it works for any field.

It is the deliberately *literal* counterpart to `requirements_matcher`. That engine asks the
*nuanced* question ("does the resume *evidence* this requirement, even if worded
differently?"); this engine asks the *exact* question ("does this word appear?"). You need
both, and they're kept separate so each stays simple.

## 3. How it works

```
analyze_keyword_coverage(resume_text, required_keywords)
        │
        ▼
  empty resume text?  ──► ReviewResult([], "Empty input")
  no keywords given?  ──► ReviewResult([], "No keywords provided to match")
        │
        ▼
  count each keyword as a WHOLE TOKEN (see the matching rule below)
        │
        ▼
  compute metrics: coverage % (unique keywords found / total),
                   density (keyword instances / total words),
                   is the density inside the optimal 2%–5% band?
        │
        ▼
  build findings:
     keywords absent?              ──► MAJOR  "N of M JD keywords absent: ..."
     density outside 2%–5% band
       AND at least one keyword present? ──► MINOR  "density X% is outside the band"
        │
        ▼
  ReviewResult(comments, summary="X% keyword coverage", score=coverage)
```

### The matching rule (this is the subtle, important part)

Matching is **whole-token**, not substring. This matters enormously for short keywords:

- `"AI"` must **not** match inside `"Maintained"` (it contains the letters `ai`). A naive
  substring check would report 100% coverage from unrelated words — that was a real bug.
- But `"API"` **should** match `"APIs"` (a plural).

So the engine uses a regex with alphanumeric look-arounds — `"AI"` only matches when it isn't
flanked by another letter/digit — plus an optional trailing `s` for regular plurals. This is
also robust to punctuated tech keywords like `C#`, `C++`, and `.NET`, where the usual `\b`
word-boundary fails. (Irregular plurals like "repository"/"repositories" still miss; a `TODO`
notes spaCy lemmatization as a future option.)

The optimal density band (`MIN_KEYWORD_DENSITY = 0.02`, `MAX_KEYWORD_DENSITY = 0.05`) is the
"enough but not stuffed" range.

## 4. Inputs and outputs

| | |
|---|---|
| **Input** | `resume_text` (the resume as plain text) and `required_keywords` (a list, from the JD). |
| **Output** | A `ReviewResult` whose `score` is the coverage fraction (0–1). All findings `high` confidence. |
| **Note** | The file also keeps an older `calculate_keyword_density(...) -> str` `@tool` for backward compatibility; new code should use `analyze_keyword_coverage`. |

## 5. Who calls it

Two agent-facing tools: `analyze_jd_keyword_coverage` (this engine alone) and
`match_job_requirements` (this engine + the requirements matcher). In the latter, the
resume is first flattened to text with `shared.render_resume`.

## 6. Gotchas

- **Whole-token, not substring — never revert this.** The substring version silently
  false-passed on `AI`/`ML`/`R`/`API`. If you change the matching, keep the look-around +
  plural behaviour and re-check the short-keyword cases.
- **Density warning is suppressed when nothing matched.** A "0% density" complaint when zero
  keywords are present is noise (the "keywords absent" finding already says it), so the
  density finding only fires when at least one keyword was actually found.

## 7. The same idea, in one line

*Whole-token-match each job keyword against the resume (so "AI" doesn't match "Maintained" but
"API" matches "APIs"), then report coverage %, missing keywords, and whether density is in the
healthy 2–5% band.*
