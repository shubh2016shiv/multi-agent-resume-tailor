# Engine — `job_requirement_extractor`

**File:** `src/tools/document_ingestion/job_requirement_extractor.py`
**Main function:** `extract_job_requirements(job_markdown: str) -> JobDescription`
**Type:** Judgment (one structured LLM call via `request_structured_output`)
**Runs in:** **Mode B only** (a job description was supplied)
**Used by:** the orchestrator (and conceptually the Job Analyzer agent)

> Pipeline-stage engine. Returns a typed **`JobDescription`**, not a `ReviewResult`. It is
> the exact mirror of `resume_section_extractor`, but for the job post instead of the resume.

---

## 1. Purpose (one sentence)

Turn a free-text job posting into a structured `JobDescription` — title, seniority, a list
of individual requirements (each with an importance level and any required years), and the
ATS keywords — so the job-matching engines have structured data to compare against.

## 2. Why it exists

In Mode B, the product measures the resume against a *specific* job. But a job posting is
unstructured prose. To ask "does the resume evidence each requirement?" you first need the
requirements as a clean list, each tagged with how important it is. That structuring is a
judgment task (the same requirement can be phrased a dozen ways; "must have" vs "a plus" has
to be inferred from wording), so the model does it, guided by the `JobDescription` schema.

Its output is what `match_requirements` and `analyze_keyword_coverage` consume — so without
this engine, Mode B has nothing to match against.

## 3. How it works

Identical shape to the resume extractor: one prompt, one gateway call.

```
extract_job_requirements(job_markdown)
        │
        ▼
  request_structured_output(JobDescription, JOB_EXTRACTION_PROMPT, job_markdown)
        │     the gateway forces and validates the JobDescription shape (retry-then-raise)
        ▼
  a validated JobDescription:
        job_title, company_name, job_level (entry..executive),
        requirements: [ {requirement, importance: must_have|should_have|nice_to_have, years_required?} ],
        ats_keywords: [...], summary, full_text
```

The prompt (`JOB_EXTRACTION_PROMPT`) tells the model how to classify each requirement's
importance from the wording ("required"/"must have" → `must_have`; "preferred" → `should_have`;
"a plus" → `nice_to_have`), to set `years_required` only when a specific number is stated,
and to draw `ats_keywords` from the posting itself rather than a generic list.

### No redaction step — and why

Unlike the resume path, the job text is **not** PII-redacted. A job posting is public
information (company, role, requirements) — it contains no candidate personal data — so it
goes straight to extraction.

## 4. Inputs and outputs

| | |
|---|---|
| **Input** | `job_markdown` — the job posting as text/Markdown. |
| **Output** | A validated `JobDescription`. |
| **Raises** | `RuntimeError` if the model can't produce a schema-valid `JobDescription` after one retry. |
| **Cost** | One LLM call. Only happens in Mode B. |

## 5. Who calls it

The orchestrator, when a job description is supplied. Its `JobDescription` output feeds the
`match_job_requirements` agent-facing tool (used by the Gap Analysis agent).

## 6. Gotchas

- **Extraction is non-deterministic.** Running the same posting twice can yield slightly
  different keyword sets or requirement phrasings — this is normal LLM variance. The
  *downstream* keyword and matching engines are deterministic given their input, so any
  run-to-run wobble in tailoring results originates here, not in those tools.
- **A valid-but-thin extraction is accepted silently** (e.g. no requirements parsed). A
  `TODO` in the file notes adding a follow-up check that warns the user when a posting
  couldn't be parsed into anything useful.

## 7. The same idea, in one line

*The Mode-B mirror of the resume extractor: one LLM call that reads the job posting and
fills in the `JobDescription` schema — requirements with importance levels and keywords —
so the matching engines have something structured to compare the resume against.*
