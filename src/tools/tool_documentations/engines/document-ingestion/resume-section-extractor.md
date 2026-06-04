# Engine — `resume_section_extractor`

**File:** `src/tools/document_ingestion/resume_section_extractor.py`
**Main function:** `extract_resume(redacted_markdown: str) -> Resume`
**Type:** Judgment (one structured LLM call via `request_structured_output`)
**Runs in:** both modes — the last ingestion step
**Used by:** the orchestrator (and conceptually the Resume Extractor agent)

> Pipeline-stage engine. It returns a typed **`Resume`** object, not a `ReviewResult` —
> it's *building* the data everything else reviews. See `concepts/02`, "Judgment".

---

## 1. Purpose (one sentence)

Turn the messy, redacted resume Markdown into a clean, structured `Resume` object (summary,
work experience with bullets, education, skills) that every downstream engine can rely on.

## 2. Why it exists

Up to this point the resume is just *text*. But the diagnostics, truthfulness, and matching
engines all take a typed `Resume` object — they read `resume.work_experience[0].achievements`,
not a blob of Markdown. Something has to convert one into the other. That conversion is not
mechanical: real resumes vary wildly in layout, ordering, and wording, so a rule-based
parser would be brittle. Instead the model reads the text and fills in the `Resume` schema —
a task LLMs do very well, because the *schema itself* defines exactly what's wanted.

## 3. How it works

This is the simplest possible judgment engine: one constant prompt, one gateway call.

```
extract_resume(redacted_markdown)
        │
        ▼
  request_structured_output(Resume, RESUME_EXTRACTION_PROMPT, redacted_markdown)
        │     the gateway forces the model's reply to fit the `Resume` Pydantic schema,
        │     validates it, retries once on bad JSON, then raises if still bad
        ▼
  a validated Resume object
```

The prompt (`RESUME_EXTRACTION_PROMPT`) gives the model firm rules: use only what's in the
text (invent nothing), convert dates to ISO format, treat a missing end-date as the current
role, capture bullets as `achievements`, and leave absent fields empty rather than
fabricating them.

### Why it runs *after* redaction

By the time this runs, PII has already been replaced with placeholders. So the resulting
`Resume` carries `full_name == "[PERSON_1]"`, `email == "[EMAIL_ADDRESS_1]"`, etc. That is
intentional and correct — the orchestrator rehydrates the real values at the very end. The
model never sees the candidate's real personal data.

## 4. Inputs and outputs

| | |
|---|---|
| **Input** | `redacted_markdown` — resume text with PII already masked by `redact_pii`. (Precondition: redaction must have happened first.) |
| **Output** | A validated `Resume`. Personal fields hold redaction placeholders. |
| **Raises** | `RuntimeError` if the model cannot produce a schema-valid `Resume` after one retry. |
| **Cost** | Exactly one LLM call. |

## 5. Who calls it

The orchestrator, as the final ingestion step, producing the `Resume` that flows into every
review engine. (It is also the engine behind the "Resume Extractor" agent's job.)

## 6. Gotchas

- **Garbage in, garbage out.** This is why `audit_extraction_quality` runs *before* it — if
  the text is broken, extraction produces a broken `Resume`. Two known edge cases are
  recorded as `TODO`s in the file: a resume with *no dates at all* fails the required
  `start_date` field and raises; and a valid-but-empty extraction (e.g. no work experience)
  is accepted silently. Watch for both on real data.
- **Dates and "current role" are model judgments.** The prompt asks for ISO dates and
  null-end-date-means-current, but the model can still mis-infer an open-ended date range
  (this is the source of the "9.8 years vs stated 3 years" kind of discrepancy seen in
  end-to-end testing). Downstream code that relies on computed experience length should be
  aware the inference isn't perfect.

## 7. The same idea, in one line

*One LLM call that reads the redacted resume text and fills in the `Resume` schema —
inventing nothing, with PII still masked — so everything downstream has structured data to
work on.*
