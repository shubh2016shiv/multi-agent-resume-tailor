# Engine — `extraction_quality_auditor`

**File:** `src/tools/document_ingestion/extraction_quality_auditor.py`
**Main function:** `audit_extraction_quality(markdown: str) -> ReviewResult`
**Type:** Mechanical (pure counts and ratios — no LLM)
**Runs in:** both modes — immediately after conversion
**Used by:** the orchestrator directly

---

## 1. Purpose (one sentence)

Catch a bad file conversion *early* — before the pipeline wastes an LLM call extracting a
`Resume` from garbage text — and tell the orchestrator whether to proceed or to ask the
user to re-upload.

## 2. Why it exists

`document_converter` always returns *something*, but "something" can be useless: a scanned
PDF might convert to almost no text, a broken export might produce
`"S o f t w a r e   E n g i n e e r"`, or a font problem might leave `(cid:73)` artifacts
everywhere. If that poisoned text flows downstream, every later step produces nonsense and
the user gets a baffling result. This engine is the **gate** that stops that: it looks at
the converted text and decides "usable" vs "re-upload needed."

It is the only ingestion engine that returns a `ReviewResult`, because its output is
genuinely a *finding* ("this conversion failed"), and a `BLOCKER`-severity finding is the
signal the orchestrator uses to halt.

## 3. How it works

Three independent mechanical checks; each either returns one finding or `None`.

```
audit_extraction_quality(markdown)
        │
        ├─► check 1: TEXT VOLUME
        │     fewer than 200 usable characters? ──► BLOCKER
        │     "conversion likely failed, re-upload"
        │
        ├─► check 2: FRAGMENTATION
        │     of all the words, what share are single letters?
        │     more than 30%? ──► MAJOR  (garbled spacing like "S o f t w a r e")
        │
        └─► check 3: FONT ARTIFACTS
              any "(cid:N)" tokens present? ──► MAJOR  (fonts didn't extract cleanly)
        │
        ▼
  collect whatever findings fired (often none)
        │
        ▼
  ReviewResult(comments=[...], summary="N issue(s), M blocker(s)" or "looks usable")
```

The thresholds (200 chars, 50-token minimum before the fragmentation check, 30% ratio)
are constants at the top of the file, marked with a `TODO` to calibrate against real
data. They are reasonable starting points, not measured values.

## 4. Inputs and outputs

| | |
|---|---|
| **Input** | `markdown` — the text from `document_converter`. |
| **Output** | A `ReviewResult`. **Empty** = the conversion looks usable. A `BLOCKER` finding = stop and ask the user to re-upload. `MAJOR` findings = the text is degraded but maybe salvageable. |
| **Confidence** | Always `high` — it's mechanical (Concept 1's rule). |

## 5. Who calls it

The orchestrator, right after `convert_document_to_markdown`. The orchestrator reads the
severities: a `BLOCKER` means do not proceed to extraction.

## 6. Gotchas

- **It judges *volume and shape*, not meaning.** It cannot tell a real but weak resume
  from a strong one — only whether the *text extraction itself* worked. Semantic quality is
  the diagnostics engines' job, much later.
- **Thresholds are unmeasured.** If you see false alarms on real resumes (e.g. a genuinely
  short one-page CV tripping the 200-char floor — unlikely, but possible), calibrate the
  constants; don't add special cases.

## 7. The same idea, in one line

*A cheap, mechanical smoke test on the converted text: near-empty, garbled, or
artifact-ridden output becomes a finding (a `BLOCKER` if it's basically empty) so the
pipeline stops before wasting work on garbage.*
