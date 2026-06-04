# Engine — `pii_redactor`

**File:** `src/tools/document_ingestion/pii_redactor.py`
**Main function:** `redact_pii(markdown: str) -> tuple[str, dict[str, str]]`
**Type:** Mechanical (Microsoft Presidio — local NLP, no LLM)
**Runs in:** both modes — after conversion, **before** extraction
**Used by:** the orchestrator directly

> Pipeline-stage engine. It does **not** return a `ReviewResult`; it returns the redacted
> text plus a mapping. See "Inputs and outputs" below.

---

## 1. Purpose (one sentence)

Mask personal data (names, emails, phones, addresses, ID numbers) in the resume text
**before any of it is sent to an external LLM**, and hand back a map so the orchestrator
can put the real values back at the very end.

## 2. Why it exists

A resume is sensitive personal data. The extraction step (and any review step) sends text
to an external model provider. Sending raw names, phone numbers, and home addresses to a
third party is a privacy problem. So this engine runs *first*, replacing each piece of PII
with a stable placeholder like `[PERSON_1]` or `[EMAIL_ADDRESS_1]`. The LLM then only ever
sees placeholders. At the very end of the pipeline, the orchestrator uses the returned map
to swap the real values back into the finished resume.

It uses **Microsoft Presidio**, not hand-written regex, because names and addresses simply
cannot be matched reliably by patterns — Presidio combines NLP (named-entity recognition)
with validated recognizers for structured IDs.

## 3. How it works

```
redact_pii(markdown)
        │
        ▼
  run Presidio over the text, looking ONLY for the entity types in REDACTED_ENTITIES
        │     (PERSON, EMAIL_ADDRESS, PHONE_NUMBER, LOCATION, URL, SSN, passport, ...)
        ▼
  drop low-confidence hits (score < 0.5)
        │
        ▼
  resolve overlaps: if two detections overlap (e.g. 'gmail.com' inside 'a@gmail.com'),
  keep the higher-confidence, longer one — so spans never collide
        │
        ▼
  assign a STABLE placeholder per unique value:
        "Jane Doe" -> [PERSON_1],  the SAME value always gets the SAME placeholder
        │
        ▼
  replace each span right-to-left (so earlier offsets stay valid)
        │
        ▼
  return (redacted_text, {placeholder: original_value})
```

### Two deliberate design choices worth understanding

- **Some entities are *excluded on purpose.*** `DATE_TIME` and `ORGANIZATION` are **not**
  redacted, because employment dates and employer names are needed by the review engines.
  Redacting them would blind the rest of the pipeline. Only truly personal data is masked.
- **Two custom recognizers are added:** `DATE_OF_BIRTH` (a date is only masked when
  birth-context words like "dob"/"born" are nearby, so ordinary employment dates survive)
  and `AGE` (e.g. "32 years old"). These are built in `_build_date_of_birth_recognizer` and
  `_build_age_recognizer`.

## 4. Inputs and outputs

| | |
|---|---|
| **Input** | `markdown` — the converted resume text. |
| **Output** | A tuple `(redacted_text, mapping)`. `redacted_text` has placeholders in place of PII; `mapping` is `{placeholder: original_value}`. |
| **Statelessness** | The engine keeps no state. The **orchestrator** owns the returned `mapping` and is responsible for rehydrating the final document. |
| **Empty input** | Returns the input unchanged with an empty map. |

## 5. Who calls it

The orchestrator, between conversion and extraction. The orchestrator stores the `mapping`
and, after the resume has been improved and rendered, uses it to restore the real name,
email, etc. (Rehydration is *state the orchestrator owns*, not something this engine does.)

## 6. Gotchas

- **Why placeholders are stable:** the same original value always maps to the same
  placeholder, so when the extractor copies `[PERSON_1]` into the resume's `full_name`
  field, the orchestrator can later swap it back unambiguously.
- **Known gaps (documented as `TODO`s in the file):** education completion years are not
  redacted (sections aren't known yet at this raw stage), and India-specific IDs (Aadhaar,
  PAN) aren't in Presidio's default set. Add custom recognizers only when real resumes
  need them.
- **Confidence threshold matters.** Hits below `0.5` are dropped; the custom DOB pattern is
  intentionally scored low so a *bare* date is not redacted unless birth-context words push
  it over the line.

## 7. The same idea, in one line

*Before any text leaves for the LLM, replace personal data with stable placeholders using
Presidio (keeping employer names and employment dates, which the review needs), and return
a map so the orchestrator can restore the real values at the end.*
