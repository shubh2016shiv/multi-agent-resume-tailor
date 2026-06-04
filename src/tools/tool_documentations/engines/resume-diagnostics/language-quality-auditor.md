# Engine — `language_quality_auditor`

**File:** `src/tools/resume_diagnostics/language_quality_auditor.py`
**Main function:** `audit_language_quality(resume: Resume) -> ReviewResult`
**Type:** Judgment (one LLM call — no mechanical half)
**Runs in:** both modes
**Used by:** the `audit_experience_quality` agent-facing tool → the Experience Optimizer agent

---

## 1. Purpose (one sentence)

Flag bullets written as *duties* instead of *achievements* ("Responsible for…", "Worked
on…") and bullets full of hollow filler ("various tasks", "team player"), judged for the
candidate's actual field.

## 2. Why it exists

Weak resumes describe responsibilities; strong resumes describe accomplishments. "Responsible
for the deployment process" says nothing; "Cut deployment time 85% by automating releases"
says everything. This engine catches the weak phrasing.

But — and this is the key reason it's a *judgment* engine, not a list of banned phrases —
what counts as "duty language" depends on the field. "Responsible for FDA audit compliance"
is perfectly legitimate for a compliance officer. No frozen list of bad phrases can know
that; only a model that first infers the candidate's field can. So the detection itself *is*
the judgment, and there is no mechanical half.

## 3. How it works

A pure judgment engine: gather the bullets, ask the model, return the result.

```
audit_language_quality(resume)
        │
        ▼
  collect every achievement bullet across all roles
        │
        ▼
  no bullets? ──► ReviewResult([], "No experience bullets to review")  (no LLM call)
        │
        ▼
  number the bullets and send them to:
  request_review("language_quality_auditor", LANGUAGE_RUBRIC, <numbered bullets>)
        │     the ONE LLM call
        ▼
  ReviewResult — one comment per weak bullet, each with honest confidence
```

The rubric (`LANGUAGE_RUBRIC`) does the careful framing: it tells the model to *first infer
the field*, NOT to flag phrasing that is standard terminology in that field, and — crucially
— to set `confidence = low` when it is unsure whether something is field-appropriate. That
honest confidence is the safety valve against false flags in niche domains.

## 4. Inputs and outputs

| | |
|---|---|
| **Input** | A `Resume`. Only `work_experience[*].achievements` is read. |
| **Output** | A `ReviewResult` of `minor`-severity comments. Confidence is `high` for clear, field-independent duty language; `medium`/`low` when the call depends on domain knowledge the model is unsure about. |
| **Cost** | One LLM call (skipped if there are no bullets). |

## 5. Who calls it

One of the four engines inside `audit_experience_quality`. It pairs naturally with the
quantification engine: quantification asks "is there a number?", language asks "is this an
achievement or a duty?".

## 6. Gotchas

- **Trust the confidence, not just the presence.** Because "hollow"/"duty" is domain-subjective,
  a niche field the model doesn't understand will produce mostly `low`-confidence comments —
  which the system surfaces as advice, not mandates. That's by design; don't "fix" it by
  forcing high confidence.
- **No banned-phrase list, ever.** If you're tempted to add `cliches.json`, don't — it's the
  exact anti-pattern this engine was built to replace. The model holds the cross-domain
  knowledge; the rubric shapes how it's applied.

## 7. The same idea, in one line

*Ask the model — which first infers the candidate's field — to flag duty-language and hollow
phrasing, with honest confidence so niche-domain guesses stay advisory rather than forced.*
