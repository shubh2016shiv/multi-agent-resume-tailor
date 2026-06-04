# Engine — `consistency_auditor`

**File:** `src/tools/resume_diagnostics/consistency_auditor.py`
**Main function:** `audit_consistency(resume: Resume) -> ReviewResult`
**Type:** Mechanical (rule-based, using the spaCy NLP library — runs offline, no LLM)
**Runs in:** both modes
**Used by:** the `audit_experience_quality` agent-facing tool → the Experience Optimizer agent

> "Mechanical" includes local NLP libraries like spaCy. spaCy runs on your machine and makes
> no network/model-provider call, so it is *not* an LLM tool (Concept 2).

---

## 1. Purpose (one sentence)

Flag two readability problems *within a single role*: bullets that mix past and present
tense, and the same verb opening many bullets in a row.

## 2. Why it exists

Two small but jarring inconsistencies make a resume read as careless:

- **Mixed tense:** "Led the migration … and *am* currently maintaining …" inside one job.
  A role should pick one tense (past for a previous role, present for the current one).
- **Repetitive openings:** every bullet starting with "Managed … Managed … Managed …" reads
  monotonously.

Both are objective and domain-neutral, so they're mechanical. But *tense* in particular
cannot be done with a naive regex: irregular verbs ("Led", "Built", "Wrote") don't end in
"-ed". So this engine uses spaCy's part-of-speech tags, which classify a verb's tense
correctly regardless of spelling.

## 3. How it works

```
audit_consistency(resume)
        │
        ▼
  no work experience? ──► ReviewResult([], "No work experience to audit")
        │
        ▼
  CHECK A — tense consistency (per role)
     for each role:
        look at the FIRST verb of each bullet (spaCy POS tag: past vs present)
        does the role contain BOTH past and present openers? ──► MINOR
        "use one tense per role"
        │
        ▼
  CHECK B — repeated opening verbs (per role)
     for each role:
        count how many bullets start with each verb
        any verb opening 3+ bullets? ──► SUGGESTION
        "vary your opening verbs"
        │
        ▼
  ReviewResult with all findings (HIGH confidence)
```

The spaCy model (`en_core_web_lg`) is loaded lazily and cached the first time it's needed,
so importing the module is cheap. `REPEATED_VERB_THRESHOLD = 3` is the constant for check B.

## 4. Inputs and outputs

| | |
|---|---|
| **Input** | A `Resume`. Only `work_experience` is read. |
| **Output** | A `ReviewResult`; empty means bullets read consistently. All findings anchor to the experience section, all `high` confidence. |
| **Dependency** | The spaCy model `en_core_web_lg` must be installed (it's a declared project dependency). |

## 5. Who calls it

One of the four engines inside `audit_experience_quality`, merged with the structure,
quantification, and language findings.

## 6. Gotchas

- **spaCy, not regex, for tense — on purpose.** If you ever "simplify" the tense check to a
  regex on "-ed", you'll silently misclassify every irregular verb. Keep the POS-tag
  approach.
- **It checks consistency *within* a role, not across the whole resume.** Different roles can
  legitimately use different tenses (your current job in present, past jobs in past).

## 7. The same idea, in one line

*Using spaCy's grammar tags, flag a role that mixes past and present tense and any verb that
opens three or more bullets — objective readability checks, no model needed.*
