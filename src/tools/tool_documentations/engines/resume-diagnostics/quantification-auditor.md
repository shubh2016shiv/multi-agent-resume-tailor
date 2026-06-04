# Engine — `quantification_auditor`

**File:** `src/tools/resume_diagnostics/quantification_auditor.py`
**Main function:** `audit_quantification(resume: Resume) -> ReviewResult`
**Type:** Hybrid (mechanical gate + one conditional LLM call) — see Concept 2, "Shape 3a"
**Runs in:** both modes (no job description needed)
**Used by:** the `audit_experience_quality` agent-facing tool → the Experience Optimizer agent

> If you have not yet read `concepts/01` (the result shape) and `concepts/02` (engine
> types), read those first — this page assumes them.

---

## 1. Purpose (one sentence)

Find the experience bullets that contain no number, and for each one suggest *what kind*
of metric would make it stronger — without ever inventing a number.

## 2. Why it exists (the real problem)

The single most common weakness in a weak resume is bullets with no measurable result.
Compare:

- ❌ "Responsible for improving the deployment process."
- ✅ "Cut deployment time from 40 minutes to 6 by automating the release pipeline."

Most candidates *know* they "should add metrics" but freeze on *what* to quantify. This
engine does two separate jobs to help:

1. **Detect** which bullets have no number. This is objective and must be reliable, so it
   is done with **pure code** — a bullet either contains a digit or it doesn't.
2. **Suggest** a fitting metric category for each weak bullet (team size? time saved?
   scale? cost? adoption?). This needs judgment about what's relevant, so it is a **single
   LLM call** — and crucially, the model suggests a *category with an example*, never an
   actual fabricated number.

Splitting it this way means the reliable part stays reliable, and the model is used only
where genuine judgment is needed.

## 3. How it works (the mechanism)

The mechanical half is a **gate**: it decides whether the LLM runs at all.

```
audit_quantification(resume)
        │
        ▼
  look at every achievement bullet across all work_experience roles
        │
        ▼
  keep only the bullets with NO digit          ◄── pure code: `any(ch.isdigit() ...)`
        │
        ├── list is EMPTY  ──►  return ReviewResult(comments=[],
        │                                summary="All experience bullets include a metric")
        │                       ★ NO LLM CALL IS MADE — this resume costs $0 here ★
        │
        └── list has N bullets
                 │
                 ▼
           format them as a numbered list
                 │
                 ▼
           request_review("quantification_auditor", QUANTIFICATION_RUBRIC, <the N bullets>)
                 │                                   (the ONE LLM call, only on the weak bullets)
                 ▼
           ReviewResult with one SUGGESTION / MEDIUM-confidence comment per weak bullet,
           each suggesting a metric category to add
```

The prompt (`QUANTIFICATION_RUBRIC`, a constant in the file) tells the model exactly what
to return for each bullet: a `suggestion`-severity, `medium`-confidence comment whose
advice names *one* metric category with a brief example, and explicitly forbids inventing
specific numbers.

Because only the already-detected weak bullets are sent, the model **cannot over-flag** —
it never even sees the bullets that already have numbers. And a fully quantified resume
skips the LLM entirely.

## 4. Inputs and outputs

| | |
|---|---|
| **Input** | A `Resume` object. Only `resume.work_experience[*].achievements` is read. |
| **Output** | A `ReviewResult`. Empty `comments` = every bullet already has a number (and no LLM call was made). Otherwise one `suggestion` / `medium` comment per number-less bullet. |
| **Confidence** | `medium` on every finding — it's a judgment suggestion, never `high`. |
| **Cost** | Zero LLM calls if the resume is fully quantified; otherwise exactly one call, regardless of how many bullets are weak (they are batched). |

A finding looks like this (see Concept 1 for the field meanings):

```
[suggestion/medium] (experience) The bullet lacks a quantified result.
    advice: Add a metric for time saved, such as deployment time reduction.
```

## 5. Who calls it

It is **not** handed to an agent directly. It is one of four engines bundled inside the
agent-facing tool `audit_experience_quality` (alongside `audit_bullet_structure`,
`audit_consistency`, and `audit_language_quality`). That tool is what the Experience
Optimizer agent actually calls; this engine's findings are merged with the other three
before the agent ever sees them. See
[../../agent-tools/audit-experience-quality.md](../../agent-tools/audit-experience-quality.md).

## 6. Gotchas and current limitations

- **Spelled-out numbers are missed.** Detection counts *digits* only, so "doubled
  throughput" or "tripled adoption" read as *unquantified* and get flagged. This is the
  safe direction (better to over-suggest a metric than to miss a number-less bullet), and
  it is recorded as a `TODO` in the file. If you add magnitude-word handling, do it in the
  mechanical `_has_number` check, not the prompt.
- **The model must never invent numbers.** The rubric forbids it. If you edit the prompt,
  keep that constraint — suggesting a *category* ("add a time-saved metric") is safe;
  inventing "reduced time by 40%" would be a fabrication the truthfulness engines would
  later have to catch.
- **It only reads `achievements`**, not the role's free-text `description`. That is
  intentional — descriptions are prose summaries, not achievement bullets.

## 7. The same idea, in one line

*Pure code finds the bullets with no number (and skips the model if there are none); the
model then suggests, for just those bullets, what kind of metric to add — never an actual
number.*
