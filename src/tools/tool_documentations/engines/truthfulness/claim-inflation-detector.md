# Engine — `claim_inflation_detector`

**File:** `src/tools/truthfulness/claim_inflation_detector.py`
**Main function:** `detect_claim_inflation(original: Resume, revised: Resume) -> ReviewResult`
**Type:** Mechanical (spaCy NER + number normalization — **zero LLM**, and that is the point)
**Runs in:** both modes — after any rewrite
**Used by:** the `audit_truthfulness` agent-facing tool → the Quality Assurance agent

---

## 1. Purpose (one sentence)

After the system rewrites a resume, catch any **number or named entity that appeared in the
rewrite but does not exist anywhere in the original** — the clearest sign of a fabricated fact.

## 2. Why it exists

When an AI optimizer rewrites a bullet to sound stronger, it can quietly invent facts: a
metric ("serving 2M users"), a company, a credential. These are the highest-risk lies —
verifiable and disqualifying. They must be caught.

Here's the crucial design decision: this engine is **deliberately mechanical, with no LLM.**
Why? Because the obvious approach — ask a model "did you overstate this?" — is structurally
broken: the *same* model that just wrote "Led a platform serving 2M users" will happily
confirm it's fine, because it already rationalised that wording. A model checking its own
homework gives itself an A+. So this engine sidesteps the model entirely and just does
arithmetic on facts: a number or name that is in the revised version and *nowhere* in the
original is, by definition, new.

## 3. How it works

```
detect_claim_inflation(original, revised)
        │
        ▼
  render BOTH resumes to text (shared.render_resume), run spaCy over each
        │
        ▼
  NEW NUMBERS:
     extract numeric values from each (spaCy CARDINAL/MONEY/PERCENT/QUANTITY),
     NORMALISE them so "5K" == "5,000" == 5000
     any value in REVISED that is in NONE of the original's values ──► MAJOR finding
        │
        ▼
  NEW NAMED ENTITIES:
     extract risk entities (ORG, PERSON, GPE, PRODUCT, ...) from each
     any entity text in REVISED absent from ALL of the original's ──► MAJOR finding
        │
        ▼
  ReviewResult (HIGH confidence — token presence is a measurement, not an opinion)
```

It compares against the **entire** original (not bullet-by-bullet), so simply moving a real
number from one bullet to another doesn't false-flag. Number normalization (`5K` → `5000`,
`$1.2M`, `15%`) lives in a small helper so differently-written-but-equal numbers match.

## 4. Inputs and outputs

| | |
|---|---|
| **Inputs** | The `original` (source of truth) and the `revised` resume. |
| **Output** | A `ReviewResult` of `major` findings, all `high` confidence. Empty means the revision introduced no new number or named entity. |
| **Cost** | Zero LLM calls, ever. |

## 5. Who calls it

One of the two engines behind `audit_truthfulness`. It is the **factual** half; its partner
`rewrite_drift_detector` is the **semantic** half (see those docs and `audit-truthfulness.md`).

## 6. Gotchas

- **spaCy's category labels can be wrong, but the *flag* is still right.** For example "2M" is
  sometimes tagged `ORG` rather than a number — so the message may say "introduces ORG '2M'".
  The important thing (a new token appeared) is still correctly caught; only the label is off.
  A watch-item, not a bug.
- **Confidence is HIGH, but that's about the *measurement*, not the verdict.** "This token is
  new" is certain; whether it's truly a lie needs a human to confirm — which is why the advice
  says "confirm this figure is real." Severity is `major` (review-worthy), not `blocker`, so a
  spaCy mislabel can't hard-block on its own.
- **Spelled-out / irregular numbers can slip through** (e.g. "two million" as one un-parseable
  token). A `TODO` notes this; missing is safer than false-flagging.

## 7. The same idea, in one line

*Deterministically diff the numbers and names in the revised resume against the whole original
— anything new is flagged — sidestepping the "model marking its own homework" trap by never
asking a model at all.*
