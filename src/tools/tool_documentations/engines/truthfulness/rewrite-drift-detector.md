# Engine — `rewrite_drift_detector`

**File:** `src/tools/truthfulness/rewrite_drift_detector.py`
**Main function:** `detect_rewrite_drift(original: Resume, revised: Resume) -> ReviewResult`
**Type:** Judgment (one LLM call)
**Runs in:** both modes — after any rewrite (it is **not** tied to job tailoring)
**Used by:** the `audit_truthfulness` agent-facing tool → the Quality Assurance agent

---

## 1. Purpose (one sentence)

Compare the original resume with its rewritten version and flag where the rewrite drifted from
the truth: **invented** claims, **exaggerated** claims, or important content that was **lost**.

## 2. Why it exists

`claim_inflation_detector` catches *hard, factual* drift mechanically (a number or name that
appeared from nowhere). But some drift is *semantic* and has no new token to catch — for
example "helped on three projects" rewritten as "led three cross-functional initiatives." No
new number, but a real exaggeration. That requires judgment, so this is the LLM half of the
truthfulness pair.

The engine is honest about its own ceiling: because a model is judging work a model produced,
it reliably catches *obvious* fabrications but can miss subtle reframings the optimizer already
rationalised. It's a **safety net, not a lie detector** — and the docs and code say so plainly.

## 3. How it works

```
detect_rewrite_drift(original, revised)
        │
        ▼
  render both resumes to text (shared.render_resume)
        │
        ▼
  are they IDENTICAL?  ── yes ──► ReviewResult([], "Revision is identical")  (NO LLM call)
        │ no
        ▼
  build a payload with both labelled versions:
        ORIGINAL RESUME: ...
        REVISED RESUME:  ...
        │
        ▼
  request_review("rewrite_drift_detector", REWRITE_DRIFT_RUBRIC, <payload>)
        │     the ONE LLM call
        ▼
  ReviewResult flagging three kinds of drift, with calibrated severity:
        invented fact (number/credential/role)  ──► BLOCKER  (or MAJOR for softer inventions)
        exaggeration                              ──► MAJOR
        lost content (a dropped achievement/role) ──► MINOR   (a quality regression, not a lie)
```

The rubric tells the model to **judge only against the original** (it is the source of truth),
to *not* reward the revision for sounding more impressive, and to set confidence honestly — so
the system can block on high-confidence inventions while treating shaky calls as advisory.

The identical-input short-circuit means that if nothing actually changed, the engine costs
nothing — no point asking the model to diff two identical texts.

## 4. Inputs and outputs

| | |
|---|---|
| **Inputs** | The `original` and the `revised` resume. |
| **Output** | A `ReviewResult` with honest confidence. Empty (and no LLM call) when the two render identically. |
| **Cost** | One LLM call, unless the revision is identical to the original. |

## 5. Who calls it

The other engine (with `claim_inflation_detector`) behind `audit_truthfulness`. Together they
cover both halves: factual drift (mechanical, certain) and semantic drift (judgment, honest
confidence).

## 6. Gotchas

- **It's a safety net, not a guarantee.** Don't expect it to catch every subtle inflation — by
  design it catches the obvious ones. The mechanical `claim_inflation_detector` is what gives
  certainty on hard facts.
- **Mode-independent on purpose.** It guards *any* rewrite, with or without a job description —
  which is why it was named `rewrite_drift` rather than anything implying job tailoring.

## 7. The same idea, in one line

*Ask the model to diff the original against the rewrite for inventions, exaggerations, and
losses — judging only against the original, blocking on confident fabrications, and skipping the
call entirely when nothing changed.*
