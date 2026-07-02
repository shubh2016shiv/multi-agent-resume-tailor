# Engine — `summary_quality_auditor`

**File:** `src/tools/resume_diagnostics/summary_quality_auditor.py`
**Main functions:** `audit_summary_quality(resume: Resume) -> ReviewResult`, `audit_summary_text(summary_text: str) -> ReviewResult`
**Type:** Hybrid — mechanical checks **plus** an always-on LLM call (Concept 2, "Shape 3b")
**Runs in:** both modes
**Used by:** the `audit_summary` agent-facing tool → the Summary Writer agent

---

## 1. Purpose (one sentence)

Audit the professional summary on four fronts: it's the right length, it's not written in
the first person, it isn't generic boilerplate, it opens with a real thesis, and it
actually states a value proposition.

## 2. Why it exists

The summary is the first thing a recruiter reads, and it's the most commonly botched section:
too long, written as "I am a results-oriented professional…", or so generic it could belong
to anyone. Two of these problems are mechanical facts (length; presence of "I"/"my"), and two
are judgment calls (is it generic? does it convey specific value?). So this engine does both
kinds of check and merges them — a textbook **hybrid**.

It's a *different* hybrid from `quantification_auditor`: there, the mechanical half *gates*
the LLM (skip the model if every bullet has a number). Here, **there is no mechanical proxy
for "generic"**, so the LLM call always runs for any non-empty summary. Both halves always
execute and their findings merge.

## 3. How it works

```
audit_summary_quality(resume)
        │
        ▼
  is the summary empty?  ── yes ──► one MAJOR finding "no summary", NO LLM call
        │ no
        ▼
  MECHANICAL half (HIGH confidence, free):
     length check:   under 80 words ──► MAJOR;  over 110 words ──► MAJOR
                     (both bounds are the writer task's hard constraints; the pipeline
                     gate blocks MAJOR+ only, so a MINOR floor was advisory in practice)
     person check:   contains "I"/"my"/"me"/... ──► MAJOR  (exact-token match, so 'academy'
                     never trips on 'my')
        │
        ▼
  JUDGMENT half (MEDIUM confidence, one LLM call — ALWAYS runs here):
     request_review("summary_quality_auditor", SUMMARY_RUBRIC, <the summary text>)
the model flags generic boilerplate, weak thesis, brochure tone, and a missing value proposition
        │
        ▼
  MERGE the mechanical findings + the model's findings into one ReviewResult
```

The constants `MIN_SUMMARY_WORDS = 80` and `MAX_SUMMARY_WORDS = 110` are deliberately aligned
with the Summary Writer agent's own target (80–110 words), so the *generator* and this
*auditor* never disagree and loop forever. The rubric explicitly tells the model NOT to
comment on length or pronouns — those are the mechanical half's job — so the two halves never
double-report.

## 4. Inputs and outputs

| | |
|---|---|
| **Input** | A `Resume`. Only `professional_summary` is read. |
| **Output** | A `ReviewResult` merging mechanical (`high`) and judgment (`medium`) findings. An empty summary yields a single `major` finding and makes **no** LLM call. |
| **Cost** | One LLM call for any non-empty summary; zero if the summary is empty. |

## 5. Who calls it

It is the single engine behind the `audit_summary` agent-facing tool, which the Summary
Writer agent calls.

## 6. Gotchas

- **The length thresholds are intentionally shared with the generator.** If you change them
  here without changing the Summary Writer agent, the two will fight. Keep them in sync.
- **First-person check is exact-token, not substring** — that's deliberate, so "my" doesn't
  match inside "academy". If you touch it, preserve the tokenisation.
- **A `score` is planned but not yet produced** (a `TODO`): eventually this engine could emit
  a 0–1 score to gate the Summary Writer's rewrite loop. No consumer reads it yet, so it's
  deferred.

## 7. The same idea, in one line

*Mechanically check length and first-person voice (certain, free), always ask the model
whether the summary is generic, formulaic, or weak on value (a judgment with no mechanical
shortcut), and merge both — the "both halves always run" flavour of hybrid.*
