# Engine — `formatting_validator`

**File:** `src/tools/ats_compliance/formatting_validator.py`
**Main function:** `audit_ats_formatting(resume_text: str) -> ReviewResult`
**Type:** Mechanical (regex + unicode category checks — no LLM)
**Runs in:** both modes
**Used by:** the `validate_ats_compliance` agent-facing tool → the ATS Optimization and QA agents

---

## 1. Purpose (one sentence)

Find formatting that breaks the automated parsers (ATS) most companies use to read
resumes — tables, tabs, embedded images, exotic characters, and the like.

## 2. Why it exists

Before a human ever sees a resume, an Applicant Tracking System tries to parse it into
fields. Certain layout choices break that parsing: a two-column layout, a table, tab
characters, an emoji, or `(cid:73)` artifacts from a bad PDF export. The content can be
excellent, but if the machine can't read it, the candidate is filtered out. This engine flags
those structural hazards so they can be removed *before* submission. It is purely about
*structure*, not section names (that's the next engine) and not wording.

## 3. How it works

It runs several independent pattern checks over the raw text and collects every hit.

```
audit_ats_formatting(resume_text)
        │
        ▼
  empty input? ──► ReviewResult([], "Empty input: nothing to audit")
        │
        ▼
  run each check; collect a finding per problem found:
     • pipe-character tables  ( |...| ),  tab characters,  <table>, <img>
     • 3+ consecutive blank lines
     • exotic symbol characters (emoji, arrows, box-drawing) — detected by UNICODE CATEGORY,
       not a hardcoded list, so it generalises; common bullets •, -, * are allowed
     • PDF font artifacts ( "(cid:N)" )
     • likely multi-column layout (heuristic: many very short, fragmented lines)
     • masked hyperlinks ( [GitHub](http://...) where the visible text hides the URL )
        │
        ▼
  ReviewResult — each problem is a MAJOR, document-level finding (section = OTHER), HIGH confidence
```

The exotic-character check is worth noting: instead of shipping a list of "bad characters", it
asks Python's unicode database for each character's *category* (e.g. "symbol-other"), so it
catches symbols it has never seen — while explicitly allowing the normal bullet/dash
characters real resumes use.

## 4. Inputs and outputs

| | |
|---|---|
| **Input** | `resume_text` — the resume as plain text or Markdown (often the *raw converted* document, so the check sees the real layout). |
| **Output** | A `ReviewResult`; empty means no formatting hazards. Findings are `major`, `high` confidence, anchored at the document level (`section = OTHER`) because most have no single line. |
| **Note** | The file also keeps an older `validate_ats_formatting(...) -> str` `@tool`; new code uses `audit_ats_formatting`. |

## 5. Who calls it

One of the two engines behind `validate_ats_compliance` (the other being
`section_header_validator`). That tool is used by the ATS Optimization agent and by Quality
Assurance.

## 6. Gotchas

- **The multi-column check is a heuristic** (lots of short lines ⇒ probably columns). It can
  false-positive on, say, a long single-column skills list of short items. A `TODO` in the
  file acknowledges this; treat that one finding as a hint, not a certainty.
- **Run it on the *raw* text, not the structured `Resume`.** Once text is parsed into a
  `Resume` object the formatting hazards are gone — so this check is meaningful only on the
  document text.

## 7. The same idea, in one line

*Regex- and unicode-category checks for the structures that break ATS parsers — tables, tabs,
images, exotic symbols, PDF artifacts, multi-column layouts, masked links — each reported as a
high-confidence, document-level finding.*
