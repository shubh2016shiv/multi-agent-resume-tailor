# Engine — `bullet_structure_auditor`

**File:** `src/tools/resume_diagnostics/bullet_structure_auditor.py`
**Main function:** `audit_bullet_structure(resume: Resume) -> ReviewResult`
**Type:** Mechanical (counting + word-length math — no LLM)
**Runs in:** both modes
**Used by:** `audit_experience_quality_for_experiences` in `src/tools/resume_diagnostics`

---

## 1. Purpose (one sentence)

Check that the experience section is *shaped* well: the most recent (most important) role
has enough bullets, no role is overloaded, and no single bullet has ballooned into a
paragraph.

## 2. Why it exists

A very common structural mistake: a candidate puts ten bullets on a job from ten years ago
and only two on their current, most relevant role — so a recruiter's eye lands on the wrong
place. Another: a "bullet" that runs 50 words is really a paragraph and won't be read. These
are *domain-neutral craft* problems — they're equally true for a nurse, a welder, or a quant
— so they can be measured mechanically, with no professional judgment and no model.

## 3. How it works

```
audit_bullet_structure(resume)
        │
        ▼
  no work experience at all? ──► ReviewResult([], "No work experience to audit")
        │
        ▼
  CHECK A — bullet counts
     find the most-recent role (latest start_date)
        has fewer than 3 bullets? ──► MINOR  "expand your most recent role"
     for every role:
        more than 8 bullets? ──► SUGGESTION  "trim to your strongest 8"
        │
        ▼
  CHECK B — bullet length
     for every bullet in every role:
        more than 35 words? ──► MINOR  "tighten; this reads as a paragraph"
        │
        ▼
  ReviewResult with all findings (all HIGH confidence — it's mechanical)
```

The numbers are constants at the top of the file: `MIN_BULLETS_RECENT_ROLE = 3`,
`MAX_BULLETS_PER_ROLE = 8`, `MAX_BULLET_WORDS = 35`. The 35-word cap is grounded ("a bullet
over ~35 words is a paragraph"); the count thresholds are reasonable starting points marked
with a `TODO` to calibrate.

## 4. Inputs and outputs

| | |
|---|---|
| **Input** | A `Resume`. Only `work_experience` is read. |
| **Output** | A `ReviewResult`; empty means the structure looks balanced. All findings anchor to the experience section, all `high` confidence. |

## 5. Who calls it

One of the four checks inside `audit_experience_quality_for_experiences`. Its structural
findings are merged with consistency, quantification, and language findings after the
professional experience writer returns typed `Experience` objects.

## 6. Gotchas

- **It judges shape, not content.** It can tell you a bullet is too long; it cannot tell you
  whether the bullet is *good*. Word choice and achievement-vs-duty are the language engine's
  job; missing numbers are the quantification engine's.
- **"Most recent" is by `start_date`,** so make sure dates are populated (they come from the
  extractor). The `Location` on a finding has no per-role index, so the role is named inside
  the message/quoted_text instead.

## 7. The same idea, in one line

*Pure counting: flag a thin most-recent role, an overloaded role, and any bullet long enough
to be a paragraph — domain-neutral structure checks, no model needed.*
