# Engine — `skills_evidence_validator`

**File:** `src/tools/truthfulness/skills_evidence_validator.py`
**Main function:** `validate_skills_evidence(resume: Resume) -> ReviewResult`
**Type:** Judgment (one LLM call)
**Runs in:** both modes
**Used by:** the `check_skills_evidence` agent-facing tool → the Skills Optimizer agent

---

## 1. Purpose (one sentence)

For every skill the candidate lists, check that *something elsewhere in the resume* actually
backs it up — and flag any skill that nothing supports.

## 2. Why it exists

There are two failure modes that get a candidate rejected *after* they pass screening, and
this engine guards one of them: **unsupported skills.** A recruiter sees "Kubernetes" in the
skills list, asks about it in the interview, and the candidate has nothing to say — trust
gone. So before submission, every listed skill should be traceable to real evidence.

Why judgment, not a text search? Because evidence is rarely a literal repeat. "Kubernetes" can
be backed by a bullet about "container orchestration" with no literal mention; a "machine
learning" skill can be backed by a CS degree. A plain string search would both miss real
evidence *and* be fooled by a skill name that appears without being backed ("Familiar with
Kubernetes through reading" contains the word but is worse than unbacked). Only reading the
context settles it.

## 3. How it works

```
validate_skills_evidence(resume)
        │
        ▼
  no skills listed? ──► ReviewResult([], "No skills listed to verify")  (no LLM call)
        │
        ▼
  build a payload with TWO clearly-labelled parts:
     1. SKILLS TO VERIFY  — the listed skill names
     2. EVIDENCE FROM THE RESUME — summary + experience (descriptions, achievements,
        skills_used) + education + certifications     ◄── evidence is gathered BROADLY
        │
        ▼
  request_review("skills_evidence_validator", SKILLS_EVIDENCE_RUBRIC, <payload>)
        │     the ONE LLM call
        ▼
  ReviewResult — one MAJOR finding per skill that NOTHING in the resume supports
```

Two deliberate design points are baked into the rubric:

- **Evidence is searched broadly** — anywhere in the resume, not just the experience section —
  so a skill backed by a degree or certification is correctly counted as supported.
- **Self-evident skills are not flagged** — nobody needs "evidence" for "Microsoft Word".
  Flagging those would be noise. When the model is unsure whether a skill is implicitly covered
  by the field, it sets `confidence = low` rather than flagging confidently.

## 4. Inputs and outputs

| | |
|---|---|
| **Input** | A `Resume`. The skills are cross-checked against the rest of the resume as evidence. |
| **Output** | A `ReviewResult` of `major` findings (an unbacked skill is a credibility risk). Confidence is honest — `low` where the call depends on field knowledge. |
| **Cost** | One LLM call (skipped if no skills are listed). |

## 5. Who calls it

The single engine behind the `check_skills_evidence` agent-facing tool, used by the Skills
Optimizer agent.

## 6. Gotchas

- **It only checks the under-evidenced direction** (listed-but-unbacked). The *opposite*
  direction — the resume clearly shows a skill the candidate forgot to list — is the Skills
  Optimizer agent's job, not this engine's.
- **"Self-evident" is role-dependent** (a `TODO` notes this): "Microsoft Word" is noise for an
  embedded engineer but a real skill to evidence for a technical writer. The engine leans on
  the model's field inference plus `low` confidence to handle the grey area.
- **It deliberately does NOT use the shared `render_resume` helper.** Its evidence corpus must
  *exclude* the skills section (otherwise a skill would "evidence itself") and *include*
  per-role `skills_used` — so it has its own formatter. Don't "tidy" it onto the shared one.

## 7. The same idea, in one line

*Ask the model whether each listed skill is backed by anything anywhere in the resume —
ignoring self-evident skills, searching evidence broadly, and using honest confidence — so no
skill is claimed without support.*
