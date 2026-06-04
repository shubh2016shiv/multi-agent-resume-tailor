# Agent-Facing Tool — `audit_experience_quality`

**File:** `src/tools/agent_facing_tools.py`
**Function:** `audit_experience_quality(resume_json: str) -> str`
**CrewAI tool name:** "Audit Experience Quality"
**Type:** Hybrid (it bundles mechanical *and* judgment engines)
**Runs in:** both modes
**Used by:** the Experience Optimizer agent

> If you've read `concepts/03` and the `match-job-requirements` exemplar, this tool follows the
> same "parse → run engines → merge → render" pattern.

---

## 1. Purpose (one sentence)

Give the Experience Optimizer agent a single tool that reviews the work-experience section from
*four* angles at once — structure, consistency, quantification, and language.

## 2. Why it exists

The experience section has several independent weaknesses worth checking (too few bullets, mixed
tense, no numbers, duty language). Rather than hand the agent four separate tools — bloating its
prompt and risking it forgetting one — this tool runs all four engines and returns one combined
report. The agent asks one question, gets the full picture.

## 3. How it works (parse → run → merge → render)

```
audit_experience_quality(resume_json)
        │
        ▼
1. PARSE   Resume.model_validate_json(resume_json)
           (bad JSON ──► return a clear error string, never crash)
        │
        ▼
2. RUN     audit_bullet_structure(resume)   (mechanical: counts, lengths)
           audit_consistency(resume)        (mechanical: tense, repeated verbs — spaCy)
           audit_quantification(resume)      (HYBRID: no-number bullets -> LLM suggests a metric)
           audit_language_quality(resume)    (JUDGMENT: duty language, hollow phrasing -> LLM)
        │
        ▼
3. MERGE   _merge([...]) -> one ReviewResult with all four engines' findings
        │
        ▼
4. RENDER  _render_review_result(merged, "Experience Quality") -> a readable string
```

So this tool's "type" is **hybrid** because of what it bundles: two mechanical engines, one
hybrid, and one judgment. The agent doesn't need to know that breakdown — it just reads the
merged report.

## 4. Inputs and outputs

| | |
|---|---|
| **Input** | `resume_json` — a `Resume` serialized to JSON. |
| **Output** | A single string: a merged report titled "Experience Quality", one block per finding (`[severity/confidence] (section) message` + advice). |
| **Cost** | Up to two LLM calls (quantification, but only if there are number-less bullets; language, but only if there are bullets). A clean section can cost zero. |

## 5. What the agent sees (example)

```
=== Experience Quality ===
1 bullet structure issue(s); Bullets read consistently; ...; Two bullets use duty language...
[minor/high]       (experience) Most recent role has only 2 bullet(s)
    advice: Expand your most recent role to at least 3 achievement bullets.
[suggestion/medium] (experience) The bullet lacks a quantified result.
    advice: Add a metric such as number of models built.
[minor/high]       (experience) Duty language: states a responsibility, not an achievement.
    advice: Reframe as a concrete achievement with an outcome.
```

## 6. Gotchas

- **Findings from four engines, one report.** Because every engine returns the same `ReviewResult`
  shape (Concept 1), merging them is trivial and the agent can't tell which engine flagged what
  (nor does it need to — though `engine_id` is preserved if you inspect the objects).
- **It reads only the experience section.** Summary, skills, and ATS checks are separate tools.

## 7. The same idea, in one line

*Parse the resume, run the four experience engines (structure, consistency, quantification,
language), merge their same-shaped findings, and hand the Experience Optimizer one combined
report.*
