# Agent-Facing Tool — `audit_truthfulness`

**File:** `src/tools/agent_facing_tools.py`
**Function:** `audit_truthfulness(original_resume_json: str, revised_resume_json: str) -> str`
**CrewAI tool name:** "Audit Truthfulness"
**Type:** Hybrid (it bundles a mechanical engine and a judgment engine)
**Runs in:** both modes — after a rewrite
**Used by:** the Quality Assurance agent

---

## 1. Purpose (one sentence)

Give the Quality Assurance agent a single tool that checks whether a rewritten resume stayed
honest — catching both invented hard facts and softer semantic exaggeration.

## 2. Why it exists

After any agent rewrites the resume, someone must confirm the rewrite didn't lie. "Honesty" has two
faces that need two different tools: a *factual* face (a number or name that appeared from nowhere —
provable, caught mechanically) and a *semantic* face (an exaggeration with no new token — a judgment
call). This tool runs both engines and merges them so QA gets one verdict.

This is also the clearest example of *why* the two backing engines are split: one must be mechanical
(because a model can't be trusted to flag its own inflation), the other must be judgment (because the
exaggeration has no token to count). Bundling them gives full coverage.

## 3. How it works (parse → run → merge → render)

```
audit_truthfulness(original_resume_json, revised_resume_json)
        │
        ▼
1. PARSE   Resume.model_validate_json(original_...)  and  ...(revised_...)
           (bad JSON in either ──► one clear error string, never crash)
        │
        ▼
2. RUN     detect_claim_inflation(original, revised)   (MECHANICAL: new numbers/names — spaCy, no LLM)
           detect_rewrite_drift(original, revised)      (JUDGMENT: invented/exaggerated/lost — one LLM call)
        │
        ▼
3. MERGE   _merge([...]) -> one ReviewResult with both engines' findings
        │
        ▼
4. RENDER  _render_review_result(merged, "Truthfulness") -> a readable string
```

## 4. Inputs and outputs

| | |
|---|---|
| **Inputs** | `original_resume_json` (the source of truth) and `revised_resume_json` (the rewrite), both `Resume` objects serialized to JSON. |
| **Output** | A string titled "Truthfulness". Empty findings = the rewrite is faithful. |
| **Cost** | One LLM call (the drift engine), unless the two resumes are identical (then the drift engine short-circuits and the mechanical engine finds nothing). |

## 5. What the agent sees (example)

```
=== Truthfulness ===
2 introduced fact(s); The revision added an unsupported claim about serving 2M users...
[major/high]   (other)      Revision introduces the figure '40%', absent from the original
    advice: Confirm this figure is real; if it cannot be sourced, remove it.
[blocker/high] (experience) The revision claims '2M users, cut cost 40%', not supported by the original.
    advice: Restore the claim to what the original supports.
```

Notice both engines fired on the same inflation from different angles — the mechanical one caught the
literal new numbers, the judgment one caught the overall exaggeration. That overlap is intentional
defence in depth, not a bug.

## 6. Gotchas

- **It compares two versions, so it needs both.** Unlike the other tools (which take one resume),
  this one takes the *original* and the *revised*. The QA agent must supply both.
- **The mechanical half is the trustworthy one; the judgment half is a safety net.** Lean on the
  high-confidence factual findings; treat the semantic ones as a guard against the obvious, not a
  perfect lie detector.

## 7. The same idea, in one line

*Parse both versions, run the mechanical fact-diff and the judgment drift-check, and merge them so
Quality Assurance gets one honesty verdict on the rewrite.*
