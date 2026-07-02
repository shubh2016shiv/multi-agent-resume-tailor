# Agent-Facing Tool — `audit_summary`

**File:** `src/tools/agent_facing_tools.py`
**Function:** `audit_summary(summary_text: str) -> str`
**CrewAI tool name:** "Audit Summary Quality"
**Type:** Hybrid (it wraps the hybrid `summary_quality_auditor` engine)
**Runs in:** both modes
**Used by:** the Summary Writer agent

---

## 1. Purpose (one sentence)

Give the Summary Writer agent a single tool that reviews the professional summary for length,
first-person voice, generic boilerplate, and a missing value proposition.

## 2. Why it exists

The summary is one focused section, served by exactly one engine (`summary_quality_auditor`). This
tool is the thin, agent-friendly wrapper around it: it accepts the summary text the writer just
generated, runs the engine, and returns a readable string (what an agent reads). It's the simplest
kind of agent-facing tool — a one-engine wrapper — and it now matches the Summary Writer's actual
working context instead of requiring a full resume payload the agent does not have.

## 3. How it works

```
audit_summary(summary_text)
        │
        ▼
1. RUN     audit_summary_text(summary_text)
              mechanical: length + first-person pronouns  (HIGH confidence)
              judgment (always-on LLM): generic? weak thesis? brochure tone?  (MEDIUM confidence)
        │
        ▼
2. RENDER  _render_review_result(result, "Summary Quality") -> a readable string
```

Because the backing engine is itself a hybrid (the "both halves always run" flavour — see its
engine doc), this tool inherits that type.

## 4. Inputs and outputs

| | |
|---|---|
| **Input** | `summary_text` — the professional summary text to review. |
| **Output** | A string titled "Summary Quality". |
| **Cost** | One LLM call for any non-empty summary; zero if the summary is empty (the engine short-circuits with a single "no summary" finding). |

## 5. What the agent sees (example)

```
=== Summary Quality ===
2 summary issue(s)
[major/high]   (summary) Summary is 22 words, under the 80-word floor
    advice: Expand toward 80-110 words so the summary can land a clear thesis and evidence.
[minor/medium] (summary) The summary is generic and states no clear value proposition.
    advice: Name your domain, a concrete achievement, and what sets you apart.
```

## 6. Gotchas

- **Length thresholds match the Summary Writer's own target on purpose.** The auditor and the
  generator share the 80–110 word goal so they never loop forever disagreeing. If you retune one,
  retune both.
- **Mechanical and judgment findings never double up.** The engine's rubric forbids the model from
  commenting on length/pronouns (the mechanical half's territory), so you won't see the same issue
  reported twice.

## 7. The same idea, in one line

*A thin wrapper: take the generated summary text, run the one summary engine (mechanical
length/voice + an always-on LLM check for thesis/value/tone), and return its report to the
Summary Writer agent.*
