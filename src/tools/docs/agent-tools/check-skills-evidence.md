# Agent-Facing Tool — `check_skills_evidence`

**File:** `src/tools/agent_facing_tools.py`
**Function:** `check_skills_evidence(resume_json: str) -> str`
**CrewAI tool name:** "Check Skills Evidence"
**Type:** Judgment (it wraps the judgment `skills_evidence_validator` engine)
**Runs in:** both modes
**Used by:** the Skills Optimizer agent

---

## 1. Purpose (one sentence)

Give the Skills Optimizer agent a single tool that flags any listed skill the resume doesn't
actually back up.

## 2. Why it exists

The Skills Optimizer agent reorders and refines the skills section. Before it presents a skill, it
should know whether that skill is *defensible* — is there evidence for it elsewhere in the resume?
This tool answers that, wrapping the one engine that does the check so the agent gets a string it
can read.

## 3. How it works

```
check_skills_evidence(resume_json)
        │
        ▼
1. PARSE   Resume.model_validate_json(resume_json)   (bad JSON ──► clear error string)
        │
        ▼
2. RUN     validate_skills_evidence(resume)   (JUDGMENT, one LLM call)
              for each listed skill, is it backed by anything anywhere in the resume?
              skips self-evident skills; honest confidence on grey-area calls
        │
        ▼
3. (single engine — no merge)
        │
        ▼
4. RENDER  _render_review_result(result, "Skills Evidence") -> a readable string
```

## 4. Inputs and outputs

| | |
|---|---|
| **Input** | `resume_json` — a `Resume` serialized to JSON. |
| **Output** | A string titled "Skills Evidence"; "No issues found." when every skill is supported. |
| **Cost** | One LLM call; zero if no skills are listed (the engine short-circuits). |

## 5. What the agent sees (example)

```
=== Skills Evidence ===
The resume supports Python but lacks evidence for Kubernetes.
[major/high] (skills) The skill 'Kubernetes' lacks supporting evidence.
    advice: Either add experience that demonstrates the skill, or remove it.
```

## 6. Gotchas

- **Only checks listed-but-unbacked.** It does not suggest *new* skills the resume implies — that's
  the Skills Optimizer agent's own job. This tool only guards against claiming skills you can't
  back up.
- **An unbacked skill is `major`, not `blocker`.** It's a strong warning to fix before submission,
  not a hard stop — the candidate may genuinely have the skill and just need to add evidence.

## 7. The same idea, in one line

*A thin wrapper: parse the resume, ask the skills-evidence engine which listed skills nothing
supports, and return that list to the Skills Optimizer agent.*
