# Why the Professional Experience Agent Is Expensive

> **Root cause:** The trigger script passes full, unfiltered Pydantic JSON of all three inputs.
> The production orchestrator uses a formatter that strips 55–70%. The trigger uses zero formatting.
> Every iteration the agent calls `audit_experience_quality`, the LLM must re-copy a 10–15 KB
> resume JSON into a function call argument. `gpt-4o-mini` cannot do this reliably. The agent
> fails 12 times, burns tokens on each failure, and the context window balloons with every
> iteration.

---

## 1. What the Agent Receives Right Now

The trigger script (`trigger_professional_experience.py`) constructs the task description like this:

```python
Task(
    description=(
        f"RESUME DATA:\n{resume.model_dump_json()}\n\n"        # ← FULL resume, ~15 KB
        f"JOB DESCRIPTION DATA:\n{job.model_dump_json()}\n\n"  # ← FULL job, ~8 KB
        f"ALIGNMENT STRATEGY:\n{strategy.model_dump_json()}\n\n" # ← FULL strategy, ~5 KB
    ),
)
```

### What's actually in those JSON blobs

```
RESUME DATA (~15 KB):
├── full_name: "Shubham Singh"
├── email: "shubh2014shiv@gmail.com"
├── phone_number: "+919284178914"
├── location: "Noida, India"
├── website_or_portfolio: null
├── professional_summary: "<250-word paragraph>"
├── work_experience: [4 roles × (title, company, dates, description, achievements[], skills_used[])]
├── education: [3 entries × (institution, degree, field, year, gpa, honors)]
├── skills: [49 skills × (name, category, proficiency, years, justification, evidence, confidence)]
└── certifications: [...]

JOB DESCRIPTION DATA (~8 KB):
├── job_title: "Senior Backend Engineer"
├── company_name: "CloudTech Solutions"
├── job_level: "senior"
├── location: "San Francisco, CA (Hybrid)"
├── summary: "<3-sentence overview>"
├── full_text: "<ENTIRE job posting verbatim — 2 KB>"
├── requirements: [19 entries × (requirement, importance, years_required)]
└── ats_keywords: [22 keywords]

ALIGNMENT STRATEGY (~5 KB):
├── overall_fit_score: 42.5
├── summary_of_strategy: "<2-3 sentences>"
├── identified_matches: [SkillMatch objects]
├── identified_gaps: [SkillGap objects]
├── keywords_to_integrate: [17 keywords]
├── professional_summary_guidance: "<paragraph>"
├── experience_guidance: "<paragraph>"
└── skills_guidance: "<paragraph>"

TOTAL per task description: ~28 KB
```

### What the agent actually NEEDS for experience optimization

```
NEEDS:
├── work_experience[]  ← the 4 roles with their achievements
├── requirements[]     ← to know what skills to emphasize
├── ats_keywords[]     ← to integrate into bullets
├── experience_guidance ← from strategy
├── identified_gaps[]  ← to know what NOT to claim
└── keywords_to_integrate[] ← keyword targets

DOES NOT NEED:
├── full_name, email, phone, location      ← personal info
├── professional_summary                   ← not touching summary
├── education[]                            ← not touching education
├── skills[] (49 entries)                  ← not touching skills
├── certifications[]                       ← not touching certs
├── job.full_text (2 KB verbatim posting)  ← already extracted into requirements
├── job.company_name, location, summary    ← not needed
├── identified_matches[]                   ← already matched, nothing to change
├── professional_summary_guidance          ← for summary writer, not this agent
└── skills_guidance                        ← for skills agent, not this agent
```

**~80% of the data in the prompt is irrelevant to experience optimization.**

---

## 2. The Cost Explosion Per Iteration

The Professional Experience agent is designed to iterate: generate bullets → audit → improve → audit again. Each iteration:

### Iteration 1
```
Prompt:         28 KB (resume + job + strategy + task)
Tool call:      LLM tries to copy 15 KB resume JSON into function argument → FAILS
Context after:  28 KB + error message + re-prompt
Tokens burned:  ~7,000 tokens of context + function call attempt
```

### Iteration 2
```
Prompt:          28 KB + ~2 KB conversation history
Tool call:       LLM tries again → FAILS
Context after:   ~32 KB + more system prompt retry instructions
Tokens burned:   ~8,000 tokens
```

### By iteration 12 (what happened in the failing run)
```
Total context:   ~28 KB base + 11 × (2 KB error + tool output) ≈ 50 KB
Tokens burned:   12 × ~8,000 ≈ 96,000 tokens
LLM calls:       13 (12 tool failures + 1 reasoning call)
Model cost:      ~$0.015 (gpt-4o-mini) — affordable but WASTED
                 ~$0.25  (gpt-4o) — prohibitively expensive for a broken run
```

### If the tool call SUCCEEDED (hypothetical)
```
Iteration 1:  28 KB prompt + 15 KB tool output = 43 KB in history
Iteration 2:  43 KB history + 15 KB new tool output = 58 KB
Iteration 3:  58 KB + 15 KB = 73 KB
Iteration 4:  73 KB + 15 KB = 88 KB
                    ...
Each iteration ADDS 15 KB to the conversation permanently.
A 4-iteration run: ~88 KB context window → ~22,000 tokens → $0.0033 (mini) / $0.055 (4o)
A 8-iteration run: ~148 KB → ~37,000 tokens → $0.0056 / $0.092
```

The cost isn't just the base prompt. It's the **compounding conversation history** where every tool output stays in context forever.

---

## 3. Why the Production Orchestrator Doesn't Have This Problem

The real orchestrator (`agent_orchestrator.py`) uses **formatters** — dedicated modules that extract only relevant fields, strip metadata, convert to compact TOON format, and log token reduction.

```
TRIGGER SCRIPT (broken):
─────────────────────────────────────────────────────────
resume.model_dump_json()           → 15 KB raw JSON
job.model_dump_json()              →  8 KB raw JSON
strategy.model_dump_json()         →  5 KB raw JSON
                                   ─────────
TOTAL:                               28 KB


PRODUCTION ORCHESTRATOR (correct):
─────────────────────────────────────────────────────────
format_experience_optimizer_context(resume, job, strategy, format_type="toon")
│
├── _extract_work_experience_for_optimization(resume)
│     Only: work_experience[], removes personal info, education, skills
│     ~3 KB extracted from 15 KB
│
├── _extract_job_requirements_for_alignment(job)
│     Only: requirements[], ats_keywords[]
│     Removes: full_text (2 KB verbatim posting), company_name, location, summary
│     ~1.5 KB extracted from 8 KB
│
├── _extract_strategy_gaps_for_context(strategy)
│     Only: identified_gaps[], experience_guidance, keywords_to_integrate[]
│     Removes: professional_summary_guidance, skills_guidance, identified_matches[]
│     ~1.5 KB extracted from 5 KB
│
└── format_data(data, format_type="toon")
      Converts to compact TOON format (no JSON braces, quotes, commas)
      ~40-50% further reduction
                                   ─────────
TOTAL (formatted): ~3-5 KB

TOKEN REDUCTION: 55-70% logged by the formatter
```

### What the formatter removes (by field)

```
RESUME:
  ✅ work_experience[]                 ← KEPT (the only section this agent touches)
  ❌ full_name, email, phone, location ← REMOVED (personal info)
  ❌ professional_summary              ← REMOVED (summary writer's job)
  ❌ education[], certifications[]     ← REMOVED (not relevant)
  ❌ skills[] (49 entries)             ← REMOVED (skills agent's job)

JOB DESCRIPTION:
  ✅ requirements[]                    ← KEPT (what to align to)
  ✅ ats_keywords[]                    ← KEPT (what to integrate)
  ❌ full_text (2 KB verbatim)         ← REMOVED (already parsed)
  ❌ company_name, location, summary   ← REMOVED (not used)

ALIGNMENT STRATEGY:
  ✅ experience_guidance               ← KEPT (THIS agent's instructions)
  ✅ identified_gaps[]                 ← KEPT (what NOT to claim)
  ✅ keywords_to_integrate[]           ← KEPT (keywords to use)
  ❌ professional_summary_guidance     ← REMOVED (for summary writer)
  ❌ skills_guidance                   ← REMOVED (for skills agent)
  ❌ identified_matches[]              ← REMOVED (already matched)
  ❌ overall_fit_score                 ← REMOVED (QA context only)
  ❌ summary_of_strategy               ← REMOVED (not actionable)
```

---

## 4. The Tool Argument Problem

Even WITH formatted context, the `audit_experience_quality` tool still expects `resume_json: str` — a JSON-serialized string of the full resume. The LLM must copy a large string into a function call argument.

```
audit_experience_quality(resume_json: str)
                              │
                              └── LLM must construct this from context.
                                  gpt-4o-mini fails. gpt-4o succeeds but
                                  consumes tokens on every call.
```

### Why this design exists

The tool is designed to receive the FULL resume JSON because the four engines inside it
(bullet_structure, consistency, quantification, language_quality) each need different parts:

- `bullet_structure`: experience bullets
- `consistency`: experience bullets + opening verbs across all roles
- `quantification`: experience bullets (number-less ones)
- `language_quality`: experience bullets + full resume context (to judge if phrasing matches the role)

Engines 1-3 only need experience. Engine 4 benefits from full context. The tool accepts the
full resume to let engine 4 have context. But the cost is: the LLM must pass 15 KB on every call.

### Why the production orchestrator handles this

In the real orchestrator, tools are NOT called by the agent for extraction/conversion steps.
The orchestrator pre-extracts the resume via `convert_document_to_markdown` and
`extract_resume` as fixed pipeline steps. The agent only calls tools for QUALITY AUDITING.

But even then: the agent still needs to pass the resume JSON to `audit_experience_quality`.
The formatted context doesn't help here — the tool's parameter is `resume_json`, not the
task description text. The LLM must reconstruct the JSON string from its context.

---

## 5. The Real Fixes (ordered by impact)

### Fix A: Use the formatter (immediate, highest impact)

```python
# BEFORE (trigger script):
Task(description=f"RESUME DATA:\n{resume.model_dump_json()}\n\n...")

# AFTER:
from src.formatters.experience_optimizer_formatter import format_experience_optimizer_context

context = format_experience_optimizer_context(resume, job, strategy, format_type="toon")
Task(description=context)
```

**Impact:** Reduces initial prompt from 28 KB → ~5 KB. 80% reduction.
But does NOT fix the tool argument problem — `audit_experience_quality` still expects `resume_json: str`.

### Fix B: Add resume_json explicitly as a pre-serialized variable in the prompt

```python
resume_json = resume.model_dump_json()

Task(description=(
    f"{format_experience_optimizer_context(resume, job, strategy)}\n\n"
    f"RESUME_JSON (for tool calls):\n{resume_json}"
))
```

**Impact:** Gives the LLM an explicit, copyable string for tool calls. The formatted context handles the reasoning; the explicit JSON handles the tool argument. Combined ~30% further optimization.

### Fix C: Restructure the tool to accept formatted text instead of full JSON

The `audit_experience_quality` tool currently does:
```python
@tool("Audit Experience Quality")
def audit_experience_quality(resume_json: str) -> str:
    resume = Resume.model_validate_json(resume_json)  # parse full JSON
    return _render_review_result(
        _merge([audit_bullet_structure(resume), ...]),
        "Experience Quality"
    )
```

Only `audit_language_quality` benefits from full context. The other 3 engines only need experience bullets. Alternative: a two-tool design where the agent calls a LIGHTWEIGHT tool for bullet-only audits, and a full-context tool only for language quality. Or: have the tool accept `experience_json` instead of `resume_json` — just the experience section, ~3 KB instead of 15 KB.

### Fix D: Use a larger model for this specific agent

`gpt-4o` can reliably copy large JSON strings into function call arguments. `gpt-4o-mini` cannot. If the architecture stays as-is, the model must change. Cost: ~5-10× higher per call, but the agent will actually WORK instead of failing 12 times.

---

## 6. Summary

| Metric | Trigger Script (current) | With Formatter | With Formatter + Model Fix |
|---|---|---|---|
| Initial prompt | ~28 KB | ~5 KB | ~5 KB |
| Tool argument (per call) | ~15 KB | ~15 KB | ~15 KB |
| Agent iterations before failure | 12 (all failed) | Might fail | Should work |
| Tokens for failed 12-iteration run | ~96,000 | ~30,000 | 0 (no failed run) |
| Tokens for successful 4-iteration run | N/A (never succeeded) | ~22,000 | ~22,000 |
| Primary bottleneck | Prompt bloat + model capability | Tool argument size | Solved |

**The tool argument problem (`resume_json: str` = 15 KB per call) is the structural issue.**
**The prompt bloat (28 KB of irrelevant data) is the immediate, fixable issue.**
**The model choice (gpt-4o-mini for tool-heavy iteration) is the operational issue.**

All three compound. Fix A (formatter) is immediate and doesn't touch any agent code.
Fix C (tool restructuring) is the long-term solution but touches the shared tool layer.
Fix D (model selection) is a configuration change.
