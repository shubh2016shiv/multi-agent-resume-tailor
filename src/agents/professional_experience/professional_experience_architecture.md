# Professional Experience MAS Architecture

## How Writing, Review, and Rewrite Should Work in a Reliable Multi-Agent System

> Scope: Professional Experience agent inside the Resume Tailor MAS
> Audience: GenAI engineers, backend engineers, and MAS architects
> Core idea: the LLM writes; code controls review, rewrite, IDs, and merge

---

## Table of Contents

1. [Why This Problem Matters in MAS](#1-why-this-problem-matters-in-mas)
2. [The General MAS Failure Pattern](#2-the-general-mas-failure-pattern)
3. [The Reliable MAS Pattern](#3-the-reliable-mas-pattern)
4. [Professional Experience Agent in This Project](#4-professional-experience-agent-in-this-project)
5. [Current Runtime Flow](#5-current-runtime-flow)
6. [The Write, Review, Rewrite Cycle](#6-the-write-review-rewrite-cycle)
7. [Where the Four Audit Functions Live](#7-where-the-four-audit-functions-live)
8. [Contracts and Data Ownership](#8-contracts-and-data-ownership)
9. [Context Engineering Rules](#9-context-engineering-rules)
10. [Known Weak Point: Quality Gate Policy](#10-known-weak-point-quality-gate-policy)
11. [Anti-Patterns](#11-anti-patterns)
12. [Quick Reference](#12-quick-reference)

---

## 1. Why This Problem Matters in MAS

In a small demo, an agent can do everything:

```text
read context -> write answer -> inspect itself -> retry -> final answer
```

In a production multi-agent system, that shape becomes unreliable. The agent is
not just writing. It is also making hidden decisions about quality, retry timing,
tool inputs, and what information should be trusted.

That is the dangerous part.

The professional experience agent is especially tricky because it is not a
simple extractor. It performs a writing cycle:

```text
plan what matters in a role
write achievement bullets
review writing quality
rewrite if the output is weak
return structured data for the rest of the resume pipeline
```

If the writer LLM owns the entire cycle, the system becomes hard to debug:

```text
Was the output bad because the context was bloated?
Was the tool call malformed?
Did the review run?
Did the review find anything?
Did the model ignore the review?
Did the retry happen?
Did the retry invent facts?
```

Reliable MAS does not depend on the agent answering those questions. Reliable
MAS makes those questions visible in code.

---

## 2. The General MAS Failure Pattern

### 2.1 The tempting but unreliable design

```text
┌────────────────────────────────────────────────────────────────────┐
│                         ONE AGENT DOES ALL                         │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Big context                                                        │
│     -> writer LLM                                                   │
│     -> writer decides whether to call a review tool                 │
│     -> writer constructs tool JSON arguments from its context       │
│     -> review tool returns text                                     │
│     -> writer decides whether to rewrite                            │
│     -> writer returns final output                                  │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

This looks elegant because it puts the loop inside the agent. It is unreliable
because the most important decisions are hidden inside the LLM conversation.

The common failures:

| Failure | What it looks like | Why it happens |
|---|---|---|
| Context bloat | Model misses the role evidence | Too much unrelated resume/job data |
| Tool argument failure | Tool receives schema-shaped junk | LLM had to build JSON from prose/TOON |
| Silent review skip | Output is accepted without real audit | LLM chose not to call the tool |
| Weak retry | Agent rewrites without addressing audit | Feedback is unstructured or ignored |
| Cost explosion | Many hidden tool calls and retries | No code-owned retry budget |
| False confidence | Final output looks valid but is weak | Pydantic validates shape, not quality |

### 2.2 The specific TOON-to-JSON failure

TOON is good for LLM context:

```text
resume_work_experience:
  work_experience:
  - job_title: "Apps Dev Programmer Analyst"
    company_name: "Citicorp Services India Pvt. Ltd."
    achievements:
    - "Created the new end to end pipeline..."
```

JSON is good for tool contracts:

```json
{
  "experiences": [
    {
      "job_title": "Apps Dev Programmer Analyst",
      "company_name": "Citicorp Services India Pvt. Ltd.",
      "achievements": ["Created the new end to end pipeline..."]
    }
  ]
}
```

The failure happens when the writer LLM receives TOON but is also asked to call a
JSON-string tool. The model has to convert TOON or draft text into tool JSON.
That is not the writer's job.

Correct rule:

```text
TOON belongs in LLM input.
Pydantic/JSON belongs in code-owned contracts.
Do not make the writer LLM bridge TOON -> JSON for tool calls.
```

---

## 3. The Reliable MAS Pattern

Reliable MAS treats agents as typed workers and orchestration as the controller.

```text
┌────────────────────────────────────────────────────────────────────┐
│                      CONTROLLER-OWNED LOOP                         │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  code selects one role                                              │
│     -> formatter builds compact TOON                                │
│     -> writer LLM returns OptimizedExperienceSection                │
│     -> code audits typed Experience objects                         │
│     -> code decides whether rewrite is needed                       │
│     -> repair LLM gets exact feedback if needed                     │
│     -> code restores ID and merges                                  │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

The agent is still useful. It does the work only a language model can do:

- rewrite weak bullets into achievement bullets
- choose strong verbs
- preserve tone and seniority
- integrate truthful keywords
- turn rough wording into resume language

Code does the work that should not be left to an LLM:

- split roles
- reduce context
- validate output shape
- run audit engines
- decide whether to retry
- preserve IDs
- merge results
- stop after a fixed retry budget

### 3.1 Design principle

```text
If the operation creates language, use the LLM.
If the operation controls state, contracts, IDs, audit, or retry budget, use code.
```

### 3.2 Enterprise MAS version

```text
┌────────────┐     ┌──────────────┐     ┌───────────────┐
│ Typed Data │ --> │ Context Text │ --> │ Writer Agent  │
└────────────┘     └──────────────┘     └──────┬────────┘
                                                │
                                                v
                                      ┌─────────────────────┐
                                      │ Pydantic Output     │
                                      └──────────┬──────────┘
                                                 │
                                                 v
                                      ┌─────────────────────┐
                                      │ Code-Owned Audit    │
                                      └──────────┬──────────┘
                                                 │
                    ┌────────────────────────────┴────────────────────────────┐
                    v                                                         v
           ┌────────────────┐                                      ┌────────────────┐
           │ Accept Output  │                                      │ One Rewrite    │
           └────────────────┘                                      └───────┬────────┘
                                                                            │
                                                                            v
                                                                 ┌──────────────────┐
                                                                 │ Final Typed Data │
                                                                 └──────────────────┘
```

---

## 4. Professional Experience Agent in This Project

The professional experience agent lives inside a larger resume-tailoring MAS.
It is not an independent product. Its output feeds downstream assembly and QA.

High-level ecosystem:

```text
Resume Parser
    -> Resume with work_experience[]

Job Analyzer
    -> JobDescription

Gap Analysis
    -> AlignmentStrategy

Professional Summary Writer
    -> ProfessionalSummary

Professional Experience Writer
    -> OptimizedExperienceSection

Skills Optimizer
    -> OptimizedSkillsSection

ATS Assembly
    -> OptimizedResume

Quality Assurance
    -> QualityReport
```

The professional experience step is Stage 3 in the orchestration flow. It runs
in parallel with summary and skills optimization.

```text
Stage 3:
  write_professional_summary
  optimize_experience
  optimize_skills
```

Within `optimize_experience`, orchestration fans out by role:

```text
Resume.work_experience[0] -> one writer call
Resume.work_experience[1] -> one writer call
Resume.work_experience[2] -> one writer call
...
```

The writer agent does not manage this parallelism. Orchestration does.

---

## 5. Current Runtime Flow

```text
┌───────────────────────────────────────────────────────────────────────────┐
│ Stage 3 / Step 3.1: Build the one-role writing agent                      │
│ Location: src/agents/professional_experience/agent.py                     │
│                                                                           │
│ Agent tools: []                                                           │
│ Agent task: write one role                                                │
│ Output: OptimizedExperienceSection                                        │
└───────────────────────────────────────────────────────────────────────────┘
                                      │
                                      v
┌───────────────────────────────────────────────────────────────────────────┐
│ Stage 3 / Step 3.2: Build one-role resume context                         │
│ Location: src/orchestration/nodes.py                                      │
│                                                                           │
│ Receives: one Experience from Resume.work_experience                      │
│ Sends: one-role Resume to experience_optimizer_formatter                  │
└───────────────────────────────────────────────────────────────────────────┘
                                      │
                                      v
┌───────────────────────────────────────────────────────────────────────────┐
│ Stage 3 / Step 3.3: Ask the LLM to write one role                         │
│ Location: src/orchestration/nodes.py                                      │
│                                                                           │
│ Receives: TOON context                                                    │
│ Sends: OptimizedExperienceSection through CrewAI output_pydantic          │
└───────────────────────────────────────────────────────────────────────────┘
                                      │
                                      v
┌───────────────────────────────────────────────────────────────────────────┐
│ Stage 3 / Step 3.4: Check the written role in code                        │
│ Location: src/tools/resume_diagnostics/__init__.py                        │
│                                                                           │
│ Receives: list[Experience] from OptimizedExperienceSection                │
│ Sends: ReviewResult to orchestration                                      │
└───────────────────────────────────────────────────────────────────────────┘
                                      │
                                      v
┌───────────────────────────────────────────────────────────────────────────┐
│ Stage 3 / Step 3.5: Decide whether one rewrite is needed                  │
│ Location: src/orchestration/nodes.py                                      │
│                                                                           │
│ Receives: ReviewResult                                                    │
│ Sends: accepted output or one repair CrewAI task                          │
└───────────────────────────────────────────────────────────────────────────┘
                                      │
                                      v
┌───────────────────────────────────────────────────────────────────────────┐
│ Stage 3 / Step 3.6: Restore the code-owned experience_id                  │
│ Location: src/orchestration/nodes.py                                      │
│                                                                           │
│ Receives: original Experience.experience_id and LLM-written Experience    │
│ Sends: merged OptimizedExperienceSection to ATS assembly                  │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 6. The Write, Review, Rewrite Cycle

### 6.1 What the writer agent owns

The writer owns language generation only.

It receives:

- one role's evidence
- compact job signals
- alignment guidance
- task rules

It returns:

- one `OptimizedExperienceSection`
- exactly one optimized `Experience`
- short optimization notes
- keywords actually integrated
- role relevance score

It does not own:

- tool calls
- audit timing
- rewrite trigger policy
- final merge
- ID creation or restoration

### 6.2 What orchestration owns

Orchestration is the manager of the writing cycle.

```text
write once
  -> audit typed output
  -> if quality gate fails, ask for one rewrite
  -> restore ID
  -> merge
```

The rewrite pass receives the original context plus the audit result:

```text
ORIGINAL TOON CONTEXT

PREVIOUS_OPTIMIZED_EXPERIENCE_JSON

EXPERIENCE_AUDIT_FEEDBACK

Instruction:
Rewrite once to address blocker or major audit findings.
Return only OptimizedExperienceSection JSON.
```

This is not an unbounded self-improvement loop. It is a bounded code-owned
correction cycle.

### 6.3 Why one rewrite only

One rewrite is enough for the current architecture because:

- it proves the audit can influence output
- it controls cost
- it avoids hidden agent loops
- it keeps failure modes visible

If output is still weak after one rewrite, the system should report the audit
decision rather than silently loop.

---

## 7. Where the Four Audit Functions Live

The four experience audit checks are not tools handed to the writer LLM. They
are code-facing diagnostics.

Final composition point:

```python
audit_experience_quality_for_experiences(experiences: list[Experience]) -> ReviewResult
```

Location:

```text
src/tools/resume_diagnostics/__init__.py
```

It calls:

```text
audit_bullet_structure_for_experiences
audit_consistency_for_experiences
audit_quantification_for_experiences
audit_language_quality_for_experiences
```

Diagram:

```text
OptimizedExperienceSection
        │
        ▼
optimized_experiences: list[Experience]
        │
        ▼
audit_experience_quality_for_experiences
        │
        ├── audit_bullet_structure_for_experiences
        ├── audit_consistency_for_experiences
        ├── audit_quantification_for_experiences
        └── audit_language_quality_for_experiences
        │
        ▼
ReviewResult
        │
        ▼
orchestration rewrite decision
```

### 7.1 What each audit check contributes

| Check | Type | What it catches | Why code owns it |
|---|---|---|---|
| bullet structure | mechanical | too few bullets, too many bullets, long bullets | deterministic shape check |
| consistency | mechanical/local NLP | tense mismatch, repeated verbs | no writer judgment needed |
| quantification | hybrid | bullets with no numbers, metric suggestions | typed audit after output exists |
| language quality | judgment | duty language, hollow phrasing | review LLM gets typed bullets, not tool JSON from writer |

The language and quantification checks may use an LLM internally. That is not
the same as the writer LLM calling a tool. The important distinction is:

```text
Bad:
  writer LLM builds JSON tool args from TOON

Good:
  code passes typed Experience objects to diagnostic functions
```

---

## 8. Contracts and Data Ownership

### 8.1 Contract map

| Contract | Location | Owner | Purpose |
|---|---|---|---|
| `Experience` | `src/data_models/resume.py` | resume data model | canonical role entry |
| `OptimizedExperienceSection` | `src/agents/professional_experience/models.py` | writer output | professional experience stage output |
| `ReviewResult` | `src/tools/review_contract/review_models.py` | diagnostics | audit findings |
| `experience_id` | assigned after resume extraction | code | correlate role output back to input |

### 8.2 Ownership rules

```text
LLM may write:
  description
  achievements
  optimization_notes
  keywords_integrated
  relevance_scores

LLM must preserve:
  job_title
  company_name
  dates
  location
  skills_used unless evidence supports change

Code owns:
  experience_id
  role splitting
  audit decision
  rewrite budget
  merge order
```

### 8.3 Why experience_id is restored by code

The LLM may return:

```json
"experience_id": null
```

That is acceptable in the writer output. The ID is not a writing task. Code
restores the original ID before merge:

```text
original Experience.experience_id
  -> optimized Experience.experience_id
```

---

## 9. Context Engineering Rules

### 9.1 What the writer receives

Each role-level writer call receives:

- one role only
- role title
- company
- dates
- description
- achievements
- skills used
- compact job requirements and ATS keywords
- compact alignment guidance

### 9.2 What the writer must not receive

The role writer should not receive:

- unrelated roles
- education
- certifications
- personal contact details
- full skills inventory
- full job posting boilerplate
- large gap-analysis JSON

These fields are not needed to rewrite one role. They increase hallucination
risk because the writer may pull a skill or claim into the wrong company.

### 9.3 Why role-level parallelism

Professional experience parallelism is by role, not by bullet.

Bullet-level parallelism is too small. Bullets within one role need shared
scope, chronology, non-duplication, and consistent tone.

Role-level parallelism is the right boundary:

```text
one role = enough context for coherent bullets
one role = small enough context for cheap LLM calls
one role = natural unit for audit and rewrite
```

---

## 10. Quality Gate Policy

### 10.1 The bug that was present

The original implementation triggered rewrites only on `BLOCKER` or `MAJOR`
severity. None of the four auditors can produce those severities:

| Auditor | Max severity |
|---|---|
| bullet_structure_auditor | MINOR, SUGGESTION |
| consistency_auditor | MINOR, SUGGESTION |
| quantification_auditor | SUGGESTION (LLM rubric) |
| language_quality_auditor | MINOR (LLM rubric) |

This made the rewrite cycle permanently dead. Every run went write→review→accept.

### 10.2 Current policy (implemented in nodes.py)

```text
rewrite if:
  any blocker or major finding exists
  OR any language quality finding exists (duty language, hollow phrasing)
  OR any bullet-count-below-target finding from bullet_structure_auditor
  OR two or more minor/suggestion findings across all auditors
```

The language-quality rule and the two-minor rule are what actually drive rewrites
given current auditor severity caps.

### 10.3 Why this is not a bandaid

The audit engines already produce structured findings with `engine_id` stamps.
The policy converts those findings into a decision in a single helper function.
No new framework, service layer, or agent is required. The fix is:

```text
ReviewResult -> _experience_audit_needs_rewrite() -> accept or one rewrite
```

---

## 11. Anti-Patterns

### 11.1 Do not attach the audit tool to the writer

Wrong:

```text
writer agent tools=[audit_experience_quality]
```

Why wrong:

```text
writer receives TOON
tool needs JSON
writer must serialize tool args
contract failures return
```

### 11.2 Do not pass full resume JSON to the writer

Wrong:

```text
full Resume JSON + full JobDescription JSON + full AlignmentStrategy JSON
```

Why wrong:

```text
lost-in-the-middle
unrelated-role leakage
higher cost
more unsupported keyword injection
```

### 11.3 Do not make model power the first fix

Using a stronger model does not fix a bad boundary. In one observed run,
`gpt-4o` returned Python-style literals (`None`, `True`) instead of strict JSON
and CrewAI failed conversion.

The right question is not:

```text
Which model is smart enough to survive the messy boundary?
```

The right question is:

```text
Which boundary makes the model's job small and verifiable?
```

### 11.4 Do not let the correction loop hide inside the agent

Wrong:

```text
agent writes
agent reviews itself
agent decides retry
agent rewrites
agent says final
```

Correct:

```text
agent writes
code reviews
code decides retry
agent rewrites once if asked
code accepts or reports findings
```

---

## 12. Quick Reference

### 12.1 Who owns what

| Responsibility | Owner |
|---|---|
| role splitting | orchestration |
| context reduction | formatter |
| bullet writing | professional experience LLM |
| output schema | Pydantic/CrewAI task |
| experience audit | diagnostics helper |
| rewrite decision | orchestration |
| rewrite budget | orchestration |
| `experience_id` | code |
| final merge | orchestration |

### 12.2 The healthy flow

```text
Experience
  -> one-role Resume
  -> TOON context
  -> writer LLM
  -> OptimizedExperienceSection
  -> audit_experience_quality_for_experiences
  -> ReviewResult
  -> rewrite decision
  -> restored experience_id
  -> merged OptimizedExperienceSection
```

### 12.3 The core lesson

Professional experience writing is not a single agent problem. It is a MAS
coordination problem.

The writer agent is only one worker in a bigger system. The reliable design is
to keep the worker focused and make orchestration responsible for the cycle.
