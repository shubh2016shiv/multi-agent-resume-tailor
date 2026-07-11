# Formatters — Building LLM Context, One Agent at a Time

This module owns one job: **take large, typed data objects and produce a
compact string the LLM reads as context.** Every formatter answers the same
question: "of everything we know, what does THIS agent actually need?"

---

## 1. The Problem This Module Solves

Imagine you're building a multi-agent pipeline. Each agent does one job:
write a summary, optimize skills, audit the final resume. But every agent
receives the same raw data — the full resume, the full job description, the
full alignment strategy. That's noise. Most of it is irrelevant to any one
agent's task.

If you send everything to every agent:

- Token costs explode (each agent re-reads fields it doesn't use)
- The LLM gets distracted by irrelevant data
- Changes to one data model cascade into every agent's prompt
- Debugging "why did the agent write that?" means tracing through 10KB of
  context to find the one field it actually read

The solution: **one formatter per agent, purpose-built for what that agent
needs.**

---

## 2. The Core Idea: Filter, Then Render

Every formatter does two things, always in this order:

```text
                    ┌─────────────────────────┐
                    │   FORMATTER             │
                    │                         │
    typed models ──►│  FILTER                 │
    (Resume,       │  What does THIS agent    │
     JobDescription,│  actually need? Keep     │
     Strategy, etc.)│  only those fields.      │
                    │                         │
                    │        │                │
                    │        ▼                │
                    │  ASSEMBLE               │
                    │  Combine filtered       │
                    │  slices into one dict.  │
                    │                         │
                    │        │                │
                    │        ▼                │
                    │  RENDER                 │
                    │  Turn the dict into     │
                    │  a string (TOON or      │
                    │  Markdown).             │
                    │                         │
                    └─────────┬───────────────┘
                              │
                              ▼
                    compact context string
                    (what the LLM reads)
```

**Filter** is agent-specific. Each formatter knows what its agent cares about
and drops everything else.

**Render** is shared. All formatters use the same rendering engine
(`llm_context_rendering.py`), so output formatting never drifts between
formatters.

---

## 3. The Pattern Every Formatter Follows

Open any formatter file. You will see exactly this structure:

```text
┌─────────────────────────────────────────────────────────┐
│  MODULE DOCSTRING                                        │
│  Answers four questions:                                 │
│    1. Which orchestration node calls this formatter?     │
│    2. Which agent task consumes the output?              │
│    3. What fields are KEPT, and why?                     │
│    4. What fields are DROPPED, and why?                  │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  select_resume_context(resume) → dict                    │
│    Filter: Resume → only the fields this agent needs     │
│                                                          │
│  select_job_context(job) → dict                          │
│    Filter: JobDescription → only the fields this agent   │
│    needs                                                 │
│                                                          │
│  select_strategy_context(strategy) → dict                │
│    Filter: AlignmentStrategy → only the fields this      │
│    agent needs                                           │
│                                                          │
│  (may have additional select_* functions if this agent   │
│   receives extra inputs beyond the standard three)       │
│                                                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  build_*_payload(...) → dict                             │
│    Assemble: combine all select_* results into one dict  │
│    Each slice gets a named key (e.g. "candidate_profile",│
│    "target_job") so the LLM sees structured sections.    │
│                                                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  format_*_context(...) → str                             │
│    Entry point. Called by the orchestration node.        │
│    Calls build_*_payload(), then render_context_data().  │
│    This is the ONLY function exported from the module.   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Why this specific structure

The three layers exist for a reason:

| Layer | Purpose | What it enables |
|---|---|---|
| `select_*` | Filter one data source | **Testable in isolation.** You can write a unit test that passes a `Resume` and asserts exactly which fields survive. |
| `build_*` | Combine filtered slices | **One place to name the sections.** If you rename `"candidate_profile"` to `"applicant"`, you change one line. |
| `format_*` | Render to string | **One entry point per agent.** The orchestration node imports one function, calls it with typed arguments, gets a string back. |

---

## 4. How the Layers Connect (with a real example)

Here's `gap_analysis_formatter.py` traced through all three layers:

```text
    orchestration/nodes/strategy.py
    │
    │  "I have a resume, a job description, and a match report.
    │   Build me a context string for the gap analysis agent."
    │
    ▼
    format_gap_analysis_context(resume, job, match_report)
    │
    │  LAYER 1: FILTER — three independent select_* functions
    │
    ├─► select_resume_context(resume)
    │       Resume (20+ fields) ──► {"professional_summary": "...",
    │                                  "skills": [...],
    │                                  "work_experience": [...],
    │                                  "education": [...],
    │                                  "certifications": [...],
    │                                  "languages": [...]}
    │       ✗ DROPPED: full_name, email, phone, location, website
    │         (the strategist doesn't need contact info)
    │
    ├─► select_job_context(job)
    │       JobDescription (10+ fields) ──► {"job_title": "...",
    │                                         "company_name": "...",
    │                                         "job_level": "...",
    │                                         "summary": "...",
    │                                         "requirements": [...],
    │                                         "ats_keywords": [...]}
    │       ✗ DROPPED: full_text (raw job posting — too noisy)
    │
    ├─► select_match_report_context(match_report)
    │       ReviewResult ──► {"score": 0.78,
    │                          "summary": "...",
    │                          "findings": [...]}
    │
    │  LAYER 2: ASSEMBLE
    │
    └─► build_gap_analysis_payload(…)
            returns {
                "candidate_profile": {…},     ← from select_resume_context
                "target_job": {…},            ← from select_job_context
                "current_match_report": {…}   ← from select_match_report_context
            }
            │
            │  LAYER 3: RENDER
            │
            ▼
         render_context_data(payload, format_type="toon")
            │
            ▼
         "candidate_profile:\n  professional_summary: ...\n  skills:\n    - ..."
```

The orchestration node never sees the filtering logic. It calls one function,
gets one string.

---

## 5. The Shared Rendering Engine

All formatters render through `llm_context_rendering.py`. It supports two
output formats:

```text
                    render_context_data(payload, format_type)
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
      format_type="toon"              format_type="markdown"
      (compact, for LLM context)      (readable, for human review/debug)
              │                               │
              ▼                               ▼
        render_toon(data)              render_markdown(data, description)
```

### TOON format (default)

TOON is a compact key-value serialization format designed for **LLM context
windows**. It's deliberately minimal:

```text
# Strings with spaces or special characters are quoted
candidate_name: "Jane Doe"

# Simple strings without spaces stay bare
job_level: senior

# Lists use indented dashes
skills:
  - Python
  - AWS
  - REST APIs

# Nested dicts are indented under their key
target_job:
  job_title: "Senior Backend Engineer"
  requirements:
    - requirement: Python
      importance: must_have
```

Why TOON instead of JSON or YAML?
- JSON wastes tokens on brackets, quotes, and commas — every character counts
  in a context window
- YAML has semantic whitespace rules that LLMs sometimes misinterpret
- TOON uses minimal syntax: `key: value` with indentation for nesting

### Markdown format (debug/review)

Used when a human needs to read the context (development, debugging, manual
review). Adds headings, bold labels, and code formatting for readability.

---

## 6. File Map

| File | What it owns |
|---|---|
| `__init__.py` | Public surface: 6 `format_*_context` functions. Nothing else. |
| `llm_context_rendering.py` | Shared TOON and Markdown rendering. One function for each format, plus a dispatcher (`render_context_data`). |
| `ats_optimization_formatter.py` | Context for the ATS assembly agent: optimized sections, preserved original fields, job validation signals |
| `gap_analysis_formatter.py` | Context for the gap analysis strategist: candidate profile, target job, code-computed match report |
| `experience_optimizer_formatter.py` | Context for the experience optimizer: role-scoped bullets, prioritized JD requirements, rewrite guidance |
| `professional_summary_formatter.py` | Context for the summary writer: every experience highlight and achievement (uncapped), education, job requirements |
| `skills_optimizer_formatter.py` | Context for the skills optimizer: current skills, compact role evidence, job requirements. Also has a second entry point for the rewrite (correction) pass. |
| `quality_feedback_formatter.py` | Context for the quality reviewer: original resume, tailored resume, target job |

---

## 7. Formatter → Orchestration Node Map

Every formatter has exactly one caller. This table shows the wiring:

| Formatter entry point | Called by | Agent task that reads the output |
|---|---|---|
| `format_ats_optimization_context` | `orchestration/nodes/assembly.py` | `optimize_ats_resume_task` |
| `format_gap_analysis_context` | `orchestration/nodes/strategy.py` | `create_alignment_strategy_task` |
| `format_experience_optimizer_context` | `orchestration/nodes/experience.py` | `optimize_experience_section_task` |
| `format_professional_summary_context` | `orchestration/nodes/summary.py` | `write_professional_summary_task` |
| `format_skills_optimizer_context` | `orchestration/nodes/skills.py` | `optimize_skills_section_task` |
| `format_skills_rewrite_context` | `orchestration/nodes/skills.py` | `optimize_skills_section_task` (correction pass) |
| `format_quality_feedback_context` | `orchestration/nodes/resume_quality.py` | `write_quality_feedback_task` |

Notice: `skills_optimizer_formatter.py` has **two** entry points — one for the
initial optimization pass and one for the correction (rewrite) pass. The
rewrite pass deliberately receives a minimal payload (just the current skill
list and the names to remove) because evidence judgement is already done by
the audit layer.

---

## 8. How to Add a New Formatter

If a new agent is added to the pipeline, follow these steps:

### Step 1: Create the formatter file

Create `<agent_name>_formatter.py` following the 3-layer pattern:

```python
"""Build context for the [AGENT NAME].

Caller:  src/orchestration/nodes/<node>.py
Consumer: the <task_name>

This formatter keeps:
- [list kept fields with brief why]

This formatter drops:
- [list dropped fields with brief why]
"""

from typing import Any
from src.data_models.job import JobDescription
from src.data_models.resume import Resume
from src.formatters.llm_context_rendering import OutputFormat, render_context_data


def select_resume_context(resume: Resume) -> dict[str, Any]:
    """Keep only the resume fields [AGENT] needs."""
    return { ... }


def select_job_context(job_description: JobDescription) -> dict[str, Any]:
    """Keep only the job fields [AGENT] needs."""
    return { ... }


def build_<agent>_payload(resume: Resume, job_description: JobDescription) -> dict[str, Any]:
    """Build the filtered payload for [AGENT]."""
    return {
        "resume_slice": select_resume_context(resume),
        "job_slice": select_job_context(job_description),
    }


def format_<agent>_context(
    resume: Resume,
    job_description: JobDescription,
    format_type: OutputFormat = "toon",
) -> str:
    """Return the [AGENT]'s context string."""
    payload = build_<agent>_payload(resume, job_description)
    return render_context_data(payload, format_type=format_type, description="[Agent] Context")
```

### Step 2: Register in `__init__.py`

```python
from src.formatters.<agent>_formatter import format_<agent>_context
# Add to __all__
```

### Step 3: Wire in the orchestration node

```python
from src.formatters import format_<agent>_context

context = format_<agent>_context(
    resume=state["resume"],
    job_description=state["job_description"],
    format_type="toon",
)
```

### Step 4: Add a test

Create `tests/unit/formatters/test_<agent>_formatter.py`. Test two things:
- The payload contains the expected fields
- The payload does NOT contain the dropped fields

### Step 5: Add fixtures if needed

If the new formatter needs domain objects not already in
`tests/unit/formatters/conftest.py`, add a fixture for it there.

---

## 9. Reading Order for a New Developer

If you're encountering this module for the first time, read in this order:

1. **This README** — understand the pattern and the "why"
2. **`__init__.py`** — see the public surface (6 functions, nothing else)
3. **`gap_analysis_formatter.py`** — the simplest formatter; read it top to
   bottom. It's 150 lines with clear docstrings at every layer.
4. **`llm_context_rendering.py`** — understand how dicts become strings
5. **`professional_summary_formatter.py`** — a more opinionated formatter with
   documented design decisions (why ranking was removed, why the candidate's
   own summary text is excluded)
6. **Any other formatter** — they all follow the same pattern; pick the one
   relevant to your task

---

## 10. Design Decisions Worth Knowing

### Why filter THEN render instead of one step?

Separation of concerns. The rendering engine (`llm_context_rendering.py`) knows
nothing about Resume, JobDescription, or AlignmentStrategy. It only knows how
to turn a Python dict into TOON or Markdown text.

This means:
- You can change the output format (e.g., add a "compact_json" format) without
  touching any formatter
- You can change what a formatter filters without touching rendering logic
- The rendering engine is trivially testable with plain dicts — no domain model
  fixtures needed

### Why does every formatter have a `select_resume_context` even when it looks similar?

Because "keep only the fields this agent needs" means different things to
different agents:

| Agent | Needs from Resume |
|---|---|
| Gap analysis | skills, experience, education, certifications (everything for comparison) |
| Summary writer | experience highlights + education (narrative material) |
| Skills optimizer | skills + compact role evidence (skills are the product) |
| ATS assembler | contact info, education, certifications, languages (preserve unchanged) |
| Quality reviewer | the entire original resume (truthfulness audit) |

If these shared one `select_resume_context`, it would need a parameter like
`include_contact_info=True` — and that parameter becomes a source of bugs.
Separate functions are explicit and each is only 10-30 lines.

### Why does `professional_summary_formatter.py` deliberately NOT cap or rank?

The module docstring explains this in detail, but the short version: earlier
versions of the formatter ranked roles by relevance and capped achievements at
3 per role. Live tests showed this **silently hid the candidate's most
JD-relevant evidence** in real runs. The summary writer's own `evidence_used`
step already picks what to use — and it does a better job when it can see
everything. Removing a candidate's own facts to "reduce noise" was the bug,
not the fix.

### Why is TOON the default, not JSON?

Token efficiency. An LLM context window charges per token, and TOON uses fewer
characters than JSON for the same structured data. Over hundreds of agent
calls, the difference is measurable.

---

## 11. Best Practices in This Module

- **One formatter per agent.** If an agent's context needs change, you change
  one file. No other agent is affected.

- **Document what you drop AND why.** Every "this formatter drops" list has a
  reason. The next developer shouldn't have to guess whether a field was
  excluded intentionally or accidentally.

- **Use typed arguments.** Every `select_*` function takes a Pydantic model,
  not a dict. The IDE autocompletes field names. The type checker catches
  typos.

- **Test presence AND absence.** Every formatter test asserts that expected
  fields ARE in the output AND that dropped fields are NOT. Both are equally
  important.

- **Keep the rendering engine format-agnostic.** `render_context_data` takes a
  `format_type` parameter. Adding a new format means adding one branch in the
  dispatcher, not touching 6 formatters.

- **Never build context strings inline in orchestration nodes.**
  Orchestration nodes call `format_*_context()`. They never import
  `render_context_data` directly. The formatter is the single source of truth
  for what an agent sees.

---

## 12. Short Example: Calling a Formatter

```python
from src.formatters import format_gap_analysis_context

# The orchestration node has these three objects from earlier pipeline stages:
resume = state["resume"]                 # Resume (Pydantic model)
job = state["job_description"]           # JobDescription (Pydantic model)
match = state["match_report"]            # ReviewResult (from code-owned matching)

# Build the context string in one call:
context = format_gap_analysis_context(
    resume=resume,
    job_description=job,
    match_report=match,
    format_type="toon",
)

# context is now a string like:
# "candidate_profile:\n  skills:\n    - Python\n  ..."

# Pass it to the agent task. The agent reads this string as its context.
```
