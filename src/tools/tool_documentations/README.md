# `src/tools/` — A New Joiner's Guide

Welcome. You have been asked to work in `src/tools/`. This folder is the part of the
product that actually *inspects and improves a resume*. This guide exists so you can
understand the whole thing on your own, without needing someone to explain it over
your shoulder.

Read this page first. It gives you the mental model. Then follow the reading order at
the bottom, which walks you through the deeper documents one concept at a time.

> Assumed background: you are comfortable with Python, Pydantic models, and the basic
> idea of a CrewAI agent and an `@tool`. Everything *specific to this codebase* is
> explained from scratch — no shortcuts, no unexplained jargon.

---

## 1. The 60-second mental model

The product takes a candidate's resume (a PDF or Word file) and makes it stronger —
better bullet points, real numbers, honest claims, and, when a target job is supplied,
better alignment to that job. Finally it produces a polished PDF.

`src/tools/` is the **box of instruments** that does the *inspecting and measuring*.
Think of it like a medical lab:

- A doctor (an **AI agent**, defined in `src/agents/`) decides *what* to do and talks
  to the patient.
- The lab instruments (the **tools** in `src/tools/`) run specific, well-defined tests
  and hand back results.

An agent never eyeballs a resume and guesses "this bullet is weak." Instead it *calls a
tool*, and the tool runs a precise check and returns a structured result. This keeps the
agents honest and the logic testable.

```
   ┌──────────────────────────┐         ┌───────────────────────────────┐
   │   AI AGENTS (src/agents)  │  call   │     TOOLS (src/tools)          │
   │  decide WHAT to do,       │ ──────► │  run a specific check,         │
   │  talk to the LLM          │ ◄────── │  return a structured result    │
   └──────────────────────────┘ results └───────────────────────────────┘
```

Everything else in this guide is just zooming into that right-hand box.

---

## 2. The single most important idea: one shared result shape

Every inspection tool, no matter what it checks, returns the **same shape** of result:
a `ReviewResult` — basically "a list of findings, each one pointing at a specific spot
in the resume, with advice." A spelling-style formatting check and a deep
"is-this-claim-honest" check both come back in the exact same envelope.

Why this matters: because every tool speaks the same language, the agents and the
quality layer can treat all findings uniformly — collect them, sort them, weigh them by
how confident the tool was — without knowing or caring which tool produced them.

You cannot understand how the tools coordinate until you understand this shape, so it
gets its own document: **[concepts/01-the-review-contract.md](concepts/01-the-review-contract.md)**.
Read it second (right after this page).

---

## 3. There are two layers of tools

This is the structure you will navigate every day.

```
        ┌─────────────────────────────────────────────────────────────┐
        │  LAYER 2 — AGENT-FACING TOOLS   (agent_facing_tools.py)      │
        │  Coarse instruments an agent actually calls. Each one        │
        │  bundles several engines, merges their findings, and         │
        │  returns a human-readable string for the agent to read.      │
        │  Example: match_job_requirements(...)                        │
        └───────────────┬─────────────────────────────────────────────┘
                        │  is built out of
                        ▼
        ┌─────────────────────────────────────────────────────────────┐
        │  LAYER 1 — ENGINES   (one per file, grouped in packages)     │
        │  Small, single-purpose, independently testable units.        │
        │  Pure: structured data in, a ReviewResult out.               │
        │  Example: audit_bullet_structure(resume) -> ReviewResult     │
        └─────────────────────────────────────────────────────────────┘
```

- An **engine** answers *one* narrow question ("are the bullets a sensible length?").
  It takes typed data (a `Resume` object) and returns a `ReviewResult`. It never talks
  to a CrewAI agent directly. Engines are where the real work lives.
- An **agent-facing tool** is the thing an agent is actually handed. There are only
  **7** of them, so an agent's prompt stays simple. Each one runs a handful of engines,
  merges their `ReviewResult`s, and renders the merged result into a string (because a
  CrewAI tool must return a string to the LLM).

Both layers, and how they coordinate, are explained in
**[concepts/03-the-two-layers-and-the-two-modes.md](concepts/03-the-two-layers-and-the-two-modes.md)**.

---

## 4. Three kinds of engine: mechanical, judgment, hybrid

Whenever you read about a tool, the first thing to know is *what kind of work it does*.
There are exactly three kinds.

| Kind | What it means in plain words | Does it call the LLM? | Can it be "confidently wrong"? |
|---|---|---|---|
| **Mechanical** | Pure computation — counting, pattern-matching, math. Like a ruler: it only *measures*. | **No.** Zero LLM calls. | No. A bug is a code bug, not a hallucination. |
| **Judgment** | Asks the AI model a well-framed question, because the answer needs professional taste a rule cannot capture ("is this claim honest?"). | **Yes.** The model *is* the expert. | Yes — so it must rate its own confidence. |
| **Hybrid** | Does mechanical work *and* a judgment call, combined. | **Partly.** | Only the judgment half. |

A mechanical engine is reliable but dumb. A judgment engine is smart but fallible — so
every judgment finding carries a **confidence** level, and the system is built to trust
high-confidence findings and treat low-confidence ones as advice only. (More on this in
**[concepts/02-the-llm-gateway-and-engine-types.md](concepts/02-the-llm-gateway-and-the-engine-types.md)**.)

There are even **two different hybrid shapes** — one where the mechanical half decides
whether the LLM runs at all, and one where both always run. You'll see both; they're
explained in the per-tool docs.

---

## 5. Where the LLM is actually called (only one place)

Every single LLM call in this whole folder goes through **one** small module:
`llm_gateway/`. No engine calls OpenAI (or any model SDK) directly. This one choke
point is what makes the system traceable and lets every finding be traced back to the
exact engine that produced it.

So "which tools use the LLM?" has a precise answer: **the ones whose code imports
`request_review` or `request_structured_output` from `llm_gateway`.** Everything else is
mechanical. The gateway gets its own document:
**[concepts/02-the-llm-gateway-and-the-engine-types.md](concepts/02-the-llm-gateway-and-the-engine-types.md)**.

---

## 6. Two modes: with a job description, and without

The product runs in two modes, and some tools only exist for one of them:

- **Mode A — Resume Excellence (no job description).** Make the resume genuinely better
  in its own right: stronger bullets, real numbers, honest claims, ATS-safe formatting.
- **Mode B — Job Tailoring (a job description is supplied).** Everything in Mode A,
  **plus** measuring how well the resume answers *that specific job*.

The `job_matching/` package and the job-description extractor only run in Mode B. Almost
everything else runs in both. The full data flow for each mode is drawn out in
**[concepts/03-the-two-layers-and-the-two-modes.md](concepts/03-the-two-layers-and-the-two-modes.md)**.

---

## 7. The full catalog (every tool, what it does, who uses it)

This is your map. Each engine has its own detailed document under `engines/`; each
agent-facing tool has one under `agent-tools/`. The three foundation concepts are under
`concepts/`.

### Infrastructure (not a "tool" you call — the shared machinery)

| Module | Plain-English job |
|---|---|
| `review_contract/review_models.py` | Defines `ReviewResult` / `ReviewComment` — the one result shape everything returns. |
| `llm_gateway/` | The single place an LLM is called; stamps each finding with its engine's id. |
| `shared/resume_rendering.py` | Turns a `Resume` object into plain text, for engines that need the resume as text. |

### Layer 1 — Engines, by package

| Engine (its main function) | Kind | LLM? | Package — one-line purpose |
|---|---|---|---|
| `convert_document_to_markdown` | Mechanical | No | **document_ingestion** — turn a PDF/DOCX into clean Markdown. |
| `audit_extraction_quality` | Mechanical | No | document_ingestion — did the conversion lose text/sections? |
| `redact_pii` | Mechanical | No | document_ingestion — mask name/email/phone before any LLM sees it. |
| `extract_resume` | Judgment | **Yes** | document_ingestion — Markdown → structured `Resume`. |
| `extract_job_requirements` | Judgment | **Yes** | document_ingestion — job-post text → structured `JobDescription` (Mode B). |
| `audit_bullet_structure` | Mechanical | No | **resume_diagnostics** — bullet counts by recency, bullet length. |
| `audit_consistency` | Mechanical | No | resume_diagnostics — mixed tense, repeated opening verbs (uses spaCy). |
| `audit_quantification` | **Hybrid** | Partly | resume_diagnostics — find number-less bullets (mech), suggest a metric (LLM). |
| `audit_language_quality` | Judgment | **Yes** | resume_diagnostics — duty language vs achievement; hollow phrasing. |
| `audit_summary_quality` | **Hybrid** | Partly | resume_diagnostics — length/first-person (mech) + generic/value (LLM). |
| `analyze_keyword_coverage` | Mechanical | No | **job_matching** — which JD keywords appear, and how densely (Mode B). |
| `match_requirements` | Judgment | **Yes** | job_matching — does the resume *evidence* each job requirement? (Mode B). |
| `audit_ats_formatting` | Mechanical | No | **ats_compliance** — tables/tabs/odd characters that break ATS parsers. |
| `audit_section_headers` | Mechanical | No | ats_compliance — are the standard section names present? |
| `validate_skills_evidence` | Judgment | **Yes** | **truthfulness** — is every listed skill backed by something in the resume? |
| `detect_claim_inflation` | Mechanical | No | truthfulness — numbers/names that appeared in a rewrite but weren't in the original (uses spaCy). |
| `detect_rewrite_drift` | Judgment | **Yes** | truthfulness — did a rewrite invent, exaggerate, or drop content? |
| `build_resume_tex` / `render_resume_document` | Mechanical | No | **document_rendering** — structured `Resume` → ATS-safe PDF via LaTeX. |

### Layer 2 — Agent-facing tools

| Agent-facing tool | Kind (because of what it bundles) | Engines it coordinates | Agent that uses it |
|---|---|---|---|
| `audit_summary` | Hybrid | summary_quality | Summary Writer |
| `check_skills_evidence` | Judgment | skills_evidence | Skills Optimizer |
| `audit_truthfulness` | Hybrid | claim_inflation (mech) + rewrite_drift (LLM) | Quality Assurance |
| `match_job_requirements` | Hybrid | requirements_matcher (LLM) + keyword_coverage (mech) | Gap Analysis |
| `validate_ats_compliance` | Mechanical | ats_formatting + section_headers | ATS Optimization, QA |
| `analyze_jd_keyword_coverage` | Mechanical | keyword_coverage | ATS Optimization, QA |

Professional experience quality is code-facing now, not an agent-facing tool:
`src.tools.resume_diagnostics.audit_experience_quality_for_experiences(...)`
merges bullet structure, consistency, quantification, and language quality after
the writer LLM has returned typed `Experience` objects.

> Note: the ingestion engines (`convert`, `redact`, `extract_resume`,
> `extract_job_requirements`) and the renderer are run **directly by the orchestrator**
> as pipeline steps, not handed to a reasoning agent as `@tool`s. The reason is
> explained in the architecture doc.

---

## 8. How to read these documents (in order)

You will understand the system fastest if you read in this order. Each builds on the
one before.

1. **This page** — the mental model (done).
2. **[concepts/01-the-review-contract.md](concepts/01-the-review-contract.md)** — the one
   result shape everything returns. The keystone.
3. **[concepts/02-the-llm-gateway-and-the-engine-types.md](concepts/02-the-llm-gateway-and-the-engine-types.md)**
   — where the LLM is called, and mechanical vs judgment vs hybrid in depth.
4. **[concepts/03-the-two-layers-and-the-two-modes.md](concepts/03-the-two-layers-and-the-two-modes.md)**
   — how engines and agent-facing tools coordinate, plus a full end-to-end trace.
5. **A single engine, start to finish** — read
   **[engines/resume-diagnostics/quantification-auditor.md](engines/resume-diagnostics/quantification-auditor.md)**.
   It is a hybrid, so it shows both halves working together.
6. **A single agent-facing tool** — read
   **[agent-tools/match-job-requirements.md](agent-tools/match-job-requirements.md)** to see
   how one tool drives several engines and merges their results.
7. After that, dip into any other per-tool doc as needed — they all follow the same
   layout (Purpose → Why it exists → How it works → ASCII workflow → Inputs/Outputs →
   Who uses it → Gotchas).

---

## 9. Folder layout of these docs

```
src/tools/tool_documentations/
├── README.md                         <- you are here (the map)
├── concepts/
│   ├── 01-the-review-contract.md
│   ├── 02-the-llm-gateway-and-the-engine-types.md
│   └── 03-the-two-layers-and-the-two-modes.md
├── engines/
│   ├── document-ingestion/    (convert, extraction-quality, redact, extract-resume, extract-job-requirements)
│   ├── resume-diagnostics/    (bullet-structure, consistency, quantification, language-quality, summary-quality)
│   ├── job-matching/          (keyword-coverage, requirements-matcher)
│   ├── ats-compliance/        (ats-formatting, section-headers)
│   ├── truthfulness/          (skills-evidence, claim-inflation, rewrite-drift)
│   └── document-rendering/    (resume-renderer)
└── agent-tools/               (one file per agent-facing tool)
```
