# Tooling Plan — `src/tools/`

> **Status:** Design plan. No tools are implemented against this yet.
> **Scope:** Defines every tool the multi-agent pipeline needs, why each exists,
> how agents use it, and how the tools improve a resume **with or without** a
> target job description.

---

## 1. What a "tool" is in this system

A tool is **a callable an agent (or the orchestrator) invokes to perform one
specific job that the agent should not do freehand inside its own prompt.**

A tool may be:

- **Mechanical** — pure text math. Counting, grammar, format, presence-of-a-number.
  Encodes *no professional knowledge*; works identically for a nurse, a quant, or
  a welder because it only measures.
- **Judgment** — requires knowing what "good" means in the candidate's field.
  The **model is the expert**. There is no frozen list of clichés or "power verbs."
  A judgment tool is constrained by a schema, a rubric in its prompt, and a
  contract that forces it to cite the exact line it reacted to and rate its own
  confidence.
- **Hybrid** — mechanical detection plus a judgment-based suggestion
  (e.g. *detect* a bullet has no number deterministically, *suggest* what metric
  fits using the model).

### The rule that killed the "curated list" idea

We do **not** ship `cliche_phrases.json`, `strong_verbs.yaml`, or a metric
taxonomy. Those cannot scale across every domain (software, mathematics, law,
nursing, trades) and they are redundant with the model, which already holds that
cross-domain expertise. Professional knowledge lives in the **model + rubric
prompts**, never in static data files. The only thing we freeze is
domain-neutral ergonomics (e.g. a bullet over ~35 words is a paragraph).

---

## 2. Core design principles

1. **Mechanics vs judgment** — deterministic where it's pure measurement,
   model-backed where it's professional judgment. Never fake determinism with a
   hardcoded list.
2. **One return contract** — every tool returns the same shape: a list of
   `ReviewComment`s, each carrying the **quoted line**, a **severity**, a
   **confidence**, and a **suggestion**. This is what makes a model-backed tool
   trustworthy: it always shows its work and rates its certainty, so the QA layer
   can weigh findings instead of blindly trusting them.
3. **Two levels** — *engines* (small, pure, independently testable units) sit
   beneath *agent-facing tools* (coarse expert instruments an agent actually
   calls). An agent receives ~10 tools, not ~17 micro-functions, so orchestration
   never leaks into the agent's prompt.
4. **Two modes** — the system improves a resume **with or without** a job
   description. The JD is an *enrichment that sharpens* the review, never a
   precondition (see §7).
5. **No over-scaffolding** — a shared helper only earns its own file once a
   second tool needs it. We do not pre-build a framework ahead of need.

---

## 3. The universal return contract

Defined once in `foundation/`, used by every tool.

```
ReviewComment
  message       : str    # what was noticed, in plain language
  quoted_text   : str    # the exact bullet/line from the resume it points at
  severity      : enum   # blocker | major | minor | suggestion
  confidence    : enum   # high | medium | low   (judgment tools must set this)
  suggestion    : str    # concrete recommended fix

ReviewResult
  comments      : list[ReviewComment]
  summary       : str    # optional one-line verdict
  score         : float  # optional, only where a numeric gate is meaningful
```

- Mechanical tools always return `confidence = high` (measurement is certain).
- Judgment tools must populate `confidence` honestly; low-confidence comments are
  advisory, high-confidence comments can gate the iterative loops.

---

## 4. Directory structure

```
src/tools/
├── __init__.py                          # public surface: re-exports agent-facing tools only
│
├── foundation/                          # shared kernel (only what 2+ tools need)
│   ├── review_comment.py                # ReviewComment + ReviewResult: the universal return shape
│   └── llm_reviewer.py                  # shared harness: ask the model a rubric question,
│                                        #   get ReviewComments back with quotes + confidence
│   # resume_rules.py  and  text_helpers.py  are added ONLY when a second tool
│   # actually shares them — not pre-declared.
│
├── document_ingestion/                  # raw files → trustworthy structured input
│   ├── document_converter.py            # [mechanics] PDF/DOCX/TXT/MD → clean Markdown
│   ├── extraction_quality_auditor.py    # [mechanics] did conversion lose sections / drop text?
│   ├── resume_section_extractor.py      # [judgment]  Markdown → structured Resume
│   └── job_requirement_extractor.py     # [judgment]  Markdown → structured JobDescription
│
├── resume_diagnostics/                  # "is this resume well-built?" (domain-agnostic craft)
│   ├── bullet_structure_auditor.py      # [mechanics] bullet count by recency + word length
│   ├── consistency_auditor.py           # [mechanics] tense consistency, repeated verbs/tech
│   ├── quantification_auditor.py        # [hybrid]    detect missing numbers + suggest metric type
│   ├── language_quality_auditor.py      # [judgment]  duty-language → achievement; hollow phrasing
│   ├── action_verb_advisor.py           # [judgment]  precise, domain-true verb (not fake power words)
│   └── summary_quality_auditor.py       # [hybrid]    length/first-person (mech) + generic/value (judgment)
│
├── job_matching/                        # "how well does this resume answer THIS job?" (JD required)
│   ├── keyword_coverage_analyzer.py     # [mechanics] JD keyword presence + coverage
│   └── requirements_matcher.py          # [hybrid]    resume evidence vs job requirements (semantic)
│
├── ats_compliance/                      # "will the machine read it?"
│   ├── formatting_validator.py          # [mechanics] tables/columns/tabs/images/risky chars
│   └── section_header_validator.py      # [mechanics] ATS-readable standard section names
│
└── truthfulness/                        # trust layer: "is it still honest after rewriting?"
    ├── skills_evidence_validator.py     # [judgment]  every listed skill backed by experience
    ├── claim_inflation_detector.py      # [judgment]  rewrite stronger than the source evidence
    └── tailoring_fidelity_comparator.py # [judgment]  original vs revised: adds / losses / exaggerations
```

Each sub-package answers exactly one resume-review question. If a package can't be
described in one sentence, it shouldn't be a package.

---

## 5. The engines, package by package

### `document_ingestion/` — get trustworthy input

| Engine | Type | Why it exists | How it's used |
|---|---|---|---|
| `document_converter` | mechanics | Reliably turn any PDF/DOCX/TXT/MD into clean Markdown. | Orchestrator runs it first, before any agent. |
| `extraction_quality_auditor` | mechanics | Bad conversion (missing sections, suspiciously little text, lost layout) silently poisons everything downstream. Catch it early. | Runs right after conversion; flags low-quality extraction so the user can re-upload. |
| `resume_section_extractor` | judgment | Turn raw resume Markdown into a structured `Resume` (summary, experience, skills, education). | Backs the Resume Extractor agent. |
| `job_requirement_extractor` | judgment | Turn a JD into structured requirements: role, seniority, must-haves, keywords, responsibilities, implicit expectations. | Backs the Job Analyzer agent. **Only runs when a JD is supplied.** |

### `resume_diagnostics/` — make the resume genuinely good in its own field

| Engine | Type | What problem it fixes (real candidate pain) | How it's used |
|---|---|---|---|
| `bullet_structure_auditor` | mechanics | People put 10 bullets on a 10-year-old job and 2 on their most important recent role; bullets run too long or too short. | Reports count-by-recency and per-bullet length violations to the Experience Optimizer. |
| `consistency_auditor` | mechanics | Mixed tense ("Led … and am currently…"); every bullet opening with the same verb; the same tech repeated with no new info. | Flags tense and repetition issues for polish. |
| `quantification_auditor` | hybrid | The #1 weakness of weak resumes: no numbers. People know they "should add metrics" but not *what* to quantify. | Detects bullets with no number (mechanical), then suggests a fitting metric category — team size, time saved, scale, cost, adoption (judgment). |
| `language_quality_auditor` | judgment | Duty language ("responsible for", "worked on") instead of achievement; hollow filler. Domain-aware — what's hollow in nursing ≠ in software. | Rewrites guidance: reframe duties as achievements, strip empty phrasing. |
| `action_verb_advisor` | judgment | People can't decide between "built / developed / architected"; LLMs spit out nonsense like "spearheaded." Verb depends on the field ("proved", "litigated", "fine-tuned"). | Given what the person actually did, returns the verb that *accurately* describes their contribution. |
| `summary_quality_auditor` | hybrid | Summaries are too long, written in first person, generic ("results-oriented professional"), or just repeat the bullets. | Checks length + first-person mechanically; judges generic-ness and whether a value proposition is present. |

### `job_matching/` — answer a specific job (JD required)

| Engine | Type | Why it exists | How it's used |
|---|---|---|---|
| `keyword_coverage_analyzer` | mechanics | ATS and recruiters scan for the JD's terms. Keywords come **from the JD**, not a static list. | Reports which JD keywords are present/absent and coverage %. |
| `requirements_matcher` | hybrid | The core tailoring question: how much of this job does the resume actually evidence? Needs semantic equivalence (Flask experience partially covers a FastAPI requirement). | Produces matched / partial / gap classification feeding the Gap Analysis agent. |

### `ats_compliance/` — get past the machine

| Engine | Type | Why it exists | How it's used |
|---|---|---|---|
| `formatting_validator` | mechanics | Tables, multi-column layouts, tabs, embedded images, and risky characters break ATS parsing. | Flags structures that won't survive automated parsing. |
| `section_header_validator` | mechanics | ATS classifies content by recognizing standard section names. | Checks headers map to recognized conventions. |

### `truthfulness/` — keep it honest after any rewrite (mode-independent)

| Engine | Type | Why it exists | How it's used |
|---|---|---|---|
| `skills_evidence_validator` | judgment | Listed skills with zero supporting experience get candidates exposed in interviews (and flagged by ATS). | For each listed skill, confirm it appears in real experience; flag unsupported ones. |
| `claim_inflation_detector` | judgment | The moment an LLM rewrites a bullet it can overstate the source. This guards every rewrite — **JD or not**. | Compares each revised claim against its source evidence; flags overshoot. |
| `tailoring_fidelity_comparator` | judgment | After revision: did we invent something unsupported, drop something important, or exaggerate? | Diffs original vs revised resume for additions / losses / exaggerations. |

---

## 6. Agent-facing tools (the registered surface)

Agents receive coarse instruments, each backed by the engines above and each
returning one `ReviewResult`.

| Agent-facing tool | Type | Backed by | Used by |
|---|---|---|---|
| `convert_document` | mechanics | document_converter + extraction_quality_auditor | Orchestrator |
| `extract_resume` | judgment | resume_section_extractor | Resume Extractor |
| `extract_job_requirements` | judgment | job_requirement_extractor | Job Analyzer (JD mode) |
| `match_requirements` | hybrid | requirements_matcher + keyword_coverage_analyzer | Gap Analysis (JD mode) |
| `audit_experience_quality` | hybrid | bullet_structure + consistency + quantification + language_quality + action_verb | Experience Optimizer |
| `audit_summary_quality` | hybrid | summary_quality_auditor | Summary Writer |
| `validate_skills_evidence` | judgment | skills_evidence_validator | Skills Optimizer |
| `analyze_keyword_coverage` | mechanics | keyword_coverage_analyzer | ATS, QA (JD mode) |
| `validate_ats_compliance` | mechanics | formatting_validator + section_header_validator | ATS, QA |
| `audit_truthfulness` | judgment | claim_inflation_detector + tailoring_fidelity_comparator | QA |

The Experience Optimizer calls `audit_experience_quality` **once** and gets a
single `ReviewResult` containing comments from all five engines — each tagged with
its source, quote, and confidence. That report is what drives the iterative
refinement loop (draft → review → revise) on an objective basis.

---

## 7. Two modes — works with OR without a job description

Most people improving a resume do **not** have one specific JD in hand. They have
a domain and a career. The tool split makes both flows first-class.

**14 of 17 engines need no JD at all.** Only `job_matching/` (plus the JD
extractor) is gated on a job description. The `truthfulness/` layer is
mode-independent — it guards any rewrite, because an LLM can overshoot whether or
not a JD is present.

### Mode A — Resume Excellence (no JD)
```
document_ingestion → resume_diagnostics → ats_compliance → truthfulness
```
Outcome: a stronger, honest, professional resume in the candidate's own
field — better bullets, real quantification, precise verbs, no duty language or
hollow filler, ATS-safe, every skill backed by evidence. **No target job needed.**

### Mode B — Job Tailoring (JD provided)
```
document_ingestion → resume_diagnostics → job_matching → ats_compliance → truthfulness
```
Everything in Mode A, **plus**: `job_matching` activates, the diagnostics become
job-aware (keyword priorities, requirement-driven emphasis), and
`tailoring_fidelity_comparator` ensures the tailored version stays truthful to the
original.

| Concern | Mode A (no JD) | Mode B (with JD) |
|---|---|---|
| Resume craft (bullets, verbs, quantification, summary) | ✅ | ✅ |
| ATS formatting & headers | ✅ | ✅ |
| Truthfulness / no inflation | ✅ | ✅ |
| Skills backed by evidence | ✅ | ✅ |
| Keyword coverage vs a job | — | ✅ |
| Requirement matching / gap analysis | — | ✅ |
| Job-aware emphasis & keyword priority | — | ✅ |

The JD **sharpens** the review; it is never required for the system to deliver
value.

---

## 8. Naming conventions

- **Packages:** lowercase, full words, no leading underscore, no abbreviations
  (`document_ingestion`, not `doc_ingest`; `resume_diagnostics`, not `diag`).
- **Engine files:** `<subject>_<role>.py` where role names the act —
  `_auditor`, `_validator`, `_advisor`, `_analyzer`, `_matcher`, `_detector`,
  `_comparator`, `_extractor`. A developer reads `claim_inflation_detector.py`
  and knows the question it answers before opening it.
- **Foundation files:** named for literal content — `review_comment.py` holds the
  comment/result shapes; `llm_reviewer.py` holds the model-call harness.
- **Functions inside agent-facing tools** match their `@tool` display name in
  snake_case.

---

## 9. Deliberately NOT included (anti-overengineering)

- **No curated knowledge files** (`cliche_phrases.json`, verb taxonomies). The
  model is the cross-domain expert; freezing its knowledge breaks domain coverage.
- **No thin dead-code wrappers.** The earlier `resume_parser.py` / `job_analyzer.py`
  `@tool` wrappers were bypassed by the orchestrator and are not reproduced here.
- **No premature foundation files.** `resume_rules.py` and `text_helpers.py`
  appear only once a second tool genuinely shares them.
- **No one-CrewAI-tool-per-helper.** Engines group into ~10 agent-facing tools.

---

## 10. Suggested build order

1. **Contract first:** `foundation/review_comment.py`, then
   `foundation/llm_reviewer.py`. Nothing else can be consistent until these exist.
2. **Ingestion:** `document_converter` → `extraction_quality_auditor` →
   the two extractors. Without trustworthy input, every downstream tool is noise.
3. **Mode A core:** `resume_diagnostics/` then `ats_compliance/` then
   `truthfulness/`. This delivers a usable product **without any JD**.
4. **Mode B:** `job_matching/` and the JD extractor wiring. Additive on top of
   Mode A.
5. **Agent wiring:** register the ~10 agent-facing tools and move the four
   currently-embedded `@tool`s out of the agent files into their engines.
```
