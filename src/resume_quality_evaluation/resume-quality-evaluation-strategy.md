# The Practical Guide to Resume Quality Evaluation
## How This Project Scores Resume Faithfulness, Job Alignment, and Release Readiness

> **Scope:** `src/resume_quality_evaluation/` · `src/tools/engines/requirement_entailment/`  
> **Audience:** Backend Engineers · GenAI Engineers · New Contributors  
> **Edition:** 2026  

---

## Table of Contents

1. [Why This Problem Is Harder Than It Looks](#1-why-this-problem-is-harder-than-it-looks)
2. [The Mental Model](#2-the-mental-model)
3. [The Package Boundary](#3-the-package-boundary)
4. [Dimension 1 — Source Faithfulness](#4-dimension-1--source-faithfulness)
5. [Dimension 2 — Job Alignment](#5-dimension-2--job-alignment)
6. [Dimension 3 — Rendered Structure](#6-dimension-3--rendered-structure)
7. [The Semantic Judge](#7-the-semantic-judge)
8. [The Release Decision](#8-the-release-decision)
9. [How To Extend The Evaluator](#9-how-to-extend-the-evaluator)
10. [Anti-Patterns Reference](#10-anti-patterns-reference)
11. [Verification Playbook](#11-verification-playbook)

---

## 1. Why This Problem Is Harder Than It Looks

Resume evaluation looks simple until you ask what "good" means. A generated resume
can be keyword-rich and still dishonest. It can be faithful to the source resume and
still miss the target job. It can match the job and still fail rendering checks. Those
are different failures, and a production evaluator must keep them separate.

The common prototype mistake is to ask an LLM for a single score:

```
"Rate this resume against this job description from 0 to 100."
```

That feels semantic, but it is not reliable. The model owns the rubric, the evidence,
the score, and the release decision all at once. It can reward generic similarity,
miss unsupported claims, or invent confidence because nothing forces it to cite
evidence.

The opposite mistake is to use only string matching. That is reproducible but shallow:
`RAG systems` should match `RAG`, `MCP` should match `Model Context Protocol`, and
`Ability to influence senior stakeholders` is not proven by the word `stakeholders`.

This package uses a cascade:

```
Deterministic checks for what code can verify cheaply.
Semantic LLM judgment only where meaning is required.
Code-owned scoring and gating at the end.
```

The governing rule is:

```
LLM judges evidence.
Code judges release.
```

---

## 2. The Mental Model

Every quality run compares four artifacts:

```
SOURCE RESUME
    The candidate's original evidence. This is the truth source.

TAILORED RESUME
    The generated resume being evaluated.

TARGET JOB DESCRIPTION
    The role requirements and ATS keywords.

RENDERED STRUCTURE
    The resume artifact shape that must survive formatting/rendering.
```

The evaluator answers three separate questions:

```
1. Source faithfulness:
   Did the generated resume stay truthful to the source resume?

2. Job alignment:
   Does the generated resume evidence the target job requirements?

3. Rendered structure:
   Is the output structurally safe enough to release?
```

Then `quality_decision.py` combines those dimensions into a product decision.

---

## 3. The Package Boundary

`src/resume_quality_evaluation/` is the score-owned evaluation package. It owns
dimension scoring and release-gate inputs. It does not own rewriting, parsing,
rendering, or raw LLM infrastructure.

```
src/resume_quality_evaluation/
├── truthfulness.py              source resume -> tailored resume faithfulness
├── job_alignment.py             tailored resume -> job requirement alignment
├── rendered_structure.py        rendered artifact structure verification
├── quality_decision.py          weighted score and pass/fail gate
├── requirement_term_groups.py   requirement text -> scorable term groups
├── requirement_terminology.py   curated deterministic terminology data
└── term_presence.py             synonym-aware deterministic term evidence
```

The LLM semantic judge lives outside this package:

```
src/tools/engines/requirement_entailment/
└── entailment_judge.py
```

That separation matters. The evaluation package can accept an injected judge, but it
does not call model providers directly. That keeps the scoring layer testable with
stub verdicts and prevents the LLM from owning numeric scores.

---

## 4. Dimension 1 — Source Faithfulness

Source faithfulness asks:

```
Is the tailored resume supported by the original resume?
```

Implemented in:

```
src/resume_quality_evaluation/truthfulness.py
```

The current signals are:

```
Mechanical claim inflation:
  detects introduced numbers/entities through `src/tools/engines/truthfulness`.

Skill support:
  checks whether revised skills are evidenced in the original resume text.
```

Skill support is deterministic first:

```
"RAG systems" -> matches "RAG"
"MCP"         -> matches "Model Context Protocol"
"K8s"         -> matches "Kubernetes"
```

If a `skill_support_judge` is injected, non-curated semantic cases can be judged by
the same entailment pattern used for job alignment. The judge returns a label; the
truthfulness evaluator applies the penalty policy.

Current policy:

```
Each unsupported skill or introduced claim costs 15 accuracy points.
Score is floored at 0.
```

This is a product heuristic, not an industry standard.

---

## 5. Dimension 2 — Job Alignment

Job alignment asks:

```
Does the tailored resume evidence the target job requirements?
```

Implemented in:

```
src/resume_quality_evaluation/job_alignment.py
```

The algorithm is intentionally layered:

```
For each structured JobRequirement:

  1. Extract requirement term groups.
  2. Split groups into deterministic literal groups and semantic/generic groups.
  3. Score deterministic groups with synonym-aware evidence.
  4. Route semantic/generic groups to the injected judge when available.
  5. Mark unresolved semantic groups as "Needs semantic judgment".
  6. Compute importance-weighted fractional coverage.
```

### Requirement Term Groups

Implemented in:

```
src/resume_quality_evaluation/requirement_term_groups.py
```

A term group is a set of terms where any member can satisfy the group.

Examples:

```
"Python"
  -> [{"Python"}]

"Entra ID / RBAC / Managed Identity"
  -> [{"Entra ID"}, {"RBAC"}, {"Managed Identity"}]

"C# / .NET"
  -> [{"C#", ".NET", ".NET Core", "dotnet", "C#/.NET"}]

"Semantic Kernel / AutoGen"
  -> [{"Semantic Kernel", "AutoGen"}]
```

Distinct bundles receive partial credit. Ecosystem or either-acceptable groups are
not over-counted.

### Deterministic Terminology

Implemented in:

```
src/resume_quality_evaluation/requirement_terminology.py
```

This file is data, not logic. It contains only small, high-confidence tables:

```
SKILL_SYNONYMS:
  surface forms that mean the same skill.

EQUIVALENCE_GROUPS:
  ecosystem or either-acceptable terms that should count as one requirement unit.

GENERIC_TERMS / GENERIC_HEAD_NOUNS:
  terms too broad to trust through literal matching.
```

Examples of good deterministic entries:

```
RAG <-> RAG systems <-> Retrieval Augmented Generation
MCP <-> Model Context Protocol
K8s <-> Kubernetes
C# / .NET as one ecosystem group
Semantic Kernel / AutoGen as one either-acceptable group
```

Examples that should not be solved by table growth:

```
leadership
ownership mindset
influence senior stakeholders
regulated clinical workflows
credit risk governance
```

Those require semantic evidence.

### Scoring

Requirement importance weights:

```
must-have    = 3
should-have  = 2
nice-to-have = 1
```

Coverage is fractional by term group:

```
covered_groups / scorable_groups
```

Then weighted across requirements:

```
relevance_score =
  100 * sum(requirement_coverage * importance_weight) / sum(importance_weight)
```

If a job has no structured requirements but has ATS keywords, ATS keyword coverage is
used as a fallback relevance signal. If it has neither requirements nor keywords, the
result is inconclusive and fails safely.

---

## 6. Dimension 3 — Rendered Structure

Rendered structure asks:

```
Can the generated resume be structurally verified for release?
```

Implemented in:

```
src/resume_quality_evaluation/rendered_structure.py
```

This is deliberately mechanical. It checks rendered document structure, especially
essential section headers. It does not claim to be a complete ATS parser.

Outcomes:

```
PASS:
  structure is verified.

FAIL:
  required rendered sections are missing.

INCONCLUSIVE:
  rendering or inspection failed, so the evaluator refuses to guess.
```

Any non-`PASS` rendered-structure status hard-blocks release.

---

## 7. The Semantic Judge

Semantic judgment is required for requirements where literal matching is not enough.

Implemented in:

```
src/tools/engines/requirement_entailment/entailment_judge.py
src/config/tool_prompts/requirement_entailment/entailment_judge.md
```

The judge answers one bounded question:

```
Does this resume text support this requirement?
```

It returns only:

```
entailed
not_supported
inconclusive
```

It does not return a numeric score. It does not decide pass/fail. It must provide an
exact supporting quote for `entailed`; code verifies that the quote exists in the
resume text. If the quote is missing or the gateway fails, the verdict becomes
`inconclusive`.

This design gives us semantic reasoning without giving the model control over the
release gate.

---

## 8. The Release Decision

Implemented in:

```
src/resume_quality_evaluation/quality_decision.py
```

The overall score uses the current product policy:

```
truthfulness     40%
job alignment    35%
rendered ATS     25%
```

This weighting is not an industry standard. It is a product decision and should only
change after a separate benchmark/review.

The quality gate fails if:

```
overall score is below threshold
rendered structure is not PASS
job alignment is inconclusive
```

Advisory LLM feedback can explain how to improve the resume, but it cannot alter
scores or gate status.

---

## 9. How To Extend The Evaluator

### Add A New Synonym

Add only high-confidence surface equivalents to:

```
src/resume_quality_evaluation/requirement_terminology.py
```

Good:

```
{"MCP", "Model Context Protocol"}
{"OpenTelemetry", "OTel"}
```

Bad:

```
{"leadership", "managed teams", "stakeholder influence"}
```

The second example is semantic behavior, not synonymy.

### Add A New Ecosystem Or Either Group

Use `EQUIVALENCE_GROUPS` when a slash/list should count as one unit:

```
{"C#", ".NET", ".NET Core"}
{"Semantic Kernel", "AutoGen"}
```

Do not collapse distinct bundles:

```
"Entra ID / RBAC / Managed Identity"
```

Those are separate skills and should receive partial credit.

### Add A New Generic Guard

Use `GENERIC_TERMS` or `GENERIC_HEAD_NOUNS` when literal matching would over-credit:

```
"security"
"ownership"
"stakeholders"
terms ending in "mindset" or "practices"
```

Generic guards do not mean the requirement is impossible to satisfy. They mean the
requirement must be judged semantically.

### Add A New Quality Dimension

Do not put it into `job_alignment.py` or `truthfulness.py` unless it belongs there.
Create a new focused evaluator or tool engine, then combine it in `quality_decision.py`
only after the scoring policy is explicitly reviewed.

---

## 10. Anti-Patterns Reference

```
ANTI-PATTERN:
  Ask the LLM to grade the whole resume 0-100.

WHY WRONG:
  The model owns the rubric, evidence, score, and release decision.

CORRECT:
  Ask for closed evidence labels, then score in code.
```

```
ANTI-PATTERN:
  Add broad thesaurus entries until tests pass.

WHY WRONG:
  The table becomes an unreviewed domain ontology and over-credits weak evidence.

CORRECT:
  Keep synonym/equivalence data small; route semantic cases to the judge.
```

```
ANTI-PATTERN:
  Score generic words literally.

WHY WRONG:
  "Security" in a resume does not prove security ownership.

CORRECT:
  Mark generic groups as needing semantic judgment.
```

```
ANTI-PATTERN:
  Call the LLM for every missed deterministic term.

WHY WRONG:
  Cost rises and the LLM becomes load-bearing for basic technical matching.

CORRECT:
  Use deterministic matching first; call the judge only for semantic/generic groups.
```

```
ANTI-PATTERN:
  Treat rendered header checks as full ATS compatibility.

WHY WRONG:
  Header inspection is useful but not a full ATS parser.

CORRECT:
  Call it rendered-structure verification and keep the claim narrow.
```

---

## 11. Verification Playbook

Run the focused evaluator suite after changes to this package:

```bash
uv run pytest tests/unit/resume_quality_evaluation tests/unit/tools/engines/test_requirement_entailment.py -x -v
```

Run orchestration-adjacent checks after changing injected judge wiring or report shape:

```bash
uv run pytest tests/unit/orchestration/nodes/test_resume_quality.py tests/unit/orchestration/nodes/test_ats_patch.py -x -v
uv run pytest tests/unit/tools/agent_tools/test_resume_review_tools.py -x -v
```

Run narrow static checks on touched paths:

```bash
uv run ruff check src/resume_quality_evaluation src/tools/engines/requirement_entailment
uv run ruff format --check src/resume_quality_evaluation src/tools/engines/requirement_entailment
uv run pyright src/resume_quality_evaluation src/tools/engines/requirement_entailment
```

If the graph has structural changes, refresh the local AST graph:

```bash
uv run graphify update . --no-cluster
```

Do not claim "works for any resume/JD/domain" from unit tests alone. That claim requires
a measured multi-domain gold set with human labels and smoke runs across real artifacts.
