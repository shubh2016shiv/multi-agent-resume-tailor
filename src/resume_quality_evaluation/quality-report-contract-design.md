# Design: Quality Report Findings Contract & Presentation Layer

> **Scope:** `src/resume_quality_evaluation/`, `src/data_models/evaluation.py`,
> `src/orchestration/nodes/resume_quality.py`, `src/formatters/`
> **Companion to:** [resume-quality-evaluation-strategy.md](./resume-quality-evaluation-strategy.md)

This document has two halves, kept clearly apart:

- **Part A — What exists today** (§1): the real input → output flow of the evaluator,
  and why its current output is hard to consume. Describes code as it is now.
- **Part B — The proposal** (§2–§8): a typed findings contract, a presentation layer,
  and what a build would touch. **Not implemented yet — design only.**

---

# Part A — What exists today

## 1. The evaluation flow, stage by stage

### 1.1 Entry point and inputs

The `evaluate_resume_quality` node (`orchestration/nodes/resume_quality.py`) reads
three things from pipeline state:

| Input | Source | Role |
|---|---|---|
| `state["resume"]` | original parsed `Resume` | source-of-truth |
| `state["optimized_resume"].final_resume` | revised `Resume` | the thing being graded |
| `state["job_description"]` | `JobDescription` (requirements + ats_keywords) | target |

### 1.2 The node runs two independent paths

**Path A — advisory prose** (`_request_quality_feedback`):

```
{original, tailored, job}  →  TOON context  →  LLM agent  →  QualityFeedback(assessment_summary, feedback_for_improvement)
```

Best-effort; exceptions are swallowed. Pure prose, no numbers, cannot affect the gate.

**Path B — code-owned grounding** (`_ground_quality_dimensions`): three evaluators,
each with distinct inputs.

| Evaluator | Input | Internal work | Output object |
|---|---|---|---|
| `evaluate_resume_truthfulness` | original, revised, judge | render both → `detect_claim_inflation` (numbers) + per-skill `is_term_evidenced` | `TruthfulnessEvaluation(accuracy_score, exaggerated_claims[], unsupported_skills[], justification)` |
| `evaluate_job_alignment` | revised, job, judge | render revised → term-group extraction per requirement → deterministic + judge | `JobAlignmentEvaluation(relevance_score, must_have_skills_coverage, ats_keyword_coverage, is_conclusive, missed_requirements[], justification)` |
| `evaluate_rendered_structure` | revised | `build_resume_tex` → project headers → audit | `RenderedStructureEvaluation(status, violations[], ats_score, detail)` |

These are blended: `calculate_overall_quality_score(40/35/25)` → assembled into
`ResumeQualityReport` → gate applied → hard-blocks applied.

### 1.3 Outputs

- **Node output** (partial state): `quality_report`, `rendered_structure_evaluation`,
  `human_review_required`.
- **Final output:** the runner wraps `quality_report` into `OrchestrationResult` and
  persists the whole thing as JSON.

## 2. Why the current output is hard to consume

The evaluator produces a number plus scattered strings. As a standalone artifact a
score is meaningless — it does not say *what* is wrong, *where*, *how serious*, or
*what to do*. Four concrete defects in today's `ResumeQualityReport`:

1. **Structured findings are degraded to bare strings.** `detect_claim_inflation`
   returns rich `ReviewComment` objects (severity, confidence, location, advice,
   quoted_text, proposed_rewrite). The evaluator keeps only `.message`:

   ```python
   # truthfulness.py — discards six fields per finding
   exaggerated_claims = [c.message for c in detect_claim_inflation(original, revised).comments]
   ```

2. **Type tags smuggled into string prefixes.** `missed_requirements` mixes two
   different concepts distinguished only by a magic prefix:

   ```
   "Terraform"                                        # a real, confident miss
   "Needs semantic judgment: credit risk governance"  # an abstention — different meaning
   ```

3. **Findings scattered across four sub-objects** (`accuracy.exaggerated_claims`,
   `accuracy.unsupported_skills`, `relevance.missed_requirements`,
   `ats_optimization.formatting_issues`) with no unified, dimension-tagged list.

4. **Disposition computed then dropped.** `human_review_required` is computed in the
   QA node but never reaches `OrchestrationResult` (see the TODO in `runner.py`), so
   the final contract cannot distinguish *rendered* / *rejected on score* /
   *needs a human*.

---

# Part B — The proposal (not implemented)

## 3. The findings contract

Reuse the existing finding vocabulary (`Severity`, `Location` from
`src/tools/contracts/review.py`) rather than inventing a parallel one.

```python
class QualityDimension(str, Enum):
    TRUTHFULNESS = "truthfulness"
    JOB_ALIGNMENT = "job_alignment"
    RENDERED_STRUCTURE = "rendered_structure"

class FindingKind(str, Enum):
    CONFIRMED_ISSUE  = "confirmed_issue"    # real, code- or judge-confirmed problem
    NEEDS_JUDGMENT   = "needs_judgment"     # abstention; replaces the string prefix
    STRUCTURAL_BLOCK = "structural_block"   # missing essential header / unrenderable

class QualityFinding(BaseModel):
    dimension: QualityDimension
    kind: FindingKind
    message: str
    severity: Severity            # reuse review.py
    location: Location | None     # reuse review.py — section + bullet_index for UI anchoring
    advice: str | None

class Disposition(str, Enum):
    RELEASE            = "release"
    REJECTED_ON_SCORE  = "rejected_on_score"
    NEEDS_HUMAN_REVIEW = "needs_human_review"
```

### Mapping (today's strings → findings)

| Source today | Becomes |
|---|---|
| `truthfulness.exaggerated_claims` | `QualityFinding(TRUTHFULNESS, CONFIRMED_ISSUE, …)` — severity/location/advice carried from the original `ReviewComment` instead of dropped |
| `truthfulness.unsupported_skills` | `QualityFinding(TRUTHFULNESS, CONFIRMED_ISSUE, severity=MAJOR, location=SKILLS)` |
| `relevance.missed_requirements` — real miss | `QualityFinding(JOB_ALIGNMENT, CONFIRMED_ISSUE, severity from importance: must=BLOCKER/MAJOR, should=MINOR, nice=SUGGESTION)` |
| `relevance.missed_requirements` — `"Needs semantic judgment: …"` | `QualityFinding(JOB_ALIGNMENT, NEEDS_JUDGMENT, …)` — prefix becomes a real field |
| `rendered_structure.violations` | `QualityFinding(RENDERED_STRUCTURE, STRUCTURAL_BLOCK, severity=BLOCKER)` |

Key evaluator change: **stop flattening `ReviewComment` to `.message`** — the rich
finding data already exists; pass it through. `_RequirementScore` already carries
`importance`, so severity-from-importance needs no new data, only retention.

### Disposition derivation

Names the logic already spread across `resume_quality.py` (lines 62, 165, 172):

```
NEEDS_HUMAN_REVIEW  if rendered_structure == INCONCLUSIVE OR not relevance.is_conclusive
RELEASE             elif passes_quality_gate
REJECTED_ON_SCORE   else
```

(A FAIL is recoverable and routes to `patch_ats_assembly` before this point, so it is
not a disposition value.)

### Report shape

```python
class ResumeQualityReport(BaseModel):
    overall_quality_score: float
    passes_quality_gate: bool
    disposition: Disposition          # NEW — the verdict, surfaced
    findings: list[QualityFinding]    # NEW — one typed list, sorted by severity
    assessment_summary: str           # advisory LLM prose, unchanged
    feedback_for_improvement: str | None
    accuracy: TruthfulnessEvaluation  # kept for the numeric breakdown
    relevance: JobAlignmentEvaluation
    ats_optimization: ATSMetrics
```

Open decision for implementation time: the per-dimension string lists become
redundant with `findings`. Removing them gives a single source of truth (more test
churn); keeping them is additive/back-compat (some redundancy). Recommendation:
remove.

---

## 4. Presentation: how the report reaches a consumer

The framing "evaluation outputs JSON, the next agent needs markdown" conflates three
separate concerns. Keep them separate.

### Concern 1 — Canonical artifact (typed, never raw JSON between nodes)

The evaluator returns a **typed `ResumeQualityReport`** into LangGraph state.
Downstream *nodes* read typed fields directly — no serialization, no parsing between
nodes. JSON appears in exactly one place: the runner persists the result to disk for
audit and crash-safe replay (`runner.py`). Inter-node flow stays typed.

> The evaluation does **not** "spit out JSON" for the next node. It returns an object.
> JSON is only the on-disk serialization.

### Concern 2 — Agent-facing presentation (formatter → TOON)

This is the real answer to "an agent can't use the object directly." When a downstream
**agent** must read the report, a dedicated formatter selects the fields that agent
needs and renders them — following the established pattern in `src/formatters/`
(see `quality_feedback_formatter.py`).

In this codebase the agent runtime format is **TOON**, not markdown:
`render_context_data(..., format_type="toon")`. Every existing agent context uses TOON
for token cost. `llm_context_rendering.py` states the split explicitly:

- `render_toon(...)` — "compact runtime context" (agent-facing)
- `render_markdown(...)` — "optional review/debug output" (human-facing)

The typed `findings` list makes this formatter trivial: group by dimension, sort by
severity, render TOON.

### Concern 3 — Human / UI presentation (markdown)

The same formatter layer with `format_type="markdown"`. This is where markdown belongs:
a person reading the report — not an agent.

### Two anti-patterns to avoid

- **The evaluator must not emit markdown/text.** `data_models/` and the evaluators
  produce *data*; `src/formatters/` owns presentation. Emitting formatted text from the
  evaluator puts a presentation concern in the wrong layer.
- **Do not serialize to JSON to pass between nodes.** Keep it typed in state; serialize
  only at the persistence boundary.

---

## 5. Important: there is no downstream agent today

After `evaluate_resume_quality` the only branches are `patch_ats_assembly`
(deterministic code, not an agent) and `rehydrate_pii → render / end`. **The report is
terminal output today** (→ `OrchestrationResult` → persisted JSON). No agent consumes it.

Therefore:

- **Today:** typed model → JSON persistence. No agent formatter is required.
- **When the iterative refinement loop is added** (a refinement agent that reads
  findings and rewrites — Issue 1 in the strategy review): add
  `src/formatters/quality_report_formatter.py` that renders the typed report to TOON for
  that agent. The findings contract in §3 is the prerequisite that makes it clean.

---

## 6. Consumer impact (when built)

- `data_models/evaluation.py` — new types + report fields; also migrate the deprecated
  `class Config` to `model_config = ConfigDict(...)` while there.
- `OrchestrationResult` — add `disposition` (closes the `runner.py` TODO). Persisted
  JSON gains `findings` + `disposition` automatically.
- The three evaluators + the QA node — emit findings instead of strings.
- Tests — `test_evaluation_contracts.py` assertions on `missed_requirements == [...]`
  and `exaggerated_claims == [...]` move to assert on `findings`
  (dimension + kind + severity). This is the bulk of the work.

## 7. Out of scope under this banner

- Scoring weights (40/35/25) and thresholds — separate product decision.
- The P0 review items (entity-fabrication gap via `detect_rewrite_drift`; the binary
  ATS weight). The findings contract makes these *visible* but does not fix them.
