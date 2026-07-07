# Evaluation
## How a Score Becomes a Decision, and Why No Number Here Comes From an LLM's Opinion of Itself

> **Scope:** `src/resume_quality_evaluation/` (the code-owned evaluators),
> `src/data_models/evaluation.py` (the report shapes), and the two places this
> subsystem's numbers meet — and fail to meet — the rest of the pipeline.
> **Audience:** Contributors tuning a quality threshold; anyone asking "where
> does this score actually come from, and can I trust it?"

---

## Table of Contents

1. [What Problem Evaluation Solves](#1-what-problem-evaluation-solves)
2. [The Core Design Principle — Evidence Engines Detect, Evaluators Score, Code Decides](#2-the-core-design-principle--evidence-engines-detect-evaluators-score-code-decides)
3. [The Three Dimensions, In Full](#3-the-three-dimensions-in-full)
4. [The Shared Similarity Primitive — One Threshold, Two Consumers](#4-the-shared-similarity-primitive--one-threshold-two-consumers)
5. [The Blend — Weights, Threshold, and Where the Number Actually Comes From](#5-the-blend--weights-threshold-and-where-the-number-actually-comes-from)
6. [Two Representations of "The ATS Grade" — Why Neither Is Redundant](#6-two-representations-of-the-ats-grade--why-neither-is-redundant)
7. [The Module Docstring That No Longer Matches the Architecture](#7-the-module-docstring-that-no-longer-matches-the-architecture)
8. [Case Study: The Root-Cause Fix Behind evaluate_rendered_structure](#8-case-study-the-root-cause-fix-behind-evaluate_rendered_structure)
9. [Where the Quality Feedback Agent Actually Fits](#9-where-the-quality-feedback-agent-actually-fits)
10. [A Config Knob That Does Nothing — a Live Discoverability Hazard](#10-a-config-knob-that-does-nothing--a-live-discoverability-hazard)
11. [Evaluation as Routing Evidence, Not Just Reporting](#11-evaluation-as-routing-evidence-not-just-reporting)
12. [Design Rule — Adding a New Evaluation Dimension](#12-design-rule--adding-a-new-evaluation-dimension)
13. [Future Considerations](#13-future-considerations)

---

## 1. What Problem Evaluation Solves

By the time a tailored resume reaches evaluation, it has already survived
every generation-time safeguard this system has — the truth floor on every
rewritten bullet, the evidence audit on every listed skill, the mechanical
checks each agent could consult mid-turn. Evaluation exists because surviving
those individual, local checks is not the same thing as being, as a whole,
truthful, relevant, and ATS-safe. A resume can pass every per-bullet truth
check and still, in aggregate, drift from the original. It can list only
evidenced skills and still miss what the job actually asked for. It can read
beautifully and still fail to parse because a section header never made it
into the rendered file. Evaluation is the one place this system steps back
and grades the *finished artifact* as a whole, along three named dimensions,
using nothing but mechanical and embedding-based evidence — never an LLM's
opinion of its own work.

The package's own module docstring states this as an explicit design law:
*"Evidence engines detect facts; these evaluators convert that evidence into
quality metrics... No LLM-owned score or gate is exposed from this
package."* Everything in this document is downstream of that one sentence.

---

## 2. The Core Design Principle — Evidence Engines Detect, Evaluators Score, Code Decides

```text
   TOOL CONTRACTS LAYER                 EVALUATION LAYER              DECISION LAYER
   (src/tools/engines/)                 (src/resume_quality_          (src/resume_quality_
                                          evaluation/)                 evaluation/quality_decision.py)
   ---------------------                ---------------------          ---------------------
   detect_claim_inflation()       -->   evaluate_resume_truthfulness()
   (mechanical, ReviewResult)            turns findings into a
                                         0-100 accuracy_score
   is_required_skill_evidenced()  -->   evaluate_job_alignment()
   (embedding similarity)                turns matches into a
                                         0-100 relevance_score
   audit_section_headers()        -->   evaluate_rendered_structure()
   (mechanical, on the REAL              turns header presence into
   rendered artifact)                    a PASS/FAIL/INCONCLUSIVE
                                         status + ats_score
                                                    |
                                                    v
                                         calculate_overall_quality_score()
                                         apply_resume_quality_gate()
                                         -- ONE deterministic blend,
                                            ONE deterministic threshold
```

This three-layer separation is the same "agents reason, code decides"
principle from [Agent Roles §2](agent-roles.md#2-the-core-design-principle--agents-reason-code-decides)
and the same "never let an agent grade its own homework" principle from
[Tool Contracts §7](tool-contracts.md#7-post-hoc-engines--never-let-an-agent-grade-its-own-homework),
applied at the scale of the *entire finished resume* rather than one bullet
or one tool call. Nothing in `src/resume_quality_evaluation/` imports
`crewai`, calls `request_review`, or reads anything an agent said about its
own output's quality — the one place an agent's language does enter this
report at all is bounded to two narrative-only fields, covered fully in
[Section 9](#9-where-the-quality-feedback-agent-actually-fits).

---

## 3. The Three Dimensions, In Full

### 3.1 Accuracy — Truthfulness Against the Original

`evaluate_resume_truthfulness` (`truthfulness.py`) combines exactly two
mechanical/embedding signals into one score:

```text
   exaggerated_claims = detect_claim_inflation(original, revised).comments
     -- the SAME mechanical, regex-based engine documented in
        Tool Contracts §9.1 -- reused here, not reimplemented

   unsupported_skills = [skill for skill in revised.skills
                          if NOT embedding-evidenced by any original skill]

   finding_count = len(exaggerated_claims) + len(unsupported_skills)
   accuracy_score = max(0.0, 100.0 - 15.0 * finding_count)
```

The scoring formula is deliberately simple and fully transparent: every
mechanically-detected fabrication costs exactly `15.0` points
(`ACCURACY_PENALTY_PER_FINDING`), floored at zero. There is no hidden
weighting between "a fabricated number" and "an unsupported skill" — the
module's own docstring is explicit that this replaces an LLM self-assessment
specifically *because* that self-assessment was "the source of the false
positives this replaces (flagging supported facts, company names, or
'Python' as exaggerations)." A model grading its own rewrite for honesty had
a documented tendency to flag things that were never wrong.

### 3.2 Relevance — Job Alignment, Importance-Weighted

`evaluate_job_alignment` (`job_alignment.py`) grades how well the tailored
resume's *skills* answer the job's *structured requirements*, weighted by
each requirement's stated importance:

```text
   IMPORTANCE_WEIGHTS = { MUST_HAVE: 3, SHOULD_HAVE: 2, NICE_TO_HAVE: 1 }

   for each requirement: matched = is_required_skill_evidenced(
                                      requirement's canonical text,
                                      resume's canonicalized skills)

   relevance_score = 100 * (sum of weights of MATCHED requirements)
                          / (sum of weights of ALL requirements)

   must_have_skills_coverage = percentage of MUST_HAVE requirements matched
```

A job with no structured requirements at all falls back to literal ATS
keyword coverage instead (`_calculate_term_coverage` — deliberately literal,
whole-token matching, not embedding similarity, because "real ATS systems
match keywords literally"). Only when a job supplies **neither** structured
requirements **nor** ATS keywords does this evaluator return
`is_conclusive=False` — an honest "there was nothing here to grade," not a
guessed zero. That `is_conclusive` flag is not cosmetic:
[Orchestration Graph §4](orchestration-graph.md#4-stage-by-stage-walkthrough)
documents that an inconclusive relevance evaluation is one of the two
conditions that hard-blocks the quality gate outright, regardless of what the
blended score says.

### 3.3 ATS — Grading the Rendered Artifact, Not the Agent's Belief About It

`evaluate_rendered_structure` (`rendered_structure.py`) is the most
consequential of the three, and it is covered in full architectural depth in
[Section 8](#8-case-study-the-root-cause-fix-behind-evaluate_rendered_structure)
because its current shape is the direct fix for a real, documented production
bug. In brief: it builds the actual `.tex` file the resume will render into,
extracts the real section-header macros from that real document, and audits
*those* — never the agent's lossy in-context view of the resume.

---

## 4. The Shared Similarity Primitive — One Threshold, Two Consumers

Both Accuracy (matching a revised skill against the original resume's skills)
and Relevance (matching a job requirement against the resume's skills) route
through exactly one shared function, `is_required_skill_evidenced`
(`skill_similarity_match.py`), so the two dimensions can never silently
disagree about what "evidenced" means:

```text
   is_required_skill_evidenced(required_term, candidate_terms)
     -> max_similarity(required_term, candidate_terms)   [the LLM gateway's
                                                            embedding primitive,
                                                            Tool Contracts §7]
     -> True if best_score >= SIMILARITY_MATCH_THRESHOLD (0.50)
```

The threshold itself carries a documented empirical basis rather than being
an arbitrary round number: a real-data smoke test found that genuine
same-concept variants score `0.54-0.84` (e.g. *"Deep Neural Networks"* vs.
*"Deep Learning"* = `0.76`), while different skills within the *same* domain
sit at a noise floor of roughly `0.27-0.45` (e.g. *"Kubernetes"* vs.
*"Machine Learning"* = `0.30`). A lower cutoff (`0.30`) was tried and rejected
because it false-matched same-domain-but-different skills. The module is
candid about the threshold's remaining limitation via a dated `TODO`: the
calibration set so far has been small, and its negative examples were mostly
*cross*-domain pairs (too easy to separate), which likely inflated confidence
in the number; a proper calibration needs a labeled set whose negative pairs
are *within*-domain, the harder and more realistic case. This is a real gate
number, tuned on real evidence, with its own honestly-documented gap.

---

## 5. The Blend — Weights, Threshold, and Where the Number Actually Comes From

`quality_decision.py` is deliberately the smallest file in this package,
because its entire job is to be the one, single, undebatable place the final
number is computed:

```text
   ACCURACY_WEIGHT  = 0.40
   RELEVANCE_WEIGHT = 0.35
   ATS_WEIGHT       = 0.25
   QUALITY_PASS_THRESHOLD = 80.0

   overall_quality_score = round(
       accuracy_score  * 0.40 +
       relevance_score * 0.35 +
       ats_score       * 0.25,
     1)

   passes_quality_gate = overall_quality_score >= 80.0
```

The module docstring states plainly that these weights "mirror the contract
documented in `src/config/tasks/quality_feedback.yaml`" and exist here as
"the single code-authoritative source so the overall score is a deterministic
blend of the dimension scores, not an LLM-narrated number." The task YAML
tells the (advisory-only) quality feedback agent what weighting to *describe*
in its narrative; this module is what actually *computes* the number that
gates the run. `should_render_resume` is intentionally trivial — it reads the
already-computed `passes_quality_gate` flag rather than recomputing anything,
which matters: the decision is made exactly once, by `apply_resume_quality_gate`,
and every later reader (including the render-gate routing function documented
in [Orchestration Graph §5](orchestration-graph.md#5-the-three-routing-decisions))
consults that one stored answer rather than re-deriving it.

---

## 6. Two Representations of "The ATS Grade" — Why Neither Is Redundant

`ResumeEnhancementPipelineState` carries the ATS verdict in **two** distinct
shapes at once, and it is worth being precise about why that is correct
design rather than duplication:

```text
   rendered_structure_evaluation: RenderedStructureEvaluation
     status: AtsCheckStatus       -- PASS / FAIL / INCONCLUSIVE (an enum)
     ats_score: float             -- 0.0 or 100.0 in practice

   quality_report.ats_optimization: ATSMetrics
     ats_score: float             -- the SAME number, folded into the blend
     keyword_coverage, formatting_issues, justification
```

The score alone cannot carry what the routing logic needs: both a `FAIL`
(an essential header genuinely missing) and an `INCONCLUSIVE` (the `.tex`
could not even be built) map to the identical `ats_score = 0.0`
(`_FAIL_ATS_SCORE` and `_INCONCLUSIVE_ATS_SCORE` are both `0.0`). If routing
decisions were made by inspecting the *score* alone, a genuine build failure
and a genuine missing-section failure would be indistinguishable — and those
two cases need to route differently (recall from
[Orchestration Graph §4](orchestration-graph.md#4-stage-by-stage-walkthrough)
that only `FAIL` routes to the deterministic patch node; `INCONCLUSIVE` has
nothing to patch and escalates straight to human review). This is a design
lesson worth stating on its own: **a blended score is for humans and for the
overall number; a status enum is for machines and for routing.** Never let a
0-100 display metric double as the thing a conditional edge branches on when
two structurally different failure modes can collapse to the same number.

---

## 7. The Module Docstring That No Longer Matches the Architecture

`src/data_models/evaluation.py`'s own module docstring is worth quoting
verbatim, and then correcting against the actual current architecture: *"The
`Quality Assurance Reviewer` agent uses these models to generate a
structured, data-driven quality report."* That sentence describes an earlier
design — one where an agent computed these scores directly. It is no longer
true. `resume_quality.py`'s `_ground_quality_dimensions`
(see [Orchestration Graph §4](orchestration-graph.md#4-stage-by-stage-walkthrough))
**discards** every one of the current Quality Feedback agent's self-assessed
dimension scores and replaces all three with the code-owned evaluators
documented in this file. The `TruthfulnessEvaluation`, `JobAlignmentEvaluation`,
and `ATSMetrics` models are populated exclusively by
`src/resume_quality_evaluation/`, never by an agent — the models themselves
still carry a docstring, and `json_schema_extra` examples, written for an
architecture this codebase has since moved away from.

This is worth surfacing explicitly rather than leaving a future reader to
discover it by tracing call sites: **trust the evaluator modules over this
particular docstring.** The models' *shape* is still accurate and load-bearing
— the description of *who fills them in* is stale.

---

## 8. Case Study: The Root-Cause Fix Behind evaluate_rendered_structure

`rendered_structure.py`'s own module docstring documents a genuine production
bug and its fix in enough detail to be worth walking through in full, because
it is the clearest example in this entire evaluation subsystem of what "grade
the real artifact, not the model's belief about it" actually means in
practice.

**The bug:** an earlier version of ATS structure checking had the QA agent
inspect its own *formatted context* — the TOON-rendered text documented in
[Memory Boundaries §4](memory-boundaries.md#4-toon--the-textual-shape-memory-takes)
— where a resume section header renders as something like `[Skills]`. The
mechanical header validator (`audit_section_headers`, from
[Tool Contracts §9.5](tool-contracts.md#95-formatting_validator--section_header_validator--two-independent-mechanical-engines-merged-at-the-tool-layer))
looks for plain standalone header lines, a shape `[Skills]` structurally
cannot match. The result was a **guaranteed false positive**: every single
run reported missing headers, regardless of the resume's actual quality,
because the check was being run against a format that could never satisfy it.

**The fix:** build the resume's real `.tex` file (`build_resume_tex`), the
actual artifact that will become the candidate's PDF. That file's real
headers are LaTeX macros — `\section*{Experience}`, `\section*{Skills}`, and
so on. `_project_section_titles` extracts each macro's title text via a
regex and joins them into plain, one-per-line text — exactly the shape
`audit_section_headers` was designed to read. The same mechanical engine,
run against the *right* artifact, now produces a meaningful verdict instead
of a guaranteed failure.

```text
   BEFORE (the bug):                       AFTER (the fix):
   TOON context: "[Skills]"           ->   real .tex: "\section*{Skills}"
   audit_section_headers CANNOT             _project_section_titles extracts:
   recognize this shape                     "Skills" (plain line)
        |                                          |
        v                                          v
   FALSE "missing header" on EVERY run      audit_section_headers reads a
                                             shape it can actually recognize
```

The scope is also deliberately narrower than the ingestion-side ATS check:
this evaluator checks section-header presence **only**, explicitly skipping
the formatting/multi-column heuristics from
[Tool Contracts §9.5](tool-contracts.md#95-formatting_validator--section_header_validator--two-independent-mechanical-engines-merged-at-the-tool-layer)
— those exist to catch problems in an *uploaded, human-formatted* resume;
running them against this system's own code-generated, provably single-column
LaTeX template (`resume.tex.j2`, built on `extarticle`) would only manufacture
noise against output that structurally cannot have the problem those
heuristics look for. Reusing a mechanical engine correctly means knowing which
part of it actually applies to the artifact in front of you.

Any failure to even *build* the `.tex` — a template error, a data problem, an
escaping bug — is caught broadly on purpose and mapped to `INCONCLUSIVE`
rather than any guessed status: *"we cannot inspect what we cannot render, so
we never silently PASS."*

---

## 9. Where the Quality Feedback Agent Actually Fits

Given Sections 7 and 8, it would be reasonable to ask whether the Quality
Feedback agent (documented in
[Agent Roles §4.8](agent-roles.md#48-quality-feedback-reviewer)) does
anything at all in this picture. It does — but its contribution is bounded to
exactly two fields, both narrative, both explicitly non-scoring:
`QualityFeedback.assessment_summary` and `.feedback_for_improvement`. Those
two strings are carried, unmodified, into the final `ResumeQualityReport`'s
own `assessment_summary` / `feedback_for_improvement` fields — human-readable
color commentary sitting alongside three dimension scores the agent had no
hand in computing. If the agent's call fails for any reason,
`resume_quality.py` substitutes a placeholder string
("Automated narrative feedback was unavailable.") and evaluation proceeds
exactly as if the agent had succeeded, because nothing about the actual
*grading* depends on this call ever happening. The agent is advisory
commentary bolted onto a report it does not control the content of.

---

## 10. A Config Knob That Does Nothing — a Live Discoverability Hazard

`src/config/settings.yaml` carries a `quality_metrics` block, under a comment
reading *"Specific thresholds for the Quality Assurance agent's evaluation...
These are expert-defined values that can be tuned,"* alongside a
`workflow.quality_threshold: 80.0` sitting right beside it:

```yaml
workflow:
  quality_threshold: 80.0
  quality_metrics:
    ats_keyword_density_threshold: 0.05
    min_professional_summary_length: 100
    max_professional_summary_length: 500
    min_bullet_points_per_experience: 3
    max_bullet_points_per_experience: 8
```

None of these five numbers is read anywhere in the active evaluation
pipeline. A search across the codebase turns up `QualityMetricsConfig` used
only by the retired `ats_optimization_agent_OBSELETE.py` and re-exported from
`src/core/settings/__init__.py` — never imported by
`src/resume_quality_evaluation/`, never imported by the summary quality
engine, never consulted anywhere the actual gate is computed. The value that
*actually* governs the gate, `QUALITY_PASS_THRESHOLD = 80.0` in
`quality_decision.py`, happens to numerically coincide with
`workflow.quality_threshold: 80.0` in the YAML — which makes the
disconnection *harder* to notice, not easier, because changing one number and
not the other would produce no error, just silent divergence. And the
professional-summary length bounds actually enforced today —
`MIN_SUMMARY_WORDS = 80` / `MAX_SUMMARY_WORDS = 110` in
[Tool Contracts §9.6](tool-contracts.md#96-summary_quality_auditor--a-hybrid-engine-and-a-documented-production-fix)
— don't even match the YAML's `100`/`500` at all, in either unit (words vs.
whatever the YAML intended) or value.

This is exactly the kind of thing worth naming plainly rather than leaving
for someone to discover the hard way: **editing `workflow.quality_threshold`
or `workflow.quality_metrics` in `settings.yaml` today has zero effect on
this pipeline's actual behavior.** The comment inviting an "expert" to tune
these values is, as things currently stand, inviting a change that will
silently do nothing.

---

## 11. Evaluation as Routing Evidence, Not Just Reporting

The `ResumeQualityReport` and `RenderedStructureEvaluation` this subsystem
produces are not read-only artifacts handed to a caller after the fact — they
are the evidence three separate routing decisions in
[Orchestration Graph](orchestration-graph.md) act on directly:

```text
   rendered_structure_evaluation.status == FAIL
     -> routes to patch_ats_assembly (deterministic recovery, Orchestration
        Graph §4)

   quality_report.passes_quality_gate == True
     -> routes to render_final_resume

   quality_report.passes_quality_gate == False
     -> routes to END, or to a DISCLOSED draft render if
        render_draft_on_gate_fail is set (Orchestration Graph §5)

   human_review_required (folded from BOTH is_ats_unverifiable and
   NOT quality_report.relevance.is_conclusive)
     -> terminal escalation, no resume path (Human In The Loop §5)
```

Evaluation, in other words, is not the last step of the pipeline reporting
what happened — it is a stage whose output *is* the control-flow signal for
every stage after it.

---

## 12. Design Rule — Adding a New Evaluation Dimension

```text
Can this dimension be graded from mechanical evidence or embedding
similarity alone, without asking a model to judge its own output?
  NO  -> it does not belong as a scored dimension in this package. Route
         it through the agent-facing tool layer (Tool Contracts) as
         advisory judgment instead, the way the Quality Feedback agent's
         narrative already is (Section 9).
  YES -> proceed.

Does grading it require inspecting the REAL final artifact (the rendered
.tex, the assembled Resume) rather than an agent's lossy in-context view?
  YES -> build the real artifact first, the way evaluate_rendered_structure
         does (Section 8). Do not grade a formatted-context proxy of the
         thing you actually care about.

Does the dimension need more than a plain 0-100 score to route on
correctly -- i.e., can two structurally different failure modes collapse
to the same score?
  YES -> carry a separate status enum alongside the score, the way
         AtsCheckStatus does (Section 6). Never let routing logic
         branch on a blended number alone.

Is the new threshold/weight a genuine product decision (not something
an agent should ever be trusted to set)?
  YES -> put it in this package as a documented module constant, with
         its empirical basis stated in a comment if one exists (see
         SIMILARITY_MATCH_THRESHOLD, Section 4). Do NOT add a config.yaml
         knob for it unless you also wire the evaluator to actually read
         that config value -- Section 10 is what happens when you don't.
```

---

## 13. Future Considerations

**Whether the vestigial `quality_metrics` / `quality_threshold` config block
should be removed or actually wired up.** Section 10 documents a real,
verified disconnect between what `settings.yaml` invites an operator to tune
and what the evaluation pipeline actually reads. Either the config should be
deleted (if the module-constant approach in `quality_decision.py` and
`resume_diagnostics/summary_quality.py` is the intended, permanent design) or
the evaluators should be updated to read from it (if per-deployment tuning is
actually wanted) — leaving both in place, silently disagreeing, is the one
option that should not persist.

**Whether `SIMILARITY_MATCH_THRESHOLD`'s calibration gap should be closed
before this threshold governs more decisions.** The threshold already gates
two of the three quality dimensions (Section 4) via one shared function. Its
own documented `TODO` — that the calibration set's negative examples were too
easy (cross-domain, not within-domain) — means the `0.50` cutoff has less
evidence behind it than its central role in the system would suggest. As more
of the pipeline comes to depend on this one number, the case for a proper
labeled gold-set calibration grows stronger.

**Whether `data_models/evaluation.py`'s docstring and examples should be
updated to reflect who actually populates these models today.** Section 7
documents a real drift between the file's own stated purpose and the current
architecture. Since these models remain entirely accurate as *data
contracts*, only the prose describing their origin needs correcting — a low-risk
change that would remove a genuine source of confusion for the next person
who reads that file first.
