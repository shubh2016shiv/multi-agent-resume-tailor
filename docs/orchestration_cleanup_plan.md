# src/orchestration cleanup plan

Fixes from the orchestration review, in execution order. One step = one commit.
Every step must leave the verification gate green before the next one starts.

## Baseline (2026-09-13)

| Check | Result |
|---|---|
| `ruff check src/orchestration` | clean |
| `pyright src/orchestration` | 0 errors |
| `pytest tests/unit/orchestration tests/unit/resume_quality_evaluation tests/unit/hitl` | 42 passed, **1 failed** (step 0.2) |

## Verification gate (run after every step)

```bash
uv run pytest tests/unit/orchestration tests/unit/resume_quality_evaluation tests/unit/hitl -q -s -W ignore -p no:cacheprovider
uv run ruff check src/orchestration tests/unit/orchestration
uv run ruff format --check src/orchestration tests/unit/orchestration
uv run pyright src/orchestration
```

`-s` is needed on Windows: without it, pytest's output capture crashes at exit
(colorama wrapping a closed stream) and hides the summary line.

Phases 1 and 3 each end with one real end-to-end run (costs LLM calls) using
`sample_documents/`, comparing the disposition and rendered files with a run
made before the phase.

## Progress

- [x] 0.1 Commit the in-progress work
- [x] 0.2 Fix the out-of-date quality-feedback test
- [x] 1.1 Failing tests for the ATS-patch escalation bug
- [x] 1.2 One shared release hard-block helper
- [x] 1.3 Patch node keeps an earlier human-review flag
- [x] 1.4 Put the whole escalation policy in `human_review_policy`
- [x] 1.5 Stop the feedback fallback from hiding programming errors
- [ ] 1.6 End-to-end verification run (costs LLM calls -- ask before running)
- [ ] 2.1 Correct out-of-date docstrings and comments
- [ ] 3.0 Tests that pin current behaviour before refactoring
- [ ] 3.1 `require()` helper replaces the assert blocks
- [ ] 3.2 `@pipeline_stage` decorator replaces the logging boilerplate
- [ ] 3.3 Simplify `crew_task_execution.py`
- [ ] 3.4 Simplify `nodes/experience.py`
- [ ] 3.5 Simplify `nodes/skills.py` and `nodes/summary.py`
- [ ] 3.6 Simplify `runner.py` and `state.py`
- [ ] 4.1 Measure how long agent calls wait on the kickoff lock
- [ ] 4.2 Decide whether the lock can be narrowed

---

## Phase 0: Clean starting point

### 0.1 Commit the in-progress work
`src/orchestration/nodes/experience.py`, `runner.py`, and `checkpointing.py`
have uncommitted changes. Commit them (or stash them) first, so every later
commit is a clean diff you can revert.

### 0.2 Fix the out-of-date quality-feedback test
- **Problem:** `test_feedback_request_returns_agent_feedback` builds a fake
  state with no `"run_id"`. `state["run_id"]` raises `KeyError` inside
  `_request_quality_feedback`'s `except Exception`, which returns the fallback,
  so the assertion fails.
- **Change:** add `"run_id": "test-run"` to `_state_with_required_feedback_inputs()`
  in `tests/unit/orchestration/nodes/test_resume_quality.py`.
- **Done when:** 43 passed.

---

## Phase 1: Correctness (these steps change behaviour on purpose)

### 1.1 Failing tests for the ATS-patch escalation bug
- **Bug:** the QA node sets `human_review_required=True` and blocks the gate
  when `relevance.is_conclusive` is False. If the ATS check also FAILs, the run
  goes to `patch_ats_assembly`, which (a) overwrites the flag with
  `is_ats_unrecoverable(new_outcome)` and (b) re-gates on score only. A
  successful patch therefore turns an unverifiable run into `RENDERED`.
- **Add to `tests/unit/orchestration/nodes/test_ats_patch.py`:**
  1. `_regrade_ats_dimension` with ATS PASS and a relevance where
     `is_conclusive=False` → `passes_quality_gate is False`.
  2. `patch_ats_assembly` with `human_review_required=True` in state and
     `evaluate_rendered_structure` patched to return PASS → result keeps
     `human_review_required is True`.
  3. Guard: `human_review_required=False`, conclusive relevance, PASS → stays
     False and the gate passes (behaviour that must not change).
- **Done when:** tests 1 and 2 fail, test 3 passes. Commit only together with 1.3.

### 1.2 One shared release hard-block helper
- **Change:** add `apply_release_hard_blocks(report, ats_outcome) -> ResumeQualityReport`
  to `src/resume_quality_evaluation/quality_decision.py`, beside
  `apply_resume_quality_gate`. It forces `passes_quality_gate=False` when the ATS
  status is not PASS or `report.relevance.is_conclusive` is False, and keeps the
  two `quality_hard_block_applied` log events.
- Use it in `resume_quality._ground_quality_dimensions` (replacing the two inline
  blocks) and in `ats_patch._regrade_ats_dimension` (replacing its ATS-only
  block). That also removes the duplication the patch docstring admits to.
- **Done when:** test 1.1-1 passes; the existing `test_evaluation_contracts.py`
  tests still pass.

### 1.3 Patch node keeps an earlier human-review flag
- **Change in `ats_patch.py`:**
  `"human_review_required": state["human_review_required"] or is_ats_unrecoverable(new_outcome)`.
  On the patch path a prior True can only come from inconclusive relevance
  (INCONCLUSIVE ATS routes past the patch), so keeping it is correct.
- **Done when:** all three 1.1 tests pass. Commit 1.1 + 1.2 + 1.3 as
  `fix(orchestration): keep relevance escalation through the ATS patch`.

### 1.4 Put the whole escalation policy in `human_review_policy`
- **Change:** add `is_relevance_unverifiable(relevance: JobAlignmentEvaluation) -> bool`
  and use it in `resume_quality.py:61-63` in place of the inline
  `not quality_report.relevance.is_conclusive`. Update the module docstring from
  "exactly two situations" to three.
- **Done when:** gate green; the only escalation predicates in nodes come from this module.

### 1.5 Stop the feedback fallback from hiding programming errors
- **Change in `_request_quality_feedback`:** read `run_id = state["run_id"]`
  **before** the `try`. Agent creation and the task call stay inside it, so the
  existing "agent unavailable → fallback" test still holds.
- **Add test:** state without `run_id` → raises `KeyError` (no silent fallback).
- **End of phase 1:** one end-to-end run.

---

## Phase 2: Documentation (no code behaviour)

### 2.1 Correct out-of-date docstrings and comments (single commit)
| File | Fix |
|---|---|
| `orchestration/__init__.py:10` | `nodes.py` → `nodes/` package, one module per stage |
| `graph.py:5-13` | same; add Stage 5b `patch_ats_assembly` to the stage list |
| `graph.py:151-178` | move the "Stage 3b" comment above the Stage 4 fan-in; delete the empty "Stage 6" comment block or attach it to the rehydrate edge |
| `state.py:4` | "Every field starts as None" → except `human_review_required` (False) and `clarification_answers` (empty list) |
| `nodes/summary.py:41` | `Raises: ValueError` → `PipelineQualityGateError` |
| `nodes/skills.py:122`, `nodes/experience.py:399` | remove "validated by CrewAI output_pydantic"; it's validated by `run_agent_task` |
| `nodes/ats_patch.py:44-45` | repair the broken "guarantees this)" sentence (the router in `graph.py` guarantees FAIL) |
| `nodes/ats_patch.py:151-155` | move the TODO from after `return` into the docstring |
| `human_review_policy.py` | already done in 1.4 |

---

## Phase 3: Simplification (no behaviour change)

Rule for this phase: the test suite is **not edited** except to add tests (3.0)
or update private-name imports. If an existing assertion needs changing, the
step changed behaviour: stop and rethink.

### 3.0 Tests that pin current behaviour before refactoring
Only thin coverage exists today. Add, under `tests/unit/orchestration/`:
- **`nodes/test_experience_decision.py`**: `_decide_role_rewrite_outcome`, with
  `_request_role_rewrite_proposal`, `audit_experience_rewrite_quality`, and
  `detect_claim_inflation` patched. One test per exit:
  1. first proposal clean → accepted, no repair call;
  2. first flawed, repair clean → repaired shipped, follow-up note appended;
  3. first truthful with only MINOR comments, repair introduces a truth or
     MAJOR finding → first rewrite kept with its follow-up note;
  4. both fail truth floor → source bullets preserved, fallback warning.
  Also: bullet-count mismatch and bullet-ID reorder produce truth findings.
- **`test_crew_task_execution.py`**: `_extract_json_object` (fenced JSON, prose,
  no braces); `_validate_agent_output` raises `AgentOutputError`;
  `run_agent_task` with `Crew` patched: tool agent → `response_format is None`,
  toolless non-DeepSeek → `structured_response_format(...)`.
- **`test_runner.py`**: `_settle_fresh_run_checkpoint` (paused → archived,
  otherwise unlinked), `_settle_resumed_run_checkpoint`, both PII-cleanup
  predicates (None / paused / terminal), `_result_output_dir`.
- **`nodes/test_skills.py`**: `preserve_original_skills` (re-adds dropped,
  prunes `removed_skills`), `flagged_skill_names` (HIGH only).

### 3.1 `require()` helper replaces the assert blocks
- **Why:** 40+ `assert state[...] is not None` lines; `python -O` strips them;
  `experience.py` and `graph.py` use `ValueError` for the same thing.
- **Change:** in `state.py`:
  ```python
  def require[T](value: T | None, field: str) -> T:
      """Return a state field that an upstream node must already have set."""
      if value is None:
          raise RuntimeError(f"Pipeline state '{field}' is None; its producing node has not run.")
      return value
  ```
  Nodes become `resume = require(state["resume"], "resume")`. The return type
  narrows for pyright just as `assert` does. Apply to every node, both graph
  routers, and the runner's result builders.
- **Note:** a violated invariant now raises `RuntimeError` instead of
  `AssertionError`/`ValueError`. Both are "programming error" per
  `exceptions.py`; the CLI catches neither.

### 3.2 `@pipeline_stage` decorator replaces the logging boilerplate
- **New:** `src/orchestration/nodes/_stage.py` with `pipeline_stage(name)`:
  logs `pipeline_stage_started` / `pipeline_stage_completed` with `stage`,
  `run_id`, and `duration_ms`, matching today's event names and fields.
  No completion log when the node raises (today's behaviour).
- **Must use `functools.wraps`:** LangGraph inspects node signatures.
- **Apply to:** extract_resume, analyze_job, run_gap_analysis,
  write_professional_summary, optimize_experience, optimize_skills,
  assemble_ats_resume, evaluate_resume_quality, patch_ats_assembly,
  rehydrate_pii (removes its three copies), render_final_resume.
  **Not** `await_candidate_clarifications` (it has no stage logs, and `interrupt()` raises).
- **Extra fields:** `optimize_experience` (answered/requested counts) and
  `render_final_resume` (paths, pdf_rendered) log those in their own
  `*_details` event. First grep `web_app/`, `src/observability/`, and any
  dashboards for consumers of those fields on `pipeline_stage_*`.
- **Expected saving:** ~150 lines.

### 3.3 Simplify `crew_task_execution.py`
- Build `Task(...)` once after the if/else; the branch only sets `llm.response_format`.
- Shrink the 44-line lock comment to ~5 lines (what's shared, why a lock, where
  the full reasoning lives) and move the essay to
  `src/orchestration/orchestration_architecture.md`.
- Move `AGENT_OUTPUT_ERROR_USER_ACTION` to the top of the module.
- Remove the `run_id: str = "unknown"` and `task_name="unknown"` defaults (all callers pass them).

### 3.4 Simplify `nodes/experience.py`
Depends on 3.0 tests. Sub-commits are fine.
1. Add a frozen dataclass `_ProposalCheck(role, truth_findings, repair_required, follow_up, review)`
   and `_check_proposal(proposal, source_role_resume, original_experience) -> _ProposalCheck`.
   Call it for the first and the repaired proposal; this removes the duplicated
   STEP 1 / STEP 3 blocks (~40 lines).
2. Make the empty `ReviewResult(comments=[], summary="", score=None)` one module constant.
3. Drop `RoleRewriteDecision.selected_quality_review` (never read).
4. Inline `_collect_rewrite_quality_review` and `_build_resume_with_single_experience`.
5. Reuse `_render_quality_findings_for_repair` in `_collect_rewrite_truthfulness_findings`.
6. Merge `_optimize_experience_entries` and `_run_experience_optimization_workers`.
7. Remove the `run_id="unknown"` defaults.

### 3.5 Simplify `nodes/skills.py` and `nodes/summary.py`
- Keep the STEP map in each module docstring; delete the `#####` banners inside
  functions (especially 2-3 line helpers).
- Inline `build_skills_rewrite_context` and `skills_audit_needs_rewrite`.
- **Remove the duplicated draft-selection rule:** add
  `choose_summary_draft(summary) -> SummaryDraft` to
  `src/formatters/ats_optimization_formatter.py`, have `choose_summary_text` call
  it, and delete `select_recommended_draft` from `summary.py`. The gate then
  checks the exact draft the assembler ships. Keep the name
  `enforce_summary_quality_gate` (imported by `regression_fix/smoke_test/st08_*`).

### 3.6 Simplify `runner.py` and `state.py`
- `new_pipeline_state(run_id, resume_path, jd_path) -> ResumeEnhancementPipelineState`
  in `state.py` replaces the 17-line literal.
- One `_execute_run(pipeline_input, run_id, resume_path, jd_path, checkpointer, settle, cleanup_pii, **log_fields)`
  holds the shared log / try / except / finally; `tailor_resume` and
  `resume_paused_run` keep only their different setup.
- Merge `_build_paused_orchestration_result` and
  `_build_completed_orchestration_result` into one builder with a `paused` switch.
- Inline `_build_pipeline` and `_pipeline_interrupted`; `_persist_result` returns
  `None`; `_paused_run_directory` returns `Path`.
- Optional: move `init_observability(...)` from import time into the two entry
  points, **only if** it's safe to call twice (check `src/observability/`).
- **End of phase 3:** one end-to-end run, plus one pause → answer sheet → resume
  run to exercise the HITL path.

---

## Phase 4: Performance (investigate before changing)

### 4.1 Measure how long agent calls wait on the kickoff lock
- Time how long `_KICKOFF_LOCK` takes to acquire, and add `lock_wait_ms` to
  `agent_task_completed`. Run once end-to-end and total the wait against the
  run's duration. That shows how much of the "parallel" Stage 1/3 and the
  4-worker experience pool is actually waiting in line.

### 4.2 Decide whether the lock can be narrowed
- Find where CrewAI 0.134 creates `latest_kickoff_task_outputs.db` and whether
  the storage path can be set per thread, or task-output storage turned off.
- Only if a fix exists: write a stress test (N threads calling kickoff with a
  stub LLM) that reproduces "database is locked" without the lock, then apply
  the fix and show the test passes without it.
- If no fix exists, keep the lock and record the measured cost in
  `orchestration_architecture.md`.
