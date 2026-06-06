# Professional Experience Agent Architecture

## Current Decision

The professional experience pipeline is split by responsibility:

```text
Resume.work_experience role
  -> one-role Resume copy
  -> TOON role context
  -> professional experience LLM writes OptimizedExperienceSection
  -> code runs experience quality checks on typed Experience objects
  -> code asks for one rewrite only if blocker or major findings exist
  -> code restores experience_id
  -> code merges all optimized roles
```

The writer LLM does not call audit tools. It reads compact TOON context and
returns structured Pydantic output. Code owns audit timing, rewrite decisions,
ID restoration, and merge.

## Why This Boundary Exists

TOON is a model-facing context format. It reduces prompt size and helps the LLM
read the role evidence.

JSON and Pydantic are code-facing contracts. They validate data after the LLM
returns output.

The writer LLM must not be asked to convert TOON or draft text into JSON tool
arguments. That conversion creates failed tool calls, retries, and higher cost.
The clean boundary is:

```text
LLM input: TOON
LLM output: OptimizedExperienceSection
Code audit input: list[Experience]
Code audit output: ReviewResult
```

## Runtime Flow

### Stage 3 / Step 3.1: Build the one-role writing agent

Location: `src/agents/professional_experience/agent.py`

The agent has no tools. It receives a role-scoped CrewAI task from orchestration
and sends structured output to `Task(output_pydantic=OptimizedExperienceSection)`.

### Stage 3 / Step 3.2: Build one-role resume context

Location: `src/orchestration/nodes.py`

Orchestration selects one `Experience` from `Resume.work_experience`, copies it
into a one-role `Resume`, and sends that object to
`format_experience_optimizer_context(..., format_type="toon")`.

### Stage 3 / Step 3.3: Ask the LLM to write one role

Location: `src/orchestration/nodes.py`

The LLM receives TOON context and returns one `OptimizedExperienceSection`.
CrewAI validates the output through the Pydantic contract.

### Stage 3 / Step 3.4: Check the written role in code

Location: `src/tools/resume_diagnostics/__init__.py`

The four experience checks run here, after the LLM output exists as typed
`Experience` objects:

- `audit_bullet_structure_for_experiences`
- `audit_consistency_for_experiences`
- `audit_quantification_for_experiences`
- `audit_language_quality_for_experiences`

These checks are not orphaned and are not delegated to the writer LLM. They are
composed by `audit_experience_quality_for_experiences(...)`, which returns one
`ReviewResult` for orchestration.

### Stage 3 / Step 3.5: Decide whether one rewrite is needed

Location: `src/orchestration/nodes.py`

Only `blocker` and `major` audit findings trigger one extra LLM call. `minor`
and `suggestion` findings do not trigger another rewrite.

The rewrite context contains:

- the original TOON role context
- the previous `OptimizedExperienceSection` JSON
- the audit feedback

The pipeline does not loop beyond this one rewrite attempt.

### Stage 3 / Step 3.6: Restore the code-owned experience_id

Location: `src/orchestration/nodes.py`

`experience_id` is assigned by code after resume extraction. The LLM may omit or
change it, so orchestration restores the original ID before merge.

## Contract Placement

- `src/data_models/resume.py::Experience`
  Canonical resume role model.

- `src/agents/professional_experience/models.py::OptimizedExperienceSection`
  Output contract for the professional experience writer.

- `src/tools/review_contract/review_models.py::ReviewResult`
  Audit result contract used by diagnostics and orchestration.

- `src/tools/resume_diagnostics.audit_experience_quality_for_experiences`
  Code-facing professional-experience audit helper. It accepts typed
  `list[Experience]` and returns `ReviewResult`.

## Experience Identity

Each parsed `Experience` has an optional `experience_id` assigned by code after
resume extraction:

```text
exp_{resume_order}_{company_slug}_{title_slug}_{start_date}
```

The ID correlates role-scoped parallel work back to the original experience
entry. It is code-owned metadata, not LLM-authored content.

## Context Rules

Each role-level LLM call receives only the context needed to rewrite that role:

- one role's company, title, dates, location, description, achievements, and
  skills used
- compact job signals when available
- alignment guidance when available

The role writer should not receive unrelated roles, education, certifications,
personal information, or a full resume skills inventory. Those fields increase
hallucination risk and do not belong inside one role's evidence boundary.

## Quality Control Rules

Quality is controlled in code and prompt instructions:

- Task prompt controls what the writer should produce.
- Pydantic output controls the returned structure.
- Diagnostics control whether the written role has serious quality issues.
- Orchestration controls whether one rewrite is requested.
- Orchestration restores `experience_id` and merges role outputs.

This is intentionally a flat pipeline. No factories, service layer, strategy
registry, or persistent cache is needed for the current behavior.
