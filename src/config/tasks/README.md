# Task Configuration Files

Each YAML file in this directory configures the task or tasks that belong to one pipeline
stage. Files mirror the agent config filenames in `src/config/agents/` so agent and task
configs for the same stage are findable side by side.

## What Each File Contains

One or more top-level keys, each being a task name. Every task key accepts the following
fields:

| Field | Required | Description |
|-------|----------|-------------|
| `description` | yes | Step-by-step instructions the assigned agent follows; shapes LLM reasoning |
| `expected_output` | yes | Success criteria; the agent works until its output matches this |
| `agent` | yes | Name of the agent responsible; must match a key in `src/config/agents/` |
| `context` | no | List of task names whose outputs are injected as context for this task |
| `output_file` | no | Relative path where the task output is written to disk |

## Naming Convention

File name = agent submodule name = agent config filename. For example, the tasks assigned to
the agent configured in `src/config/agents/gap_analysis.yaml` live in
`src/config/tasks/gap_analysis.yaml`.

Task keys follow `snake_case` and use a `verb_noun_task` pattern that describes the specific
operation: `<verb>_<noun>_task`. The verb describes the action (`analyze`, `extract`,
`optimize`, `assess`); the noun names the artifact being acted on.

A single file may contain more than one task when a stage has both a legacy task key (used by
the crew-based pipeline) and a current task key (used by the LangGraph pipeline). Both keys
serve the same agent; they are co-located so both are visible when editing that stage.

## How It Is Loaded

`src/core/settings/agent_task_catalog.get_tasks_config()` globs all `*.yaml` files in this
directory, merges them into a single `dict[str, Any]`, and caches the result. Adding a new
task means adding or editing the file for the relevant stage; no Python loader code changes
are needed.

## What Does Not Belong Here

- Agent identity fields (role, goal, backstory, llm): those go in `src/config/agents/`.
- Python loading logic or Pydantic models: those live in `src/core/settings/`.
- Business logic or LLM calls: those belong in `src/agents/<submodule>/agent.py`.

## Relationship to Other Config Directories

| Directory | Owns |
|-----------|------|
| `src/config/tasks/` | Per-stage task instructions and output contracts (this directory) |
| `src/config/agents/` | Per-agent LLM identity |
| `src/config/` root | Application settings (`settings.yaml`) |
