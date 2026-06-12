# Agent Configuration Files

Each YAML file in this directory configures exactly one agent. Files are named after the
agent's submodule in `src/agents/` so the pairing is findable without a lookup.

## What Each File Contains

A single top-level key — the agent's name — with the following fields:

| Field | Required | Description |
|-------|----------|-------------|
| `role` | yes | One-sentence noun phrase describing the agent's function |
| `goal` | yes | What the agent must accomplish; shapes LLM behavior |
| `backstory` | yes | Domain expertise narrative that sets the agent's tone and reasoning style |
| `llm` | yes | Model identifier in `provider/model-name` format |
| `temperature` | yes | Sampling temperature for this agent's LLM calls |
| `verbose` | no | Emit task-level logs when `true` |
| `max_tokens` | no | Per-call token ceiling; omit to use the model default |
| `max_execution_time` | no | Wall-clock limit in seconds per task attempt |

## Naming Convention

File name = submodule directory name in `src/agents/`. For example, the agent whose Python
code lives in `src/agents/job_description_analyser/` is configured in
`src/config/agents/job_description_analyser.yaml`.

The top-level YAML key (the agent's name) follows `snake_case` and describes the agent's
domain role as a noun phrase: `<domain>_<functional_role>`. Examples of the pattern:
`<domain>_analyst`, `<domain>_strategist`, `<domain>_optimizer`, `<domain>_reviewer`.

## How It Is Loaded

`src/core/settings/agent_task_catalog.get_agents_config()` globs all `*.yaml` files in this
directory, merges them into a single `dict[str, Any]`, and caches the result. Each file
contributes exactly one top-level key. Adding a new agent means adding a new file here and
registering the agent in the orchestrator; no Python loader code changes are needed.

## What Does Not Belong Here

- Task descriptions, expected outputs, or context chains: those go in `src/config/tasks/`.
- Python loading logic or Pydantic models: those live in `src/core/settings/`.
- Raw API keys or secrets: use environment variables or `.env`.
- Deeply nested structures: the loader expects a flat key-value structure one level below the
  agent name.

## Relationship to Other Config Directories

| Directory | Owns |
|-----------|------|
| `src/config/agents/` | Per-agent LLM identity (this directory) |
| `src/config/tasks/` | Per-stage task descriptions and expected outputs |
| `src/config/` root | Application settings (`settings.yaml`) |
