# Rules for src/config/ — loaded ONLY when touching config files

## Configuration Convention
- All YAML configs live here: `agents.yaml`, `tasks.yaml`, `settings.yaml`
- NO Pydantic Settings classes in this directory — those go in `src/core/config.py`
- YAML files are loaded by CrewAI's built-in config loader
- Never hardcode values in agent files — always reference YAML keys

## YAML Structure
- `agents.yaml`: Keys match agent class names (snake_case), values = role/goal/backstory/tools
- `tasks.yaml`: Keys match task names, values = description/expected_output/agent association
- `settings.yaml`: Environment-specific overrides, LLM model names, API endpoints

## What NOT to do
- Do NOT put raw secrets or API keys in these YAML files
- Do NOT nest deeply — CrewAI expects a flat key-value structure
