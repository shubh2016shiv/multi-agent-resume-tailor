# Python Settings Subsystem

This package contains the Python code that loads, validates, caches, and exposes
configuration files to the application.

## What Belongs Here

- `schema.py`: every typed setting available through `get_config()`.
- `runtime.py`: Pydantic `Settings` aggregate, source precedence, `.env` behavior, cache.
- `yaml_source.py`: strict YAML mapping loading and validation.
- `paths.py`: project root and config file locations.
- `agent_task_catalog.py`: accessors for `agents.yaml` and `tasks.yaml`.
- `exceptions.py`: package-specific settings exceptions.
- `__init__.py`: public facade and import map.

## What Does Not Belong Here

- YAML configuration files.
- Agent/task prompt content.
- Secrets or API key values.

Those belong in `src/config/`, `.env`, or the deployment environment.

## How To Use

```python
from src.core.settings import get_config

config = get_config()
model_name = config.llm.model
```

To discover what can be configured, open `schema.py`. To change values,
edit `src/config/settings.yaml` or environment variables.
