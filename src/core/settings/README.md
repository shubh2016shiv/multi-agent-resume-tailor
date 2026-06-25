# Python Settings Subsystem

`src/core/settings` is the Python code that turns configuration files and
environment values into something the application can use safely.

This package does not hold the configuration values themselves. Those live in:

- `src/config/settings.yaml`
- `src/config/agents/`
- `src/config/tasks/`
- environment variables
- `.env`

This package owns the Python side of configuration:

- locating config files
- reading YAML
- validating settings into typed models
- deciding which source wins when the same setting exists in multiple places
- exposing one shared settings object to the rest of the codebase

## Common Entry Points

Most code should import from the package root:

```python
from src.core.settings import get_config, get_agents_config, get_tasks_config
```

These three functions do different jobs.

### `get_config()`

Returns the typed application settings object.

Use this when code needs values from `src/config/settings.yaml`, `.env`, or
environment variables.

Example:

```python
from src.core.settings import get_config

config = get_config()
model_name = config.llm.model
cache_enabled = config.feature_flags.enable_cache
```

### `get_agents_config()`

Loads and merges the YAML files in `src/config/agents/`.

Use this when code needs the CrewAI agent catalog, not the main application
settings object.

Example:

```python
from src.core.settings import get_agents_config

agents = get_agents_config()
summary_agent = agents["professional_summary"]
```

### `get_tasks_config()`

Loads and merges the YAML files in `src/config/tasks/`.

Use this when code needs the CrewAI task catalog.

Example:

```python
from src.core.settings import get_tasks_config

tasks = get_tasks_config()
gap_analysis_task = tasks["run_gap_analysis_task"]
```

## Configuration Source Precedence

When `get_config()` builds the `Settings` object, it reads values from several
places. The order matters.

Highest priority to lowest priority:

1. values passed directly in Python with `Settings(...)`
2. environment variables
3. `.env`
4. file secrets
5. `src/config/settings.yaml`
6. defaults defined in `schema.py`

Earlier sources override later ones.

### Example

Suppose `src/config/settings.yaml` contains:

```yaml
llm:
  provider: openai
  temperature: 0.3
```

And `.env` contains:

```env
APP_LLM__TEMPERATURE=0.7
```

Then:

```python
config = get_config()
```

produces:

```python
config.llm.provider == "openai"
config.llm.temperature == 0.7
```

`provider` comes from YAML because nothing overrides it.
`temperature` comes from `.env` because `.env` has higher priority than YAML.

## Design Pattern Used In This Package

This package uses a common application-configuration pattern built from three
ideas.

### One typed settings object

The application reads runtime settings through one Python object:

```python
config = get_config()
```

That object is defined in `schema.py` and built in `runtime.py`.

This is useful because:

- callers get attribute access instead of raw dictionary lookups
- configuration is validated once, up front
- invalid config fails early instead of surfacing later in unrelated code

### Project defaults plus runtime overrides

`src/config/settings.yaml` provides project defaults.

`.env` and environment variables provide machine-specific or deployment-specific
overrides.

This keeps the responsibilities clean:

- YAML expresses the repo's default behavior
- runtime overrides change behavior without editing tracked files
- secrets stay outside version-controlled YAML

### One shared resolved snapshot

`get_config()` is cached with `@lru_cache`.

The first call builds the `Settings` object. Later calls reuse that same
resolved object for the life of the process.

This avoids reparsing YAML and `.env` repeatedly and keeps config access cheap.

## File Responsibilities

### `__init__.py`

Public facade for this package.

This is the import surface most of the application should use. It gathers the
main entry points and commonly used types into one place.

Open this file first if you want to see what the package exposes publicly.

### `schema.py`

Defines the typed shape of the application settings.

Open this file when you want to answer:

- what configuration exists?
- what is each field called?
- what type does it have?
- what default does it fall back to?

This file is long because it is the schema catalog for the whole application,
not because it mixes unrelated responsibilities.

### `runtime.py`

Builds the `Settings` object and defines source precedence.

Open this file when you want to understand:

- how settings are constructed
- which source wins when values conflict
- why the codebase uses `get_config()` instead of building `Settings()` everywhere

This is the center of the runtime settings mechanism.

### `yaml_source.py`

Reads YAML files and validates that they are top-level mappings.

It is intentionally small and low-level. Its job is not to understand the whole
application, only to load YAML safely and fail clearly when the shape is wrong.

It intentionally uses stdlib logging instead of the main application logger.
This file sits on the settings bootstrap path, and the application logger itself
depends on settings. Keeping this file simple avoids a circular dependency at
startup.

### `agent_task_catalog.py`

Loads and merges the YAML files in the agent and task catalog directories.

Open this file when you want to understand:

- how `src/config/agents/*.yaml` becomes one dictionary
- how `src/config/tasks/*.yaml` becomes one dictionary
- where missing-directory and malformed-YAML errors come from

### `paths.py`

Defines filesystem locations used by the settings subsystem.

Open this file when you need to know where the package expects:

- project root
- `settings.yaml`
- agent catalog directory
- task catalog directory
- tool prompt directory

### `exceptions.py`

Holds settings-specific exception types.

Open this file when you need to understand or extend the package's error surface.

## Where To Make Changes

Use this table when you know what kind of change you need.

| Change needed | File to open |
|---|---|
| Add a new typed app setting | `schema.py` |
| Change a default value used across the repo | `src/config/settings.yaml` |
| Override a value on one machine or in one deployment | `.env` or environment variables |
| Change source precedence | `runtime.py` |
| Change YAML validation behavior | `yaml_source.py` |
| Change where config files live | `paths.py` |
| Change how agent/task YAML files are merged | `agent_task_catalog.py` |

## Reading Order For A New Developer

If you are new to this package, read it in this order:

1. `__init__.py`
   to see the public entry points
2. `README.md`
   to understand the mental model
3. `schema.py`
   to see what can be configured
4. `runtime.py`
   to see how those values are constructed and overridden
5. `yaml_source.py`
   to see how raw YAML is read safely
6. `agent_task_catalog.py`
   to see how agent/task catalogs are merged
7. `paths.py`
   to see where the files actually live

## Best Practices In This Codebase

- Keep secrets out of tracked YAML files.
- Put project defaults in `src/config/settings.yaml`.
- Put machine-specific or deployment-specific overrides in `.env` or
  environment variables.
- Use `get_config()` in normal application code instead of constructing
  `Settings()` directly.
- Use the typed settings object instead of passing raw config dictionaries
  around the codebase.
- Keep runtime settings loading separate from agent/task catalog loading.
- Treat `yaml_source.py` as a bootstrap module: keep it simple and free from
  dependencies that themselves need settings.

## Short Examples

Read runtime settings:

```python
from src.core.settings import get_config

config = get_config()
provider = config.llm.provider
output_dir = config.file_paths.output_dir
```

Read merged agent YAML:

```python
from src.core.settings import get_agents_config

agents = get_agents_config()
agent_llm = agents["professional_summary"]["llm"]
```

Read merged task YAML:

```python
from src.core.settings import get_tasks_config

tasks = get_tasks_config()
task_description = tasks["run_gap_analysis_task"]["description"]
```

Construct settings directly in a test or one-off script:

```python
from src.core.settings.runtime import Settings

config = Settings(llm={"provider": "openai", "temperature": 0.1})
```

Direct construction is useful in tightly scoped tests or scripts. It is not the
normal pattern for application code, where `get_config()` should remain the
shared entry point.
