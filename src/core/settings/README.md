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

> **Looking for the reusable theory, not this project's specifics?** See
> [`CONFIGURATION_PATTERNS.md`](CONFIGURATION_PATTERNS.md) next to this file. It
> documents the general enterprise configuration patterns this module is built
> on, with generic examples you can carry to any project. This README stays
> project-specific; that doc stays project-agnostic.

## How Configuration Flows (Start Here)

Every value the app reads follows the same three-stage journey. Learn it once and
you never have to scan the module to find where a setting comes from:

```
   WHERE IT'S DECLARED              HOW IT'S LOADED                  WHERE CODE READS IT
   ───────────────────             ───────────────                  ───────────────────
   settings.yaml        ┐
   environment vars     ├──►  get_config() -> Settings      ──►  config.<section>.<field>
   .env                 ┘     (merges by precedence,
                               validates, caches once)

   config/agents/*.yaml ──►   get_agents_config()  (merge)  ──►  agents["<name>"]["<field>"]
   config/tasks/*.yaml  ──►   get_tasks_config()   (merge)  ──►  tasks["<name>"]["<field>"]
```

Two rules decide everything:

1. **Which source wins** when the same value is set in more than one place:
   `environment var  >  .env  >  settings.yaml  >  code default`.
2. **How an environment variable maps to a setting** — the naming rule below, and
   the single most common thing new developers get wrong.

### The environment-variable naming rule (read this before editing `.env`)

`get_config()` does **not** read every variable in `.env`. It recognizes exactly
two shapes:

| To override… | The env var must be named… | Example |
|---|---|---|
| a nested setting | `APP_` + section + `__` + field (case-insensitive) | `APP_LLM__MODEL=gpt-4o-mini` overrides `llm.model` |
| a secret / API key | its exact aliased name, **no** `APP_` prefix | `OPENAI_API_KEY=sk-...` |

Any other variable in `.env` — e.g. `LOG_LEVEL`, `OPENAI_MODEL`, `ENVIRONMENT`,
`QUALITY_THRESHOLD` — is **ignored by `get_config()`** and does **not** override
`settings.yaml`. (Some may still be read directly elsewhere via `os.getenv`, but
they do not flow through the typed settings object.) If you "changed `.env` and
nothing happened," this rule is almost always why.

### Worked example: change the LLM model

- **Declared in:** `src/config/settings.yaml` → `llm.model` (currently `"gpt-4o"`).
  Its type/default live in `LLMConfig.model` in `schema.py`.
- **To change the project default:** edit `llm.model` in `settings.yaml`.
- **To override for one machine/run** without touching tracked files:
  `export APP_LLM__MODEL=gpt-4o-mini`.
- **Read in code via:** `get_config().llm.model`.
- **Gotcha:** an individual agent can override the model in its own
  `config/agents/*.yaml` (`llm:` key), which agent factories read through
  `get_agents_config()`. If one agent ignores your new default, check its agent YAML.

### Worked example: set or change an API key

- **These are secrets** — they live only in the environment / `.env`, never in
  `settings.yaml` (that separation is Pattern 4 in `CONFIGURATION_PATTERNS.md`).
- **Declared as fields on `Settings`** (`runtime.py`): `openai_api_key`
  (alias `OPENAI_API_KEY`), `gemini_api_key` (`GEMINI_API_KEY`), `serper_api_key`
  (`SERPER_API_KEY`), `langsmith_api_key` (`LANGSMITH_API_KEY`).
- **To set one:** add `OPENAI_API_KEY=sk-...` to `.env` — the exact name, **no**
  `APP_` prefix (aliased fields bypass the prefix; that's why they look different
  from `APP_LLM__MODEL`).
- **Read in code via:** `get_config().openai_api_key`.
- **Gotcha:** `.env` may contain other key-shaped vars (e.g. `OPENAI_MODEL`) that
  are **not** wired into `Settings`. Only the four aliased names above are read.

### Worked example: change a tool prompt / tool configuration

- **The *location* of tool prompts is configuration:** `settings.yaml` →
  `prompt_catalog.tool_prompts_dir` (default `src/config/tool_prompts`).
  Its type/default live in `PromptCatalogConfig` in `schema.py`.
- **The prompt *content*** lives in files under that directory.
- **Read in code via:** `get_config().prompt_catalog.tool_prompts_dir`, consumed
  by `src/core/prompt_catalog.py`.
- **To change where prompts load from:** edit `prompt_catalog.tool_prompts_dir`.
  **To change a prompt's text:** edit the file under `src/config/tool_prompts/`.

**The full chain, once a prompt file is loaded** (useful when a prompt change
doesn't seem to take effect):

```
settings.yaml (prompt_catalog.tool_prompts_dir)
  -> prompt_catalog.load_tool_prompt("category/name.md")   [reads + caches the file]
  -> ENGINE_RUBRIC = load_tool_prompt(...)                  [module-level constant
                                                              in the engine file,
                                                              e.g. src/tools/engines/
                                                              job_matching/requirement_matching.py]
  -> request_review() / request_structured_output()        [src/tools/llm_gateway/]
  -> messages = [{"role": "system", "content": ENGINE_RUBRIC}, ...]  [the real LLM call]
```

Two gotchas that follow directly from this chain:

- **Prompts are loaded once per process, at import time, and cached.** Every
  engine file assigns the loaded text to a module-level constant
  (`SOME_RUBRIC = load_tool_prompt(...)`), not inside the function that uses it.
  Combined with `load_tool_prompt()`'s own `@cache`, this means editing a
  `.md` prompt file **requires restarting the process** to take effect — there
  is no hot-reload.
- **Prompt text counts against the token budget.** The loaded prompt becomes
  the `system_prompt` argument to `request_structured_output()`, which checks
  `system_prompt + user_content` against `llm.structured_input_token_budget`
  before calling the provider. A much longer prompt shrinks the room left for
  the actual resume/job text being reviewed.

### Worked example: change an agent's configuration

- **Agent config is a declarative catalog, not the typed `Settings`** (Pattern 10):
  it lives in `src/config/agents/*.yaml`, one file per agent.
- **Loaded by:** `get_agents_config()`, which merges every YAML in that directory;
  **read by** agent factories through `load_agent_config("<agent_key>")`.
- **To change an agent's role / goal / backstory / llm:** edit its file in
  `src/config/agents/`. No Python change is needed — it's data, not code.

---

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
