"""Public facade for the Python settings subsystem.

Use this package when Python code needs typed settings:

    from src.core.settings import get_config, get_agents_config, get_tasks_config

Do not put YAML files here. Declarative configuration files live in `src/config/`.

Where to look:
- `schema.py`: every typed, settable application settings section.
- `runtime.py`: `Settings`, source precedence, `.env` behavior, `get_config()`.
- `paths.py`: project roots and YAML file locations.
- `yaml_source.py`: YAML loading and validation.
- `agent_task_catalog.py`: `agents.yaml` and `tasks.yaml` accessors.
- `exceptions.py`: package-specific settings exceptions.

Settable sections exposed by `get_config()` from `src/config/settings.yaml`:
- `application`: environment and debug mode.
- `feature_flags`: feature toggles and formatter token optimization.
- `llm`: provider, model, temperature, native agent defaults, resilience.
- `logging`: log level, format, and log file destination.
- `workflow`: iteration count, quality threshold, QA metric thresholds.
- `file_paths`: default resume and job-description input paths.
- `services`: optional Redis/database URLs.
- API keys: `OPENAI_API_KEY`, `GEMINI_API_KEY`, `SERPER_API_KEY`, `LANGSMITH_API_KEY`.
"""

from src.core.settings.agent_task_catalog import get_agents_config, get_tasks_config
from src.core.settings.exceptions import ConfigurationError
from src.core.settings.paths import (
    AGENTS_CONFIG_DIR,
    PROJECT_ROOT,
    SETTINGS_YAML_PATH,
    SRC_ROOT,
    TASKS_CONFIG_DIR,
)
from src.core.settings.runtime import Settings, get_config
from src.core.settings.schema import (
    AgentDefaults,
    ApplicationConfig,
    FeatureFlags,
    FilePathsConfig,
    LLMConfig,
    LLMGoogleConfig,
    LLMResilienceConfig,
    LoggingConfig,
    ObservabilityConfig,
    QualityMetricsConfig,
    ServicesConfig,
    WorkflowConfig,
)
from src.core.settings.yaml_source import read_yaml_mapping, yaml_config_settings_source

__all__ = [
    "AGENTS_CONFIG_DIR",
    "PROJECT_ROOT",
    "SETTINGS_YAML_PATH",
    "SRC_ROOT",
    "TASKS_CONFIG_DIR",
    "AgentDefaults",
    "ApplicationConfig",
    "ConfigurationError",
    "FeatureFlags",
    "FilePathsConfig",
    "LLMConfig",
    "LLMGoogleConfig",
    "LLMResilienceConfig",
    "LoggingConfig",
    "ObservabilityConfig",
    "QualityMetricsConfig",
    "ServicesConfig",
    "Settings",
    "WorkflowConfig",
    "get_agents_config",
    "get_config",
    "get_tasks_config",
    "read_yaml_mapping",
    "yaml_config_settings_source",
]
