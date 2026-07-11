"""Public facade for the Python settings subsystem.

Most callers only need one of these three entry points. Note they split along
Pattern 10 (the two kinds of config — see CONFIGURATION_PATTERNS.md):

- `get_config()` — the TYPED application settings object, from
  `src/config/settings.yaml` plus environment variables and `.env`.
- `get_agents_config()` / `get_tasks_config()` — DECLARATIVE catalogs, merged
  from the YAML directories under `src/config/agents/` and `src/config/tasks/`.

Do not put declarative YAML files in this package. This package owns the Python
code that locates, loads, validates, caches, and exposes them — that split of
data (the YAML files) from loader (this package) is Pattern 1.

Where to look when you need details:
- `schema.py`: every typed settings section and field
- `runtime.py`: source precedence, `.env` loading, and `get_config()`
- `agent_task_catalog.py`: merged CrewAI agent/task YAML accessors
- `yaml_source.py`: low-level YAML mapping loading and validation
- `paths.py`: filesystem locations for settings and catalogs
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
    PromptCatalogConfig,
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
    "PromptCatalogConfig",
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
