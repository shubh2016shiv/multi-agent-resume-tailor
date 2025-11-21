"""
Core Configuration Management
-----------------------------
This module provides a centralized and type-safe system for managing all
application configurations. It follows modern best practices by using Pydantic
for data validation and settings management.

WHY THIS APPROACH?
- Type Safety: Pydantic enforces correct data types (e.g., int, float, str)
  for all settings, preventing common configuration errors at runtime.
- Centralized Access: A single `get_config()` function provides a singleton
  instance of the configuration, ensuring consistency across the application.
- Hierarchical Loading: Settings are loaded in a specific order, allowing for
  flexible overrides:
    1. Pydantic Model Defaults (lowest priority)
    2. Values from `config/settings.yaml`
    3. Environment Variables (e.g., from a `.env` file, highest priority)
- Self-Documenting: The Pydantic models serve as clear documentation for all
  available settings, their types, and default values.
- IDE Friendly: Provides autocompletion and type checking for configuration
  access in your IDE.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ==============================================================================
# 1. PATHS AND FILE LOCATIONS
# ==============================================================================
# Define base paths to locate configuration files and other resources.

# The root directory of the source code (`src/`).
# We use `Path(__file__).parent.parent` to go up two levels from `src/core/config.py`.
SRC_ROOT = Path(__file__).parent.parent

# The root directory of the entire project.
PROJECT_ROOT = SRC_ROOT.parent

# Path to the main configuration file.
SETTINGS_YAML_PATH = SRC_ROOT / "config" / "settings.yaml"
AGENTS_YAML_PATH = SRC_ROOT / "config" / "agents.yaml"
TASKS_YAML_PATH = SRC_ROOT / "config" / "tasks.yaml"

# ==============================================================================
# 2. YAML LOADING UTILITY
# ==============================================================================
# A helper function to load settings from the `settings.yaml` file. This is a
# key part of the hierarchical loading strategy.


def yaml_config_settings_source() -> dict[str, Any]:
    """
    A Pydantic settings source that loads variables from a YAML file.
    This function is designed to be used with Pydantic's `customise_sources`.
    It is a simple callable that takes no arguments and returns a dictionary
    of settings.

    It reads the `settings.yaml` file and returns its contents as a dictionary,
    which Pydantic then uses to populate the settings models.
    """
    if not SETTINGS_YAML_PATH.exists():
        return {}  # Return empty dict if file doesn't exist

    try:
        with open(SETTINGS_YAML_PATH) as f:
            return yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as e:
        # In a real application, you'd want a logger here.
        print(f"Warning: Could not load {SETTINGS_YAML_PATH}. Error: {e}")
        return {}


# ==============================================================================
# 3. PYDANTIC CONFIGURATION MODELS
# ==============================================================================
# These models define the structure and validation rules for all settings.
# Each class corresponds to a section in the `settings.yaml` file.


class ApplicationConfig(BaseModel):
    """General application settings."""

    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False


class FeatureFlags(BaseModel):
    """Feature flags for enabling/disabling experimental functionality."""

    enable_cache: bool = True
    enable_web_search: bool = False
    enable_human_in_the_loop: bool = False


class LLMGoogleConfig(BaseModel):
    """Configuration specific to Google's LLM models."""

    model: str = "gemini-pro"


class LLMConfig(BaseModel):
    """Default LLM settings for all agents."""

    provider: Literal["openai", "anthropic", "ollama", "google"] = "openai"
    model: str = "gpt-4"
    temperature: float = 0.3
    timeout: int = 120
    max_retries: int = 3
    google: LLMGoogleConfig = Field(default_factory=LLMGoogleConfig)


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: Literal["json", "text"] = "text"
    log_file: str | None = "logs/resume_tailor.log"


class QualityMetricsConfig(BaseModel):
    """Thresholds for the Quality Assurance agent's evaluation."""

    ats_keyword_density_threshold: float = 0.05
    min_professional_summary_length: int = 100
    max_professional_summary_length: int = 500
    min_bullet_points_per_experience: int = 3
    max_bullet_points_per_experience: int = 8


class WorkflowConfig(BaseModel):
    """Parameters that control the resume tailoring workflow logic."""

    max_iterations: int = 3
    quality_threshold: float = 80.0
    quality_metrics: QualityMetricsConfig = Field(default_factory=QualityMetricsConfig)


class ServicesConfig(BaseModel):
    """Configuration for external services."""

    redis_url: str | None = None
    database_url: str | None = None


# ==============================================================================
# 4. MAIN SETTINGS CLASS (AGGREGATOR)
# ==============================================================================
# This is the main class that aggregates all other configuration models.
# It uses Pydantic's `BaseSettings` to automatically read from environment
# variables and the custom YAML source.


class Settings(BaseSettings):
    """
    The main settings model for the entire application.
    It orchestrates the loading of all configurations.
    """

    # Nested configuration models
    application: ApplicationConfig = Field(default_factory=ApplicationConfig)
    feature_flags: FeatureFlags = Field(default_factory=FeatureFlags)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)
    services: ServicesConfig = Field(default_factory=ServicesConfig)

    # You can also have top-level settings here, which can be loaded from
    # environment variables directly (e.g., `OPENAI_API_KEY`).
    # Pydantic automatically finds env vars that match field names.
    openai_api_key: str | None = Field(None, alias="OPENAI_API_KEY")
    gemini_api_key: str | None = Field(None, alias="GEMINI_API_KEY")
    serper_api_key: str | None = Field(None, alias="SERPER_API_KEY")

    # This model_config tells Pydantic how to load settings.
    model_config = SettingsConfigDict(
        # For environment variables, use this prefix. E.g., `APP_LOGGING__LEVEL`.
        env_prefix="APP_",
        # Use `__` to separate nested levels in env vars.
        env_nested_delimiter="__",
        # Allow loading from a .env file.
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        # Case-insensitivity for environment variables.
        case_sensitive=False,
        # Ignore extra variables from sources (e.g., old .env variables)
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        """
        Customizes the loading sources for Pydantic settings, defining the hierarchy.

        The order of sources returned determines their priority. Sources at the
        beginning of the tuple have a higher priority and will override later sources.

        HIERARCHY (from highest to lowest priority):
        1. `init_settings`: Explicit keyword arguments passed to the constructor.
        2. `env_settings`: System environment variables.
        3. `dotenv_settings`: Variables loaded from the `.env` file.
        4. `file_secret_settings`: Variables loaded from secrets files.
        5. `yaml_config_settings_source`: Loads base settings from `settings.yaml`.
        """
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
            yaml_config_settings_source,
        )


# ==============================================================================
# 5. CONFIGURATION SINGLETON
# ==============================================================================
# The `@lru_cache` decorator turns the `get_config` function into a fast,
# thread-safe singleton. The first time it's called, it will create the
# Settings instance. Subsequent calls will return the same cached instance,
# avoiding the cost of re-reading files and environment variables.


@lru_cache
def get_config() -> Settings:
    """
    Returns the global, singleton instance of the application settings.
    """
    return Settings()


@lru_cache
def get_agents_config() -> dict:
    """
    Loads and returns the agent definitions from agents.yaml.
    """
    if not AGENTS_YAML_PATH.exists():
        raise FileNotFoundError(f"Agent configuration not found at: {AGENTS_YAML_PATH}")
    with open(AGENTS_YAML_PATH) as f:
        return yaml.safe_load(f)


@lru_cache
def get_tasks_config() -> dict:
    """
    Loads and returns the task definitions from tasks.yaml.
    """
    if not TASKS_YAML_PATH.exists():
        raise FileNotFoundError(f"Task configuration not found at: {TASKS_YAML_PATH}")
    with open(TASKS_YAML_PATH) as f:
        return yaml.safe_load(f)


# ==============================================================================
# 6. EXAMPLE USAGE (for testing and demonstration)
# ==============================================================================
# This block will only run when the script is executed directly, e.g.,
# `python src/core/config.py`. It's useful for quickly validating that your
# configuration system is working as expected.

if __name__ == "__main__":
    # Load the main settings
    config = get_config()

    # Pretty-print the loaded settings
    import json

    print("--- Loaded Settings ---")
    print(json.dumps(config.model_dump(), indent=2))

    print("\n--- Testing Access ---")
    print(f"Environment: {config.application.environment}")
    print(f"Log Level: {config.logging.level}")
    print(f"LLM Provider: {config.llm.provider}")
    print(f"Default LLM Model: {config.llm.model}")
    print(f"Quality Threshold: {config.workflow.quality_threshold}")

    # Load agent and task configs
    try:
        agents_config = get_agents_config()
        print("\n--- Loaded Agent Config (first agent) ---")
        first_agent_key = next(iter(agents_config))
        print(f"'{first_agent_key}': {agents_config[first_agent_key]['role']}")

        tasks_config = get_tasks_config()
        print("\n--- Loaded Task Config (first task) ---")
        first_task_key = next(iter(tasks_config))
        print(f"'{first_task_key}': {tasks_config[first_task_key]['agent']}")

    except FileNotFoundError as e:
        print(f"\nCould not load agent/task configs: {e}")
    except (KeyError, StopIteration):
        print("\nAgent/Task config files seem to be empty.")

    print("\nConfiguration system appears to be working.")
