"""Settings construction, source ordering, and process-wide caching.

Open this file when you want to answer three questions:

1. What Python object represents the whole application's typed settings?
2. In what order do YAML, environment variables, `.env`, and direct Python
   overrides win over each other?
3. Why does the rest of the codebase call `get_config()` instead of building
   `Settings()` everywhere?

Source precedence, highest to lowest:
1. Explicit `Settings(...)` initialization values.
2. Environment variables.
3. `.env` values.
4. File secrets.
5. `src/config/settings.yaml`.
6. Pydantic model defaults.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.core.settings.paths import PROJECT_ROOT
from src.core.settings.schema import (
    ApplicationConfig,
    FeatureFlags,
    FilePathsConfig,
    LLMConfig,
    LoggingConfig,
    ObservabilityConfig,
    PromptCatalogConfig,
    ServicesConfig,
    WorkflowConfig,
)
from src.core.settings.yaml_source import yaml_config_settings_source


class Settings(BaseSettings):
    """The typed application settings exposed to Python code."""

    application: ApplicationConfig = Field(default_factory=ApplicationConfig)
    feature_flags: FeatureFlags = Field(default_factory=FeatureFlags)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)
    file_paths: FilePathsConfig = Field(default_factory=FilePathsConfig)
    prompt_catalog: PromptCatalogConfig = Field(default_factory=PromptCatalogConfig)
    services: ServicesConfig = Field(default_factory=ServicesConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    openai_api_key: str | None = Field(None, alias="OPENAI_API_KEY")
    gemini_api_key: str | None = Field(None, alias="GEMINI_API_KEY")
    serper_api_key: str | None = Field(None, alias="SERPER_API_KEY")
    langsmith_api_key: str | None = Field(None, alias="LANGSMITH_API_KEY")

    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_nested_delimiter="__",
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        _settings_cls: type[BaseSettings],
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        """Return settings sources in override order.

        `pydantic-settings` calls this method with a fixed parameter list.
        `_settings_cls` is part of that call shape, even though this repo does
        not need it. We keep it because the library passes it in; we ignore it
        because this project uses one fixed source order for one `Settings`
        class.
        """
        ####################################################
        # STEP 1: KEEP EXPLICIT PYTHON OVERRIDES FIRST#
        ####################################################
        # Callers who instantiate Settings(...) directly should win over every
        # file-based or environment-based source below.

        ####################################################
        # STEP 2: KEEP ENV VARS AND .ENV AHEAD OF YAML#
        ####################################################
        # Runtime overrides belong above the project-default YAML so operators
        # can change behavior without editing tracked files.

        ####################################################
        # STEP 3: TREAT settings.yaml AS THE LAST EXTERNAL SOURCE#
        ####################################################
        # YAML supplies project defaults, not the highest-priority override.
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
            yaml_config_settings_source,
        )


@lru_cache
def get_config() -> Settings:
    """Return the process-wide cached application settings instance."""
    ####################################################
    # STEP 1: BUILD SETTINGS ON FIRST ACCESS#
    ####################################################
    # BaseSettings resolves all configured sources only once here.

    ####################################################
    # STEP 2: REUSE THE SAME SETTINGS OBJECT FOR THE PROCESS#
    ####################################################
    # The LRU cache keeps settings lookup cheap and ensures callers across the
    # process all read the same resolved configuration snapshot.
    # Pyright cannot model BaseSettings' dynamic source resolution here and
    # incorrectly treats aliased environment-backed fields as required args.
    return Settings()  # pyright: ignore[reportCallIssue]
