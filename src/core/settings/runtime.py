"""Settings aggregation and cache.

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
        settings_cls: type[BaseSettings],
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        """Return settings sources in override order."""
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
            yaml_config_settings_source,
        )


@lru_cache
def get_config() -> Settings:
    """Return the process-wide application settings instance."""
    return Settings()
