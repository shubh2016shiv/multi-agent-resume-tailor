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
    """The typed application settings exposed to Python code.

    Implements Pattern 3 (typed/validated) and Pattern 7 (domain sectioning) —
    each field below is one cohesive config section. See CONFIGURATION_PATTERNS.md.
    """

    # Pattern 7 (domain sectioning): one typed section per area of the app.
    application: ApplicationConfig = Field(default_factory=ApplicationConfig)
    feature_flags: FeatureFlags = Field(default_factory=FeatureFlags)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)
    file_paths: FilePathsConfig = Field(default_factory=FilePathsConfig)
    prompt_catalog: PromptCatalogConfig = Field(default_factory=PromptCatalogConfig)
    services: ServicesConfig = Field(default_factory=ServicesConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    # Pattern 4 (config is not secrets): API keys are env-only fields (via alias),
    # never read from tracked YAML. See CONFIGURATION_PATTERNS.md.
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

        This method IS Pattern 2 (layered override chain) — see
        CONFIGURATION_PATTERNS.md. The returned tuple's order is the precedence.

        `pydantic-settings` calls this method with a fixed parameter list.
        `_settings_cls` is part of that call shape, even though this repo does
        not need it. We keep it because the library passes it in; we ignore it
        because this project uses one fixed source order for one `Settings`
        class.

        There is no sequential logic here — the return value IS the answer.
        pydantic-settings tries each source in this tuple's order and the
        first one that defines a given field wins, so position in the tuple is
        priority, highest first:

        1. `init_settings`     — explicit `Settings(...)` constructor args win
                                  over every file/env source below them.
        2. `env_settings`      — real environment variables outrank `.env` and
                                  YAML, so an operator can override behavior
                                  without touching any tracked file.
        3. `dotenv_settings`   — `.env` file values, one step below real env vars.
        4. `file_secret_settings` — Docker/K8s secret-file mounts, if used.
        5. `yaml_config_settings_source` — `src/config/settings.yaml` supplies
                                  project *defaults*, so it sits last: any of
                                  the four sources above override it.
        """
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
            yaml_config_settings_source,
        )


@lru_cache
def get_config() -> Settings:
    """Return the process-wide cached application settings instance.

    This is Pattern 6 (resolve once, share one immutable snapshot) — see
    CONFIGURATION_PATTERNS.md.

    `@lru_cache` on a zero-argument function is what makes this a singleton
    accessor: `Settings()` — which resolves every source in
    `settings_customise_sources()` above — runs exactly once per process, on
    the first call. Every call after that returns the same cached instance
    instead of re-reading files/env vars, and every caller across the codebase
    sees one consistent configuration snapshot for the process's lifetime.

    Pyright cannot model BaseSettings' dynamic source resolution and
    incorrectly treats aliased environment-backed fields as required
    constructor args — hence the ignore comment below.
    """
    return Settings()  # pyright: ignore[reportCallIssue]
