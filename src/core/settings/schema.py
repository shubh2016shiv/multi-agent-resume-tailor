"""Typed application settings models.

This is the file to open when asking, "what config exists and what can I set?"
Each model corresponds to a section in `src/config/settings.yaml` or an
environment variable consumed by `Settings`.
"""

from typing import Literal

from pydantic import BaseModel, Field


class ApplicationConfig(BaseModel):
    """General application settings."""

    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False


class FeatureFlags(BaseModel):
    """Feature flags for enabling/disabling experimental functionality."""

    enable_cache: bool = True
    enable_web_search: bool = False
    enable_human_in_the_loop: bool = False
    enable_condensed_formatting: bool = Field(
        default=True,
        description=(
            "Enable aggressive token optimization in formatters. "
            "True condenses verbose job requirements, deduplicates keywords, "
            "and uses compact TOON format. False sends full data for debugging."
        ),
    )


class LLMGoogleConfig(BaseModel):
    """Configuration specific to Google's LLM models."""

    model: str = "gemini-pro"


class LLMResilienceConfig(BaseModel):
    """Retry, circuit breaker, rate-limit, and timeout settings for LLM calls."""

    retry_max_attempts: int = 3
    retry_initial_delay: float = 1.0
    retry_max_delay: float = 60.0
    retry_exponential_multiplier: float = 2.0
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_timeout: int = 60
    rate_limit_calls_per_minute: int = 60
    timeout_seconds: int = 30


class AgentDefaults(BaseModel):
    """Default CrewAI resilience settings applied by agent factories."""

    max_retry_limit: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum number of retries when an agent encounters an error",
    )
    max_rpm: int = Field(
        default=60,
        ge=1,
        description="Maximum requests per minute to respect API rate limits",
    )
    max_iter: int = Field(
        default=15,
        ge=1,
        le=50,
        description="Maximum iterations before forcing an agent's best answer",
    )
    max_execution_time: int = Field(
        default=300,
        ge=30,
        description="Maximum execution time in seconds for a single task",
    )
    respect_context_window: bool = Field(
        default=True,
        description="Auto-summarize context to prevent context limit errors",
    )
    verbose: bool = Field(default=True, description="Enable detailed logging")


class LLMConfig(BaseModel):
    """Default LLM settings for all agents."""

    provider: Literal["openai", "anthropic", "ollama", "google"] = "openai"
    model: str = "gpt-4"
    temperature: float = 0.3
    timeout: int = 120
    max_retries: int = 3
    structured_input_token_budget: int = Field(
        default=100_000,
        ge=1,
        description=(
            "Maximum combined system and user input tokens allowed before the "
            "structured-output LLM gateway makes a provider call."
        ),
    )
    google: LLMGoogleConfig = Field(default_factory=LLMGoogleConfig)
    resilience: LLMResilienceConfig = Field(default_factory=LLMResilienceConfig)
    agent_defaults: AgentDefaults = Field(default_factory=AgentDefaults)


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
    """Parameters that control the resume tailoring workflow."""

    max_iterations: int = 3
    quality_threshold: float = 80.0
    quality_metrics: QualityMetricsConfig = Field(default_factory=QualityMetricsConfig)


class FilePathsConfig(BaseModel):
    """Input file paths used by local runs and examples."""

    resume_path: str = "sample_documents/resume.pdf"
    job_description_path: str = "sample_documents/job_description.txt"


class ServicesConfig(BaseModel):
    """External service URLs."""

    redis_url: str | None = None
    database_url: str | None = None


class ObservabilityConfig(BaseModel):
    """LangSmith tracing settings (non-secret).

    The API key is NOT here — it is a secret read from the environment as
    `LANGSMITH_API_KEY` (see `Settings.langsmith_api_key`). These fields only
    control whether tracing is on and which project/endpoint it targets.
    """

    enabled: bool = Field(
        default=True,
        description=(
            "Master switch for LangSmith tracing. On by default; still no-ops "
            "safely if LANGSMITH_API_KEY is unset, so the pipeline always runs."
        ),
    )
    project: str = Field(
        default="resume-tailor-agents",
        description="LangSmith project name that traces are grouped under.",
    )
    endpoint: str = Field(
        default="https://api.smith.langchain.com",
        description="LangSmith API endpoint (override for self-hosted/EU region).",
    )
