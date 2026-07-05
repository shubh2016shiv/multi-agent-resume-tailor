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

    enable_cache: bool = Field(
        default=True,
        description=(
            "Cache LLM responses on disk (.litellm_cache/). When True, an identical "
            "(model, messages, params) call is served from disk instead of re-billing the "
            "provider -- so repeated development runs cost nothing after the first. The key "
            "includes the full request, so different resumes/JDs never collide. Set False to "
            "force every call to hit the provider."
        ),
    )
    enable_web_search: bool = False
    enable_human_in_the_loop: bool = False
    enable_pii_redaction: bool = Field(
        default=True,
        description=(
            "Master switch for the resume PII pipeline (redact before the LLM, then "
            "rehydrate after QA). When False, the redact tool passes Markdown through "
            "unchanged, the extraction guard is skipped, and rehydration is a no-op -- "
            "the resume reaches the LLM with real PII. Default True keeps PII masked."
        ),
    )
    enable_condensed_formatting: bool = Field(
        default=True,
        description=(
            "Enable aggressive token optimization in formatters. "
            "True condenses verbose job requirements, deduplicates keywords, "
            "and uses compact TOON format. False sends full data for debugging."
        ),
    )
    render_draft_on_gate_fail: bool = Field(
        default=False,
        description=(
            "When True, render md and docx even if the quality gate fails. "
            "Useful in development: the draft is viewable without inspecting JSON. "
            "Defaults False so gate-failing runs don't litter the output directory."
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
    verbose: bool = Field(
        default=True,
        description=(
            "Centralized on/off switch for CrewAI's own rich-console output -- crew/task "
            "execution trees, tool and LLM call status, and the panels CrewAI prints on "
            "failure (e.g. 'Crew Execution Failed', 'Tool Usage Failed'). Read once in "
            "crew_task_execution.run_agent_task() and passed as Crew(verbose=...) on "
            "every agent call in the pipeline, so this one flag governs it everywhere. "
            "Set to false for quiet runs (scripts, CI) where only our own structlog "
            "output and the final result should print."
        ),
    )


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
    third_party_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "WARNING"
    litellm_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "ERROR"


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
    """Input and output file paths used by local runs and examples."""

    resume_path: str = "sample_documents/resume.pdf"
    job_description_path: str = "sample_documents/job_description.txt"
    # Base directory for produced resume artifacts. Renderers nest under it as
    # <output_dir>/<candidate>/<designation>/ (see document_rendering.output_location).
    output_dir: str = "tailored_resumes"


class PromptCatalogConfig(BaseModel):
    """Central locations for application-owned prompt files."""

    tool_prompts_dir: str = Field(
        default="src/config/tool_prompts",
        description=(
            "Project-relative or absolute path to the centralized tool prompt catalog. "
            "This is application configuration, not agent configuration."
        ),
    )


class ServicesConfig(BaseModel):
    """External service URLs."""

    redis_url: str | None = None
    database_url: str | None = None
    pii_mapping_ttl_seconds: int = Field(
        default=3600,
        ge=60,
        description=(
            "TTL for a run's PII placeholder mapping stored in Redis. A safety net "
            "that reclaims the key if a run dies before its explicit cleanup runs."
        ),
    )


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
