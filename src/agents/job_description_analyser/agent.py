"""
Job Description Analyst agent factory.

Builds a CrewAI Agent that extracts a structured JobDescription from job posting
Markdown supplied by the orchestration layer. No runtime tools — the document is
converted to Markdown before the task is created, and the text arrives in context.
The agent's persona and task instructions in agents.yaml shape the extraction.
"""

from crewai import LLM, Agent

from src.core.logger import get_logger
from src.core.settings import get_agents_config, get_config

logger = get_logger(__name__)

# ── tool set ──────────────────────────────────────────────────────────────────
# No runtime tools: job description Markdown is provided in task context by the
# orchestration node. The agent reads and extracts — it does not fetch or parse files.

_JDA_TOOLS: list = []


# ── config ────────────────────────────────────────────────────────────────────


def _load_agent_config(name: str) -> dict:
    """Load and validate an agent config block from agents.yaml.

    Expects: agents.yaml has a key matching `name` with role, goal, backstory, llm.
    Returns: the config dict.
    Raises: RuntimeError if any required field is missing.
    """
    agents_config = get_agents_config()
    config = agents_config.get(name, {})

    required = ["role", "goal", "backstory", "llm"]
    missing = [f for f in required if not config.get(f)]
    if missing:
        raise RuntimeError(
            f"FATAL: Missing required field(s) in '{name}' agent config: {missing}\n"
            "Add all required fields to src/config/agents.yaml."
        )
    return config


def _build_runtime_limits(config: dict, defaults) -> dict:
    """Build CrewAI runtime limit kwargs from agent config.

    Expects an agent config dict and the configured default limit object.
    Returns kwargs for Agent construction, falling back to defaults when unset.
    """
    return {
        "max_retry_limit": config.get("max_retry_limit", defaults.max_retry_limit),
        "max_rpm": config.get("max_rpm", defaults.max_rpm),
        "max_iter": config.get("max_iter", defaults.max_iter),
        "max_execution_time": config.get("max_execution_time", defaults.max_execution_time),
        "respect_context_window": config.get(
            "respect_context_window",
            defaults.respect_context_window,
        ),
    }


# ── factory ───────────────────────────────────────────────────────────────────


def create_job_analyzer_agent() -> Agent:
    """Build a CrewAI Agent that extracts JobDescription from Markdown context.

    Expects: agents.yaml has a 'job_description_analyst' key with
             role, goal, backstory, and llm fields.
    Returns: a configured CrewAI Agent.
    Raises: RuntimeError if required config fields are missing.
    """
    config = _load_agent_config("job_description_analyst")
    llm_instance = LLM(model=config["llm"], temperature=config.get("temperature", 0.2))

    app_config = get_config()
    defaults = app_config.llm.agent_defaults
    runtime_limits = _build_runtime_limits(config, defaults)

    agent = Agent(
        role=config["role"],
        goal=config["goal"],
        backstory=config["backstory"],
        llm=llm_instance,
        verbose=config.get("verbose", True),
        allow_delegation=False,
        tools=_JDA_TOOLS,
        **runtime_limits,
    )

    logger.info(
        "Job Description Analyst agent created",
        model=config["llm"],
        tools="none (content-in-context)",
    )
    return agent
