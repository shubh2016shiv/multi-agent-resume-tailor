"""
Professional Experience agent factory.

Builds a CrewAI Agent that writes one role's experience bullets from role-scoped
TOON context and returns OptimizedExperienceSection through Task output_pydantic.
Audit, rewrite decisions, ID restoration, and merge happen in orchestration.
"""

from crewai import LLM, Agent

from src.core.logger import get_logger
from src.core.settings import get_agents_config, get_config

logger = get_logger(__name__)

# ── tool set ──────────────────────────────────────────────────────────────────

# Stage 3 / Step 3.1: Build the one-role writing agent
# Receives from: src/orchestration/nodes.py role-scoped CrewAI task.
# Sends to: CrewAI Task output_pydantic=OptimizedExperienceSection.
_EXPERIENCE_TOOLS = []


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
            f"Add all required fields to src/config/agents.yaml."
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


def create_professional_experience_agent() -> Agent:
    """Build a CrewAI Agent that writes one role and calls no tools.

    Expects: agents.yaml has a 'experience_section_optimizer' key with
             role, goal, backstory, and llm fields.
    Returns: a configured CrewAI Agent.
    Raises: RuntimeError if required config fields are missing.
    """
    config = _load_agent_config("experience_section_optimizer")
    llm_instance = LLM(model=config["llm"], temperature=config.get("temperature", 0.5))

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
        tools=_EXPERIENCE_TOOLS,
        **runtime_limits,
    )

    tool_names = [t.name for t in _EXPERIENCE_TOOLS]
    logger.info(
        "Professional Experience agent created",
        model=config["llm"],
        tools=tool_names,
    )
    return agent
