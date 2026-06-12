"""
Professional Summary agent factory.

Builds a CrewAI Agent wired with the audit_summary tool — a hybrid tool that
reviews a professional summary for length, first-person voice, generic boilerplate,
and missing value proposition, using both mechanical checks and LLM judgment.

The agent receives Resume + JobDescription + AlignmentStrategy as context, reads
the strategic guidance (professional_summary_guidance, keywords_to_integrate),
generates 4 summary drafts using different narrative frameworks, self-critiques
each one, and recommends the strongest version.

Output contract: ProfessionalSummary (via Task output_pydantic=ProfessionalSummary).
"""

from crewai import LLM, Agent

from src.core.logger import get_logger
from src.core.settings import get_agents_config, get_config
from src.tools.agent_facing_tools import audit_summary

logger = get_logger(__name__)

# ── tool set ──────────────────────────────────────────────────────────────────

_SUMMARY_TOOLS: list = [
    audit_summary,
]


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


# ── factory ───────────────────────────────────────────────────────────────────


def create_professional_summary_agent() -> Agent:
    """Build a CrewAI Agent with the audit_summary tool.

    Expects: agents.yaml has a 'professional_summary_writer' key with
             role, goal, backstory, and llm fields.
    Returns: a configured CrewAI Agent.
    Raises: RuntimeError if required config fields are missing.
    """
    config = _load_agent_config("professional_summary_writer")
    llm_instance = LLM(model=config["llm"], temperature=config.get("temperature", 0.7))

    app_config = get_config()
    defaults = app_config.llm.agent_defaults

    agent = Agent(
        role=config["role"],
        goal=config["goal"],
        backstory=config["backstory"],
        llm=llm_instance,
        verbose=config.get("verbose", True),
        allow_delegation=False,
        tools=_SUMMARY_TOOLS,
        max_retry_limit=defaults.max_retry_limit,
        max_rpm=defaults.max_rpm,
        max_iter=defaults.max_iter,
        max_execution_time=defaults.max_execution_time,
        respect_context_window=defaults.respect_context_window,
    )

    tool_names = [t.name for t in _SUMMARY_TOOLS]
    logger.info(
        "Professional Summary agent created",
        model=config["llm"],
        tools=tool_names,
    )
    return agent
