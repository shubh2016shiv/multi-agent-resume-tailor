"""
Gap Analysis agent factory.

Builds a tool-free CrewAI Agent. The requirement/keyword match is computed in code
by the strategy node (match_resume_to_job) and rendered into the agent's context as
match_findings, so the agent reads pre-computed facts rather than calling a tool that
would force it to serialize the resume back from its lossy TOON view.

The agent reasons over the formatted context (candidate profile, job requirements,
match findings) to produce an AlignmentStrategy: identified matches, gaps, keyword
targets, and section-specific guidance for the three downstream content-generation
agents (summary, experience, skills).

Output contract: AlignmentStrategy (via Task output_pydantic=AlignmentStrategy).
"""

from crewai import LLM, Agent

from src.core.logger import get_logger
from src.core.settings import get_agents_config, get_config

logger = get_logger(__name__)

# ── tool set ──────────────────────────────────────────────────────────────────

_GAP_TOOLS: list = []


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


def create_gap_analysis_agent() -> Agent:
    """Build a tool-free CrewAI Agent that reasons from its formatted context.

    Expects: agents.yaml has a 'gap_analysis_specialist' key with
             role, goal, backstory, and llm fields.
    Returns: a configured CrewAI Agent.
    Raises: RuntimeError if required config fields are missing.
    """
    config = _load_agent_config("gap_analysis_specialist")
    llm_instance = LLM(model=config["llm"], temperature=config.get("temperature", 0.3))

    app_config = get_config()
    defaults = app_config.llm.agent_defaults

    agent = Agent(
        role=config["role"],
        goal=config["goal"],
        backstory=config["backstory"],
        llm=llm_instance,
        verbose=config.get("verbose", True),
        allow_delegation=False,
        tools=_GAP_TOOLS,
        max_retry_limit=defaults.max_retry_limit,
        max_rpm=defaults.max_rpm,
        max_iter=defaults.max_iter,
        max_execution_time=defaults.max_execution_time,
        respect_context_window=defaults.respect_context_window,
    )

    tool_names = [t.name for t in _GAP_TOOLS]
    logger.info(
        "Gap Analysis agent created",
        model=config["llm"],
        tools=tool_names,
    )
    return agent
