"""
Skill Optimizer agent factory.

Builds a CrewAI Agent that receives Resume + JobDescription + AlignmentStrategy
as context and produces an OptimizedSkillsSection. The agent reorders, categorizes,
and prioritizes skills based on the strategic guidance (skills_guidance,
keywords_to_integrate).

The agent uses tools=[] — all quality checks run code-owned on typed output after
the agent finishes, not through agent tool calls. The check_skills_evidence engine
(in src/tools/truthfulness/) validates that every listed skill is evidenced in
the resume. The audit runs in orchestration/nodes, not inside the agent loop.

Output contract: OptimizedSkillsSection (via Task output_pydantic).
"""

from crewai import LLM, Agent

from src.core.config import get_agents_config, get_config
from src.core.logger import get_logger

logger = get_logger(__name__)


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


def create_skill_optimizer_agent() -> Agent:
    """Build a CrewAI Agent for skill selection, ordering, and categorization.

    Expects: agents.yaml has a 'skills_section_strategist' key with
             role, goal, backstory, and llm fields.
    Returns: a configured CrewAI Agent with no tools.
    Raises: RuntimeError if required config fields are missing.
    """
    config = _load_agent_config("skills_section_strategist")
    llm_instance = LLM(model=config["llm"])

    app_config = get_config()
    defaults = app_config.llm.agent_defaults

    agent = Agent(
        role=config["role"],
        goal=config["goal"],
        backstory=config["backstory"],
        llm=llm_instance,
        temperature=config.get("temperature", 0.4),
        verbose=config.get("verbose", True),
        allow_delegation=False,
        tools=[],
        max_retry_limit=defaults.max_retry_limit,
        max_rpm=defaults.max_rpm,
        max_iter=defaults.max_iter,
        max_execution_time=defaults.max_execution_time,
        respect_context_window=defaults.respect_context_window,
    )

    logger.info(
        "Skill Optimizer agent created",
        model=config["llm"],
        tools="none (audit is code-owned)",
    )
    return agent
