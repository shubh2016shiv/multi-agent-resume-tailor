"""
Professional Experience agent factory.

Builds a CrewAI Agent that writes one role's experience bullets from role-scoped
TOON context and returns OptimizedExperienceSection through Task output_pydantic.
Audit, rewrite decisions, ID restoration, and merge happen in orchestration.
"""

from crewai import LLM, Agent

from src.agents.agent_config import load_agent_config
from src.core.logger import get_logger
from src.core.settings import get_config

logger = get_logger(__name__)


# ── factory ───────────────────────────────────────────────────────────────────


def create_professional_experience_agent() -> Agent:
    """Build a CrewAI Agent that writes one role and calls no tools.

    Expects: agents.yaml has a 'experience_section_optimizer' key with
             role, goal, backstory, and llm fields.
    Returns: a configured CrewAI Agent.
    Raises: RuntimeError if required config fields are missing.
    """
    ####################################################
    # STEP 1: LOAD CONFIG AND BUILD THE LLM INSTANCE
    ####################################################
    config = load_agent_config("experience_section_optimizer")
    llm_instance = LLM(model=config["llm"], temperature=config.get("temperature", 0.0))

    ####################################################
    # STEP 2: BUILD THE AGENT WITH RUNTIME DEFAULTS
    ####################################################
    defaults = get_config().llm.agent_defaults

    agent = Agent(
        role=config["role"],
        goal=config["goal"],
        backstory=config["backstory"],
        llm=llm_instance,
        verbose=config.get("verbose", True),
        allow_delegation=False,
        tools=[],
        max_retry_limit=defaults.max_retry_limit,
        max_rpm=defaults.max_rpm,
        max_iter=defaults.max_iter,
        max_execution_time=defaults.max_execution_time,
        respect_context_window=defaults.respect_context_window,
    )

    ####################################################
    # STEP 3: LOG AND RETURN
    ####################################################
    logger.info(
        "Professional Experience agent created",
        model=config["llm"],
        tools="none (audit and merge are orchestration-owned)",
    )
    return agent
