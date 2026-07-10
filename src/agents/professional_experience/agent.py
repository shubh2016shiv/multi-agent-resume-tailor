"""
Professional Experience agent factory.

Builds a CrewAI Agent that writes one role's experience bullets from role-scoped
TOON context and returns OptimizedExperienceSection through Task output_pydantic.
Audit, rewrite decisions, ID restoration, and merge happen in orchestration.
"""

from crewai import LLM, Agent  # LLM wraps the configured model; Agent is the CrewAI persona

from src.agents.agent_config import load_agent_config  # shared YAML config loader/validator
from src.core.logger import get_logger
from src.core.settings import get_config  # runtime defaults: max_iter, max_rpm, retries, etc.

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
    config = load_agent_config("experience_section_optimizer")  # role/goal/backstory/llm from YAML
    llm_instance = LLM(model=config["llm"], temperature=config.get("temperature", 0.0))

    ####################################################
    # STEP 2: BUILD THE AGENT WITH RUNTIME DEFAULTS
    ####################################################
    defaults = get_config().llm.agent_defaults  # shared retry/rate-limit/timeout settings

    agent = Agent(
        role=config["role"],
        goal=config["goal"],
        backstory=config["backstory"],
        llm=llm_instance,
        verbose=config.get("verbose", True),
        allow_delegation=False,  # this agent must not hand its rewrite task off to another agent
        tools=[],  # no tools — audit, ID restoration, and merge are orchestration-owned
        max_retry_limit=defaults.max_retry_limit,  # retries on a failed/malformed LLM call
        max_rpm=defaults.max_rpm,  # caps requests-per-minute to this agent's LLM
        max_iter=defaults.max_iter,  # caps reasoning loops before forcing an answer
        max_execution_time=defaults.max_execution_time,  # hard wall-clock timeout for one run
        respect_context_window=defaults.respect_context_window,  # auto-trim context instead of erroring
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
