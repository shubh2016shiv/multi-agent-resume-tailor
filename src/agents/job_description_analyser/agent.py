"""
Job Description Analyst agent factory.

Builds a CrewAI Agent that extracts a structured JobDescription from job posting
Markdown supplied by the orchestration layer. No runtime tools — the document is
converted to Markdown before the task is created, and the text arrives in context.
The agent's persona and task instructions in agents.yaml shape the extraction.
"""

from crewai import LLM, Agent

from src.agents.agent_config import load_agent_config
from src.core.logger import get_logger
from src.core.settings import get_config

logger = get_logger(__name__)


# ── factory ───────────────────────────────────────────────────────────────────


def create_job_analyzer_agent() -> Agent:
    """Build a CrewAI Agent that extracts JobDescription from Markdown context.

    Expects: agents.yaml has a 'job_description_analyst' key with
             role, goal, backstory, and llm fields.
    Returns: a configured CrewAI Agent.
    Raises: RuntimeError if required config fields are missing.
    """
    ####################################################
    # STEP 1: LOAD CONFIG AND BUILD THE LLM INSTANCE
    ####################################################
    config = load_agent_config("job_description_analyst")
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
        "Job Description Analyst agent created",
        model=config["llm"],
        tools="none (content-in-context)",
    )
    return agent
