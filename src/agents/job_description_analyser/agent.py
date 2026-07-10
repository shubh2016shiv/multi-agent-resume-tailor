"""
Job Description Analyst agent factory.

Builds a CrewAI Agent that extracts a structured JobDescription from job posting
Markdown supplied by the orchestration layer. No runtime tools — the document is
converted to Markdown before the task is created, and the text arrives in context.
The agent's persona and task instructions in agents.yaml shape the extraction.
"""

from crewai import LLM, Agent  # LLM wraps the configured model; Agent is the CrewAI persona

from src.agents.agent_config import load_agent_config  # shared YAML config loader/validator
from src.core.logger import get_logger
from src.core.settings import get_config  # runtime defaults: max_iter, max_rpm, retries, etc.

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
    config = load_agent_config("job_description_analyst")  # role/goal/backstory/llm from YAML
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
        allow_delegation=False,  # this agent must not hand its extraction task off to another agent
        tools=[],  # no tools — the job posting text arrives pre-converted in the task context
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
        "Job Description Analyst agent created",
        model=config["llm"],
        tools="none (content-in-context)",
    )
    return agent
