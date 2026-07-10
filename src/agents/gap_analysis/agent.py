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

from crewai import LLM, Agent  # LLM wraps the configured model; Agent is the CrewAI persona

from src.agents.agent_config import load_agent_config  # shared YAML config loader/validator
from src.core.logger import get_logger
from src.core.settings import get_config  # runtime defaults: max_iter, max_rpm, retries, etc.

logger = get_logger(__name__)


# ── factory ───────────────────────────────────────────────────────────────────


def create_gap_analysis_agent() -> Agent:
    """Build a tool-free CrewAI Agent that reasons from its formatted context.

    Expects: agents.yaml has a 'gap_analysis_specialist' key with
             role, goal, backstory, and llm fields.
    Returns: a configured CrewAI Agent.
    Raises: RuntimeError if required config fields are missing.
    """
    ####################################################
    # STEP 1: LOAD CONFIG AND BUILD THE LLM INSTANCE
    ####################################################
    config = load_agent_config("gap_analysis_specialist")  # role/goal/backstory/llm from YAML
    llm_instance = LLM(model=config["llm"], temperature=config.get("temperature", 0.3))

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
        allow_delegation=False,  # this agent must not hand its analysis task off to another agent
        tools=[],  # no tools — match findings arrive pre-computed in the task context
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
        "Gap Analysis agent created",
        model=config["llm"],
        tools="none (match findings are pre-computed in context)",
    )
    return agent
