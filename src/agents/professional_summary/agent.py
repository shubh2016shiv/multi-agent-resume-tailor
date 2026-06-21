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

from src.agents.agent_config import load_agent_config
from src.core.logger import get_logger
from src.core.settings import get_config
from src.tools.agent_tools import audit_summary

logger = get_logger(__name__)

# ── tool set ──────────────────────────────────────────────────────────────────

_SUMMARY_TOOLS: list = [
    audit_summary,
]


# ── factory ───────────────────────────────────────────────────────────────────


def create_professional_summary_agent() -> Agent:
    """Build a CrewAI Agent with the audit_summary tool.

    Expects: agents.yaml has a 'professional_summary_writer' key with
             role, goal, backstory, and llm fields.
    Returns: a configured CrewAI Agent.
    Raises: RuntimeError if required config fields are missing.
    """
    ####################################################
    # STEP 1: LOAD CONFIG AND BUILD THE LLM INSTANCE
    ####################################################
    config = load_agent_config("professional_summary_writer")
    llm_instance = LLM(model=config["llm"], temperature=config.get("temperature", 0.7))

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
        tools=_SUMMARY_TOOLS,
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
        "Professional Summary agent created",
        model=config["llm"],
        tools=[tool.name for tool in _SUMMARY_TOOLS],
    )
    return agent
