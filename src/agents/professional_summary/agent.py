"""
Professional Summary agent factory.

Pipeline position: node STEP 3 (see src/orchestration/nodes/summary.py). The node
calls create_professional_summary_agent() to build the writer, then runs the task.

Builds a CrewAI Agent wired with the audit_summary tool — a hybrid tool that
reviews a professional summary for length, first-person voice, generic boilerplate,
and missing value proposition, using both mechanical checks and LLM judgment.

The agent receives Resume + JobDescription + AlignmentStrategy as context, reads
the strategic guidance (professional_summary_guidance, keywords_to_integrate),
generates 4 summary drafts using different narrative frameworks, self-critiques
each one, and recommends the strongest version.

Output contract: ProfessionalSummary (via Task output_pydantic=ProfessionalSummary).
"""

from crewai import LLM, Agent  # LLM wraps the configured model; Agent is the CrewAI persona

from src.agents.agent_config import load_agent_config  # shared YAML config loader/validator
from src.core.logger import get_logger
from src.core.settings import get_config  # runtime defaults: max_iter, max_rpm, retries, etc.
from src.tools.agent_tools import audit_summary

logger = get_logger(__name__)

# ── tool set ──────────────────────────────────────────────────────────────────
# Hybrid tool (mechanical checks + LLM judgment) the agent calls while
# reasoning to self-critique each draft before recommending one.

SUMMARY_AUDIT_TOOLS: list = [
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
    config = load_agent_config("professional_summary_writer")  # role/goal/backstory/llm from YAML
    llm_instance = LLM(model=config["llm"], temperature=config.get("temperature", 0.7))

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
        allow_delegation=False,  # this agent must not hand its writing task off to another agent
        tools=SUMMARY_AUDIT_TOOLS,  # the audit_summary self-critique tool defined above
        max_retry_limit=defaults.max_retry_limit,  # retries on a failed/malformed LLM call
        max_rpm=defaults.max_rpm,  # caps requests-per-minute to this agent's LLM
        max_iter=defaults.max_iter,  # caps reasoning/tool-call loops before forcing an answer
        max_execution_time=defaults.max_execution_time,  # hard wall-clock timeout for one run
        respect_context_window=defaults.respect_context_window,  # auto-trim context instead of erroring
    )

    ####################################################
    # STEP 3: LOG AND RETURN
    ####################################################
    logger.info(
        "Professional Summary agent created",
        model=config["llm"],
        tools=[tool.name for tool in SUMMARY_AUDIT_TOOLS],
    )
    return agent
