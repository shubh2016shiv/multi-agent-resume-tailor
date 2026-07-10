"""
ATS Optimizer agent factory.

Builds a CrewAI Agent that receives the optimized summary, experience, and skills
(plus the original resume for contact/education and the job description for keyword
targets) as context, and assembles them into a single ATS-aligned Resume.

The agent is wired with two mechanical, read-only audit tools -- validate_ats_compliance
(formatting + standard section headers) and analyze_jd_keyword_coverage (keyword
coverage + density). It consults these WHILE reasoning to verify its assembly is
ATS-safe; the tools never mutate the resume. The agent's output is the assembled
Resume plus decision notes (AtsOptimizedResume) -- it does not render markdown/JSON
or self-score a validation report. That measurement is code-owned (engines.py).

Output contract: AtsOptimizedResume (via Task output_pydantic=AtsOptimizedResume).
"""

from crewai import LLM, Agent  # LLM wraps the configured model; Agent is the CrewAI persona

from src.agents.agent_config import load_agent_config  # shared YAML config loader/validator
from src.core.logger import get_logger
from src.core.settings import get_config  # runtime defaults: max_iter, max_rpm, retries, etc.
from src.tools.agent_tools import analyze_jd_keyword_coverage, validate_ats_compliance

logger = get_logger(__name__)

# ── tool set ──────────────────────────────────────────────────────────────────
# Read-only audit tools the agent calls WHILE reasoning to self-check its
# assembly; neither tool mutates the resume.

ATS_AUDIT_TOOLS: list = [
    validate_ats_compliance,
    analyze_jd_keyword_coverage,
]


# ── factory ───────────────────────────────────────────────────────────────────


def create_ats_optimizer_agent() -> Agent:
    """Build a CrewAI Agent for ATS assembly and compatibility optimization.

    Expects: agents.yaml has an 'ats_optimization_specialist' key with
             role, goal, backstory, and llm fields.
    Returns: a configured CrewAI Agent wired with the two ATS audit tools.
    Raises: RuntimeError if required config fields are missing.
    """
    ####################################################
    # STEP 1: LOAD CONFIG AND BUILD THE LLM INSTANCE
    ####################################################
    config = load_agent_config("ats_optimization_specialist")  # role/goal/backstory/llm from YAML
    llm_instance = LLM(model=config["llm"], temperature=config.get("temperature", 0.1))

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
        allow_delegation=False,  # this agent must not hand its assembly task off to another agent
        tools=ATS_AUDIT_TOOLS,  # the two read-only audit tools defined above
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
        "ATS Optimizer agent created",
        model=config["llm"],
        tools=[tool.name for tool in ATS_AUDIT_TOOLS],
    )
    return agent
