"""
Quality Feedback agent factory.

Builds a CrewAI Agent that audits the tailored resume and writes advisory feedback.

It is wired with two read-only audit tools the agent calls WHILE reasoning:
  - audit_truthfulness          -> accuracy  (original vs revised: invented/inflated facts)
  - analyze_jd_keyword_coverage -> relevance (job keyword coverage + density)
The tools never mutate the resume.

ATS is NOT an agent tool here. ATS compatibility is graded by code on the REAL rendered
artifact (resume_quality_evaluation.evaluate_rendered_structure inspects the .tex).
The agent has no rendered
artifact, only its lossy context, so any self-cert it made would be a false positive that
code discards anyway -- so the tool is removed rather than overridden.

Output contract: QualityFeedback (src/data_models/evaluation.py). The agent cannot
write scores or release decisions.
"""

from crewai import LLM, Agent  # LLM wraps the configured model; Agent is the CrewAI persona

from src.agents.agent_config import load_agent_config  # shared YAML config loader/validator
from src.core.logger import get_logger
from src.core.settings import get_config  # runtime defaults: max_iter, max_rpm, retries, etc.
from src.tools.agent_tools import (
    analyze_jd_keyword_coverage,
    audit_truthfulness,
)

logger = get_logger(__name__)

# ── tool set ──────────────────────────────────────────────────────────────────
# ATS is graded by code on the rendered .tex
# (resume_quality_evaluation.evaluate_rendered_structure), not by the
# agent -- so validate_ats_compliance is intentionally NOT in this list.

QUALITY_FEEDBACK_TOOLS: list = [
    audit_truthfulness,
    analyze_jd_keyword_coverage,
]


# ── factory ───────────────────────────────────────────────────────────────────


def create_quality_feedback_agent() -> Agent:
    """Build the optional agent that writes grounded resume quality feedback.

    Expects: agent config has a 'quality_feedback_reviewer' key with
             role, goal, backstory, and llm fields.
    Returns a configured agent with truthfulness and keyword-coverage tools.
    Raises: RuntimeError if required config fields are missing.
    """
    ####################################################
    # STEP 1: LOAD CONFIG AND BUILD THE LLM INSTANCE
    ####################################################
    config = load_agent_config("quality_feedback_reviewer")  # role/goal/backstory/llm from YAML
    llm_instance = LLM(model=config["llm"], temperature=config.get("temperature", 0.2))

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
        allow_delegation=False,  # this agent must not hand its audit task off to another agent
        tools=QUALITY_FEEDBACK_TOOLS,  # truthfulness + keyword-coverage audit tools defined above
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
        "Quality Feedback agent created",
        model=config["llm"],
        tools=[tool.name for tool in QUALITY_FEEDBACK_TOOLS],
    )
    return agent
