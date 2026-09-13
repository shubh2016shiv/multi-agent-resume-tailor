"""
ATS Optimizer agent factory.

Builds a CrewAI Agent that receives the optimized summary, experience, and skills
(plus the original resume for contact/education and the job description for keyword
targets) as context, and assembles them into a single ATS-aligned Resume.

The agent emits the assembled Resume plus decision notes (AtsOptimizedResume).
Mechanical ATS measurement remains code-owned (engines.py), which lets this stage
use provider-enforced structured output instead of a narration-prone tool loop.

Output contract: AtsOptimizedResume (via Task output_pydantic=AtsOptimizedResume).
"""

from crewai import Agent

from src.agents.agent_config import load_agent_config  # shared YAML config loader/validator
from src.core.llm_factory import build_llm
from src.core.logger import get_logger
from src.core.settings import get_config  # runtime defaults: max_iter, max_rpm, retries, etc.

logger = get_logger(__name__)

# ── factory ───────────────────────────────────────────────────────────────────


def create_ats_optimizer_agent() -> Agent:
    """Build a CrewAI Agent for ATS assembly and compatibility optimization.

    Expects: agents.yaml has an 'ats_optimization_specialist' key with
             role, goal, backstory, and llm fields.
    Returns: a configured CrewAI Agent for strict structured assembly.
    Raises: RuntimeError if required config fields are missing.
    """
    ####################################################
    # STEP 1: LOAD CONFIG AND BUILD THE LLM INSTANCE
    ####################################################
    config = load_agent_config("ats_optimization_specialist")  # role/goal/backstory/llm from YAML
    llm_instance = build_llm(config)

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
        structured_output=True,
    )
    return agent
