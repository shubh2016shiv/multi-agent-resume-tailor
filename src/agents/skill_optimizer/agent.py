"""
Skill Optimizer agent factory.

Pipeline position: node STEP 3 (see src/orchestration/nodes/skills.py). The node
calls create_skill_optimizer_agent() to build the agent, then runs the task
(write_skills_section, node STEP 3 & 4).

Builds a CrewAI Agent that receives Resume + JobDescription + AlignmentStrategy
as context and produces an OptimizedSkillsSection. The agent reorders, categorizes,
and prioritizes skills based on the strategic guidance (skills_guidance,
keywords_to_integrate).

The agent uses tools=[] — all quality checks run code-owned on typed output after
the agent finishes, not through agent tool calls. The check_skills_evidence engine
(in src/tools/truthfulness/) validates that every listed skill is evidenced in
the resume. The audit runs in orchestration/nodes, not inside the agent loop.

Output contract: OptimizedSkillsSection (via Task output_pydantic).
"""

from crewai import LLM, Agent  # LLM wraps the configured model; Agent is the CrewAI persona

from src.agents.agent_config import load_agent_config  # shared YAML config loader/validator
from src.core.logger import get_logger
from src.core.settings import get_config  # runtime defaults: max_iter, max_rpm, retries, etc.

logger = get_logger(__name__)


# ── factory ───────────────────────────────────────────────────────────────────


def create_skill_optimizer_agent() -> Agent:
    """Build a CrewAI Agent for skill selection, ordering, and categorization.

    Expects: agents.yaml has a 'skills_section_strategist' key with
             role, goal, backstory, and llm fields.
    Returns: a configured CrewAI Agent with no tools.
    Raises: RuntimeError if required config fields are missing.
    """
    ####################################################
    # STEP 1: LOAD CONFIG AND BUILD THE LLM INSTANCE
    ####################################################
    config = load_agent_config("skills_section_strategist")  # role/goal/backstory/llm from YAML
    llm_instance = LLM(model=config["llm"], temperature=config.get("temperature", 0.4))

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
        allow_delegation=False,  # this agent must not hand its selection task off to another agent
        tools=[],  # no tools — evidence checking runs code-owned, after the agent finishes
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
        "Skill Optimizer agent created",
        model=config["llm"],
        tools="none (audit is code-owned)",
    )
    return agent
