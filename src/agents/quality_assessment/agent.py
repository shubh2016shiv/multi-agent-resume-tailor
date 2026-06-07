"""
Quality Assessment agent factory.

Builds a CrewAI Agent that audits the optimized resume against the original and the
job, then produces a QualityReport. The agent is the final gatekeeper before render.

It is wired with three read-only audit tools -- one per QualityReport dimension:
  - audit_truthfulness        -> accuracy   (original vs revised: invented/inflated facts)
  - validate_ats_compliance   -> ats        (formatting + standard section headers)
  - analyze_jd_keyword_coverage -> relevance (job keyword coverage + density)
The agent calls these WHILE reasoning so its dimension scores are grounded in real
checks rather than unaided judgment. The tools never mutate the resume.

The pass/fail decision is NOT the agent's to make: it produces the dimension scores,
and code (engines.apply_quality_gate) sets passed_quality_threshold deterministically
from overall_quality_score. That code-owned boolean is what gates PDF rendering.

Output contract: QualityReport (src/data_models/evaluation.py), via Task output_pydantic.
"""

from crewai import LLM, Agent

from src.core.config import get_agents_config, get_config
from src.core.logger import get_logger
from src.tools.agent_facing_tools import (
    analyze_jd_keyword_coverage,
    audit_truthfulness,
    validate_ats_compliance,
)

logger = get_logger(__name__)

# ── tool set ──────────────────────────────────────────────────────────────────

_QA_TOOLS = [
    audit_truthfulness,
    validate_ats_compliance,
    analyze_jd_keyword_coverage,
]


# ── config ────────────────────────────────────────────────────────────────────


def _load_agent_config(name: str) -> dict:
    """Load and validate an agent config block from agents.yaml.

    Expects: agents.yaml has a key matching `name` with role, goal, backstory, llm.
    Returns: the config dict.
    Raises: RuntimeError if any required field is missing.
    """
    agents_config = get_agents_config()
    config = agents_config.get(name, {})

    required = ["role", "goal", "backstory", "llm"]
    missing = [f for f in required if not config.get(f)]
    if missing:
        raise RuntimeError(
            f"FATAL: Missing required field(s) in '{name}' agent config: {missing}\n"
            f"Add all required fields to src/config/agents.yaml."
        )
    return config


# ── factory ───────────────────────────────────────────────────────────────────


def create_quality_assessment_agent() -> Agent:
    """Build a CrewAI Agent for final quality assessment of the optimized resume.

    Expects: agents.yaml has a 'quality_assurance_reviewer' key with
             role, goal, backstory, and llm fields.
    Returns: a configured CrewAI Agent wired with the three QA audit tools.
    Raises: RuntimeError if required config fields are missing.
    """
    config = _load_agent_config("quality_assurance_reviewer")
    llm_instance = LLM(model=config["llm"])

    app_config = get_config()
    defaults = app_config.llm.agent_defaults

    agent = Agent(
        role=config["role"],
        goal=config["goal"],
        backstory=config["backstory"],
        llm=llm_instance,
        temperature=config.get("temperature", 0.2),
        verbose=config.get("verbose", True),
        allow_delegation=False,
        tools=_QA_TOOLS,
        max_retry_limit=defaults.max_retry_limit,
        max_rpm=defaults.max_rpm,
        max_iter=defaults.max_iter,
        max_execution_time=defaults.max_execution_time,
        respect_context_window=defaults.respect_context_window,
    )

    tool_names = [t.name for t in _QA_TOOLS]
    logger.info(
        "Quality Assessment agent created",
        model=config["llm"],
        tools=tool_names,
    )
    return agent
