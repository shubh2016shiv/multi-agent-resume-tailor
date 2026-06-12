"""
Resume Extractor agent factory.

Builds a CrewAI Agent wired with 4 document-ingestion tools:
  1. convert_resume_document_to_markdown  — PDF/DOCX -> Markdown
  2. check_resume_markdown_quality        — audit extraction completeness
  3. redact_pii_from_resume_markdown      — mask PII before LLM sees it
  4. extract_structured_resume_from_markdown — schema-constrained Resume extraction

All tools are defined in src/tools/agent_facing_tools.py.
The agent orchestrates them in order: convert -> quality check -> redact -> extract.

Output contract: Resume (via Task output_pydantic=Resume).
"""

from crewai import LLM, Agent

from src.core.logger import get_logger
from src.core.settings import get_agents_config, get_config
from src.tools.agent_facing_tools import (
    check_resume_markdown_quality,
    convert_resume_document_to_markdown,
    extract_structured_resume_from_markdown,
    redact_pii_from_resume_markdown,
)

logger = get_logger(__name__)

# ── tool set ──────────────────────────────────────────────────────────────────

_RESUME_TOOLS: list = [
    convert_resume_document_to_markdown,
    check_resume_markdown_quality,
    redact_pii_from_resume_markdown,
    extract_structured_resume_from_markdown,
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


def create_resume_extractor_agent() -> Agent:
    """Build a CrewAI Agent with 4 document-ingestion tools.

    Expects: agents.yaml has a 'resume_content_extractor' key with
             role, goal, backstory, and llm fields.
    Returns: a configured CrewAI Agent.
    Raises: RuntimeError if required config fields are missing.
    """
    config = _load_agent_config("resume_content_extractor")
    llm_instance = LLM(model=config["llm"], temperature=config.get("temperature", 0.0))

    app_config = get_config()
    defaults = app_config.llm.agent_defaults

    agent = Agent(
        role=config["role"],
        goal=config["goal"],
        backstory=config["backstory"],
        llm=llm_instance,
        verbose=config.get("verbose", True),
        allow_delegation=False,
        tools=_RESUME_TOOLS,
        max_retry_limit=defaults.max_retry_limit,
        max_rpm=defaults.max_rpm,
        max_iter=defaults.max_iter,
        max_execution_time=defaults.max_execution_time,
        respect_context_window=defaults.respect_context_window,
    )

    tool_names = [t.name for t in _RESUME_TOOLS]
    logger.info(
        "Resume Extractor agent created",
        model=config["llm"],
        tools=tool_names,
        tool_count=len(tool_names),
    )
    return agent
