"""
Resume Extractor agent factory.

Builds a CrewAI Agent wired with 4 document-ingestion tools:
  1. convert_resume_document_to_markdown  — PDF/DOCX -> Markdown
  2. check_resume_markdown_quality        — audit extraction completeness
  3. redact_pii_from_resume_markdown      — mask PII before LLM sees it (feature-flagged)
  4. extract_structured_resume_from_markdown — schema-constrained Resume extraction

The agent orchestrates them in order: convert -> quality check -> redact -> extract.

Output contract: Resume (via Task output_pydantic=Resume).
"""

from crewai import LLM, Agent

from src.agents.agent_config import load_agent_config
from src.core.logger import get_logger
from src.core.settings import get_config
from src.tools.agent_tools import (
    check_resume_markdown_quality,
    convert_resume_document_to_markdown,
    extract_structured_resume_from_markdown,
    redact_pii_from_resume_markdown,
)

logger = get_logger(__name__)


# ── tool set ──────────────────────────────────────────────────────────────────


def build_resume_ingestion_tools(enable_pii_redaction: bool) -> list:
    """Return the document-ingestion tool list for this agent.

    PII redaction is included only when the feature flag is on — when off, the
    redact tool is dropped entirely so the agent never has a redaction step to invoke.

    Expects: the feature_flags.enable_pii_redaction setting from app config.
    Returns: ordered list of CrewAI tools (convert -> quality -> [redact] -> extract).
    """
    tools = [convert_resume_document_to_markdown, check_resume_markdown_quality]
    if enable_pii_redaction:
        tools.append(redact_pii_from_resume_markdown)
    tools.append(extract_structured_resume_from_markdown)
    return tools


# ── factory ───────────────────────────────────────────────────────────────────


def create_resume_extractor_agent() -> Agent:
    """Build a CrewAI Agent with document-ingestion tools.

    Expects: agents.yaml has a 'resume_content_extractor' key with
             role, goal, backstory, and llm fields.
    Returns: a configured CrewAI Agent.
    Raises: RuntimeError if required config fields are missing.
    """
    ####################################################
    # STEP 1: LOAD CONFIG AND BUILD THE LLM INSTANCE
    ####################################################
    config = load_agent_config("resume_content_extractor")
    llm_instance = LLM(model=config["llm"], temperature=config.get("temperature", 0.0))

    ####################################################
    # STEP 2: ASSEMBLE TOOLS AND RUNTIME DEFAULTS
    ####################################################
    app_config = get_config()
    defaults = app_config.llm.agent_defaults
    resume_tools = build_resume_ingestion_tools(app_config.feature_flags.enable_pii_redaction)

    ####################################################
    # STEP 3: BUILD THE AGENT
    ####################################################
    agent = Agent(
        role=config["role"],
        goal=config["goal"],
        backstory=config["backstory"],
        llm=llm_instance,
        verbose=config.get("verbose", True),
        allow_delegation=False,
        tools=resume_tools,
        max_retry_limit=defaults.max_retry_limit,
        max_rpm=defaults.max_rpm,
        max_iter=defaults.max_iter,
        max_execution_time=defaults.max_execution_time,
        respect_context_window=defaults.respect_context_window,
    )

    ####################################################
    # STEP 4: LOG AND RETURN
    ####################################################
    tool_names = [tool.name for tool in resume_tools]
    logger.info(
        "Resume Extractor agent created",
        model=config["llm"],
        tools=tool_names,
        tool_count=len(tool_names),
    )
    return agent
