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

from crewai import LLM, Agent  # LLM wraps the configured model; Agent is the CrewAI persona

from src.agents.agent_config import load_agent_config  # shared YAML config loader/validator
from src.core.logger import get_logger
from src.core.settings import get_config  # runtime defaults + the PII-redaction feature flag
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
    # Order matters: the agent must run these steps in sequence, so the list
    # order below is also the order the agent is instructed to call them in.
    tools = [convert_resume_document_to_markdown, check_resume_markdown_quality]
    if enable_pii_redaction:
        tools.append(redact_pii_from_resume_markdown)  # only when the feature flag is on
    tools.append(extract_structured_resume_from_markdown)  # always runs last
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
    config = load_agent_config("resume_content_extractor")  # role/goal/backstory/llm from YAML
    llm_instance = LLM(model=config["llm"], temperature=config.get("temperature", 0.0))

    ####################################################
    # STEP 2: ASSEMBLE TOOLS AND RUNTIME DEFAULTS
    ####################################################
    app_config = get_config()
    defaults = app_config.llm.agent_defaults  # shared retry/rate-limit/timeout settings
    # tool list changes shape based on whether PII redaction is enabled
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
        allow_delegation=False,  # this agent must not hand its extraction task off to another agent
        tools=resume_tools,  # convert -> quality check -> [redact] -> extract, in order
        max_retry_limit=defaults.max_retry_limit,  # retries on a failed/malformed LLM call
        max_rpm=defaults.max_rpm,  # caps requests-per-minute to this agent's LLM
        max_iter=defaults.max_iter,  # caps reasoning/tool-call loops before forcing an answer
        max_execution_time=defaults.max_execution_time,  # hard wall-clock timeout for one run
        respect_context_window=defaults.respect_context_window,  # auto-trim context instead of erroring
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
