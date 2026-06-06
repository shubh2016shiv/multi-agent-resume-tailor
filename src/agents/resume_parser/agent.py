"""
Resume Extractor agent factory.

The agent receives converted resume Markdown (the orchestrator converts the PDF
first) and reasons over extraction quality before producing a structured Resume.

It calls check_resume_markdown_quality to inspect the Markdown, reads the report,
and only proceeds to extract if no BLOCKER-severity issues are found.

Tool assigned: check_resume_markdown_quality (defined in src/tools/agent_facing_tools.py)
Output contract: Resume (via Task output_pydantic=Resume)
"""

from crewai import Agent

from src.core.config import get_agents_config, get_config
from src.core.logger import get_logger
from src.tools.agent_facing_tools import (
    check_resume_markdown_quality,
    convert_resume_document_to_markdown,
    extract_structured_resume_from_markdown,
    redact_pii_from_resume_markdown,
)

logger = get_logger(__name__)


def create_resume_extractor_agent() -> Agent:
    """Create the Resume Content Extractor agent.

    Expects: agents.yaml to have a 'resume_content_extractor' key with
             role, goal, backstory, and llm fields.
    Returns: a configured CrewAI Agent with check_resume_markdown_quality assigned.
    Raises: RuntimeError if required config fields are missing.
    """
    agents_config = get_agents_config()
    config = agents_config.get("resume_content_extractor", {})

    required_fields = ["role", "goal", "backstory", "llm"]
    missing_fields = [f for f in required_fields if not config.get(f)]
    if missing_fields:
        raise RuntimeError(
            f"FATAL: Missing required field(s) in resume_content_extractor config: {missing_fields}\n"
            "Add all required fields to src/config/agents.yaml."
        )

    from crewai import LLM

    llm_instance = LLM(model=config["llm"])

    app_config = get_config()
    agent_defaults = app_config.llm.agent_defaults

    agent = Agent(
        role=config["role"],
        goal=config["goal"],
        backstory=config["backstory"],
        llm=llm_instance,
        temperature=config.get("temperature", 0.0),
        verbose=config.get("verbose", True),
        allow_delegation=False,
        # All four document ingestion tools. The agent reasons through them in order:
        # convert -> check quality -> redact PII -> extract.
        # Each tool is defined in src/tools/agent_facing_tools.py.
        tools=[
            convert_resume_document_to_markdown,
            check_resume_markdown_quality,
            redact_pii_from_resume_markdown,
            extract_structured_resume_from_markdown,
        ],
        max_retry_limit=agent_defaults.max_retry_limit,
        max_rpm=agent_defaults.max_rpm,
        max_iter=agent_defaults.max_iter,
        max_execution_time=agent_defaults.max_execution_time,
        respect_context_window=agent_defaults.respect_context_window,
    )

    logger.info(
        "Resume Extractor agent created",
        model=config["llm"],
        tool="Check Resume Markdown Quality",
    )
    return agent
