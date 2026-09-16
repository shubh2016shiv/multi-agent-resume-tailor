"""Resume extractor agent factory."""

from crewai import Agent

from src.agents.agent_config import load_agent_config
from src.core.llm_factory import build_llm
from src.core.logger import get_logger
from src.core.settings import get_config
from src.tools.agent_tools import (
    check_resume_markdown_quality,
    convert_resume_document_to_markdown,
    extract_structured_resume_from_markdown,
)

logger = get_logger(__name__)


def build_resume_ingestion_tools() -> list:
    """Return tools in the order required to ingest a resume."""
    return [
        convert_resume_document_to_markdown,
        check_resume_markdown_quality,
        extract_structured_resume_from_markdown,
    ]


def create_resume_extractor_agent() -> Agent:
    """Build the source-faithful resume extraction agent."""
    config = load_agent_config("resume_content_extractor")
    defaults = get_config().llm.agent_defaults
    resume_tools = build_resume_ingestion_tools()

    agent = Agent(
        role=config["role"],
        goal=config["goal"],
        backstory=config["backstory"],
        llm=build_llm(config),
        verbose=config.get("verbose", True),
        allow_delegation=False,
        tools=resume_tools,
        max_retry_limit=defaults.max_retry_limit,
        max_rpm=defaults.max_rpm,
        max_iter=defaults.max_iter,
        max_execution_time=defaults.max_execution_time,
        respect_context_window=defaults.respect_context_window,
    )

    tool_names = [tool.name for tool in resume_tools]
    logger.info(
        "Resume Extractor agent created",
        model=config["llm"],
        tools=tool_names,
        tool_count=len(tool_names),
    )
    return agent
