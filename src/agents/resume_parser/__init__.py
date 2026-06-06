"""
Resume Extractor sub-module.

Public surface:
  create_resume_extractor_agent  -- build the CrewAI agent (use in orchestrator)
  pipeline.*                     -- the four typed steps (use directly or in tests)
"""

from src.agents.resume_parser.agent import create_resume_extractor_agent
from src.agents.resume_parser.pipeline import (
    check_resume_markdown_quality,
    convert_resume_pdf_to_markdown,
    extract_structured_resume,
    redact_pii_from_resume_markdown,
)

__all__ = [
    "create_resume_extractor_agent",
    "convert_resume_pdf_to_markdown",
    "check_resume_markdown_quality",
    "redact_pii_from_resume_markdown",
    "extract_structured_resume",
]
