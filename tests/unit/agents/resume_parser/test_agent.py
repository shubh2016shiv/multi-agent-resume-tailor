"""Tests for the resume extractor's fixed ingestion tool sequence."""

from src.agents.resume_parser.agent import build_resume_ingestion_tools
from src.tools.agent_tools import (
    check_resume_markdown_quality,
    convert_resume_document_to_markdown,
    extract_structured_resume_from_markdown,
)


def test_resume_ingestion_tools_follow_required_order() -> None:
    """Conversion, quality checking, and extraction run in that order."""
    assert build_resume_ingestion_tools() == [
        convert_resume_document_to_markdown,
        check_resume_markdown_quality,
        extract_structured_resume_from_markdown,
    ]
