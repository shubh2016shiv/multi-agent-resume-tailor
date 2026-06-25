"""Unit tests for src/agents/resume_parser/agent.py

Contract under test: build_resume_ingestion_tools(enable_pii_redaction) returns
an ordered list of CrewAI tools. PII redaction tool is included only when the
flag is True. Order is always: convert → quality_check → [redact] → extract.
"""

from src.agents.resume_parser.agent import build_resume_ingestion_tools
from src.tools.agent_tools import (
    check_resume_markdown_quality,
    convert_resume_document_to_markdown,
    extract_structured_resume_from_markdown,
    redact_pii_from_resume_markdown,
)


class TestBuildResumeIngestionTools:
    """Tests for build_resume_ingestion_tools — assembles the document-ingestion tool list."""

    def test_build_resume_ingestion_tools_with_pii_disabled_returns_3_tools(self):
        """
        Contract: PII redaction disabled → 3 tools returned (convert, quality, extract).
        Expected value derived from docstring: "redact tool is dropped entirely when off."
        """
        tools = build_resume_ingestion_tools(enable_pii_redaction=False)

        assert len(tools) == 3

    def test_build_resume_ingestion_tools_with_pii_enabled_returns_4_tools(self):
        """
        Contract: PII redaction enabled → 4 tools returned (convert, quality, redact, extract).
        Expected value derived from docstring: "PII redaction is included only when the flag is on."
        """
        tools = build_resume_ingestion_tools(enable_pii_redaction=True)

        assert len(tools) == 4

    def test_build_resume_ingestion_tools_with_pii_disabled_does_not_include_redact_tool(self):
        """
        Contract: redact_pii_from_resume_markdown is absent when PII redaction is off.
        """
        tools = build_resume_ingestion_tools(enable_pii_redaction=False)

        assert redact_pii_from_resume_markdown not in tools

    def test_build_resume_ingestion_tools_with_pii_enabled_includes_redact_tool(self):
        """
        Contract: redact_pii_from_resume_markdown is present when PII redaction is on.
        """
        tools = build_resume_ingestion_tools(enable_pii_redaction=True)

        assert redact_pii_from_resume_markdown in tools

    def test_build_resume_ingestion_tools_with_pii_disabled_preserves_correct_order(self):
        """
        Contract: tool order without redaction is convert → quality → extract.
        Order matters because the agent orchestrates them in this sequence per the docstring.
        """
        tools = build_resume_ingestion_tools(enable_pii_redaction=False)

        assert tools[0] is convert_resume_document_to_markdown
        assert tools[1] is check_resume_markdown_quality
        assert tools[2] is extract_structured_resume_from_markdown

    def test_build_resume_ingestion_tools_with_pii_enabled_preserves_correct_order(self):
        """
        Contract: tool order with redaction is convert → quality → redact → extract.
        Redact must come after quality check and before extraction per the docstring.
        """
        tools = build_resume_ingestion_tools(enable_pii_redaction=True)

        assert tools[0] is convert_resume_document_to_markdown
        assert tools[1] is check_resume_markdown_quality
        assert tools[2] is redact_pii_from_resume_markdown
        assert tools[3] is extract_structured_resume_from_markdown
