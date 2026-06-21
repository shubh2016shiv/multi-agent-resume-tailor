"""Agent-facing wrappers for resume ingestion steps."""

from crewai.tools import tool

from src.core.pii_mapping_store import assert_extraction_input_redacted, save_pii_mapping
from src.core.run_context import get_current_run_id
from src.core.settings import get_config
from src.tools.engines.document_ingestion import (
    audit_extraction_quality,
    convert_document_to_markdown,
    extract_resume,
    redact_pii,
)

from .resume_review_tools import render_review_result


@tool("Convert Resume Document to Markdown")
def convert_resume_document_to_markdown(file_path: str) -> str:
    """Convert a resume document into Markdown text."""
    return convert_document_to_markdown(file_path)


@tool("Redact PII from Resume Markdown")
def redact_pii_from_resume_markdown(markdown: str) -> str:
    """Mask personal data before any LLM sees the resume text."""
    if not get_config().feature_flags.enable_pii_redaction:
        return markdown

    redacted_markdown, placeholder_mapping = redact_pii(markdown)
    save_pii_mapping(get_current_run_id(), placeholder_mapping)
    return redacted_markdown


@tool("Extract Structured Resume from Markdown")
def extract_structured_resume_from_markdown(redacted_markdown: str) -> str:
    """Turn privacy-redacted Markdown into structured resume JSON."""
    if get_config().feature_flags.enable_pii_redaction:
        assert_extraction_input_redacted(get_current_run_id(), redacted_markdown)

    structured_resume = extract_resume(redacted_markdown)
    return structured_resume.model_dump_json()


@tool("Check Resume Markdown Quality")
def check_resume_markdown_quality(markdown: str) -> str:
    """Explain whether converted Markdown is clean enough for extraction."""
    return render_review_result(
        audit_extraction_quality(markdown),
        "Resume Markdown Quality",
    )
