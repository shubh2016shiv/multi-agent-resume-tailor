"""Agent-facing wrappers for resume ingestion steps."""

from crewai.tools import tool

from src.tools.engines.document_ingestion import (
    audit_extraction_quality,
    convert_document_to_markdown,
    extract_resume,
)

from .resume_review_tools import render_review_result


@tool("Convert Resume Document to Markdown")
def convert_resume_document_to_markdown(file_path: str) -> str:
    """Convert a resume document into Markdown text."""
    return convert_document_to_markdown(file_path)


@tool("Extract Structured Resume from Markdown")
def extract_structured_resume_from_markdown(markdown: str) -> str:
    """Turn resume Markdown into structured resume JSON."""
    structured_resume = extract_resume(markdown)
    return structured_resume.model_dump_json()


@tool("Check Resume Markdown Quality")
def check_resume_markdown_quality(markdown: str) -> str:
    """Explain whether converted Markdown is clean enough for extraction."""
    return render_review_result(
        audit_extraction_quality(markdown),
        "Resume Markdown Quality",
    )
