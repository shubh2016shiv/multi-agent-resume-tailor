"""
Four deterministic steps that turn a resume file into a validated Resume.

Every step wraps an existing engine from src/tools/document_ingestion/.
No LLM calls here except inside extract_resume (schema-constrained).

The agent in agent.py reasons over the quality step — it calls
check_resume_markdown_quality as a tool to decide whether the
conversion produced extractable text before committing to extraction.
"""

from src.data_models.resume import Resume
from src.tools.document_ingestion import (
    audit_extraction_quality,
    convert_document_to_markdown,
    extract_resume,
    redact_pii,
)
from src.tools.review_contract.review_models import ReviewResult


def convert_resume_pdf_to_markdown(file_path: str) -> str:
    """Convert a resume PDF/DOCX to Markdown text.

    Expects: a file path to a supported document (PDF, DOCX, MD).
    Returns: the document as a Markdown string.
    Raises: RuntimeError if the file format is unsupported or unreadable.
    """
    return convert_document_to_markdown(file_path)


def check_resume_markdown_quality(markdown: str) -> ReviewResult:
    """Audit whether the converted Markdown is extractable.

    Expects: the Markdown string produced by convert_resume_pdf_to_markdown.
    Returns: ReviewResult. A BLOCKER-severity comment means the conversion
             failed and the agent should stop and report the problem.
             An empty result means the text is usable.
    """
    return audit_extraction_quality(markdown)


def redact_pii_from_resume_markdown(markdown: str) -> tuple[str, dict[str, str]]:
    """Mask PII (name, email, phone) before sending text to the LLM.

    Expects: raw Markdown from conversion.
    Returns: (redacted_markdown, pii_mapping) where pii_mapping holds
             placeholder -> original value for later restoration at render time.
    """
    return redact_pii(markdown)


def extract_structured_resume(redacted_markdown: str) -> Resume:
    """Turn redacted Markdown into a validated Resume using a schema-constrained LLM call.

    Expects: Markdown with PII already masked by redact_pii_from_resume_markdown.
    Returns: a validated Resume. PII fields hold redaction placeholders.
    Raises: RuntimeError if the model cannot produce a schema-valid Resume.
    """
    return extract_resume(redacted_markdown)
