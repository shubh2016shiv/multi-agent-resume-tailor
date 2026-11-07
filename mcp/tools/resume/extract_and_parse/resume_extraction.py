"""Tool: turn an uploaded resume file (PDF or DOCX) into Markdown text."""

import base64
import binascii
import io
from pathlib import Path

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from markitdown import MarkItDown, StreamInfo

ALLOWED_EXTENSIONS = {".pdf", ".docx"}
MAX_RESUME_FILE_BYTES = 10 * 1024 * 1024

resume_to_markdown_converter = MarkItDown()


def extract_resume_text(resume_file_base64: str, resume_filename: str) -> str:
    """Convert a resume file to Markdown text.

    Expects the file bytes as a base64 string and the original filename; the
    extension (.pdf or .docx) decides how the file is read. Returns the
    Markdown text. Raises ToolError for a bad extension, invalid base64, an
    oversized file, or a file with no extractable text (e.g. a scanned PDF).
    """
    resume_file_extension = Path(resume_filename).suffix.lower()
    if resume_file_extension not in ALLOWED_EXTENSIONS:
        raise ToolError(f"Unsupported file type '{resume_file_extension}'. Use .pdf or .docx")

    try:
        resume_file_bytes = base64.b64decode(resume_file_base64, validate=True)
    except (binascii.Error, ValueError) as base64_decode_error:
        raise ToolError("resume_file_base64 is not valid base64") from base64_decode_error

    if len(resume_file_bytes) > MAX_RESUME_FILE_BYTES:
        raise ToolError(f"File is larger than {MAX_RESUME_FILE_BYTES // (1024 * 1024)} MB")

    markdown_conversion_result = resume_to_markdown_converter.convert_stream(
        io.BytesIO(resume_file_bytes), stream_info=StreamInfo(extension=resume_file_extension)
    )
    resume_markdown = markdown_conversion_result.text_content.strip()
    if not resume_markdown:
        raise ToolError("No text could be extracted. The file may be a scanned image")
    return resume_markdown


def register_resume_extraction_tool(mcp: FastMCP) -> None:
    """Add this module's tool to the server."""
    mcp.tool(annotations={"readOnlyHint": True})(extract_resume_text)
