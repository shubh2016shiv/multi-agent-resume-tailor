"""
Document conversion: any supported file format → clean Markdown text.

Supported: .pdf, .docx, .pptx, .xlsx via markitdown; .md and .txt read directly.
"""

from pathlib import Path

from markitdown import MarkItDown

from src.core.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_FORMATS = frozenset({".pdf", ".docx", ".md", ".txt", ".pptx", ".xlsx"})

# Module-level instance — MarkItDown is stateless; one instance is sufficient.
markdown_conversion_engine = MarkItDown()


def convert_document_to_markdown(file_path: str) -> str:
    """Convert a supported document to clean Markdown text.

    Args:
        file_path: Absolute or relative path to the document.

    Returns:
        Markdown string extracted from the document.

    Raises:
        FileNotFoundError: File does not exist at the given path.
        ValueError: File format is not supported.
        OSError: markitdown conversion failed.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format '{suffix}'. Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )
    if suffix in {".md", ".txt"}:
        return path.read_text(encoding="utf-8")
    try:
        result = markdown_conversion_engine.convert(str(path))
        logger.info(f"Converted {path.name} ({len(result.text_content)} chars)")
        return result.text_content
    except Exception as conversion_error:
        raise OSError(
            f"Conversion failed for {path.name}: {conversion_error}"
        ) from conversion_error


def get_supported_formats() -> list[str]:
    """Return the file extensions this module can convert."""
    return sorted(SUPPORTED_FORMATS)


def is_format_supported(file_path: str) -> bool:
    """Return True if the file extension is supported for conversion."""
    return Path(file_path).suffix.lower() in SUPPORTED_FORMATS


if __name__ == "__main__":
    # Run as a module so src.* imports resolve:
    #   python -m src.tools.document_ingestion.document_converter <path>
    import sys

    file_path = sys.argv[1] if len(sys.argv) > 1 else "/home/shubham_singh/Downloads/Shubham_Resume_2026_April_version2.pdf"
    print(f"Converting: {file_path}\n")
    markdown = convert_document_to_markdown(file_path)
    print(markdown)
