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
    ####################################################
    # STEP 1: TURN THE INPUT PATH INTO A PATH OBJECT#
    ####################################################
    # This gives us one clear object for file existence checks,
    # suffix inspection, and logging.
    path = Path(file_path)

    ####################################################
    # STEP 2: FAIL FAST IF THE FILE DOES NOT EXIST#
    ####################################################
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    ####################################################
    # STEP 3: CHECK WHETHER THE FILE TYPE IS SUPPORTED#
    ####################################################
    # We normalize the suffix first so ".PDF" and ".pdf" are treated the same.
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format '{suffix}'. Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )

    ####################################################
    # STEP 4: READ PLAIN TEXT FILES DIRECTLY#
    ####################################################
    # Markdown and text files do not need document conversion.
    # Reading them directly is simpler and avoids unnecessary tooling.
    if suffix in {".md", ".txt"}:
        return path.read_text(encoding="utf-8")

    ####################################################
    # STEP 5: USE MARKITDOWN FOR BINARY DOCUMENT FORMATS#
    ####################################################
    # PDF, DOCX, PPTX, and XLSX need a conversion engine that can
    # extract their text layer into plain Markdown.
    try:
        result = markdown_conversion_engine.convert(str(path))
        logger.info(f"Converted {path.name} ({len(result.text_content)} chars)")
        return result.text_content
    except Exception as conversion_error:
        ####################################################
        # STEP 6: RAISE ONE CLEAN CONVERSION ERROR FOR CALLERS#
        ####################################################
        # We preserve the original error as the cause, but expose
        # a simpler message at this module boundary.
        raise OSError(
            f"Conversion failed for {path.name}: {conversion_error}"
        ) from conversion_error


def get_supported_formats() -> list[str]:
    """Return the file extensions this module can convert."""
    ####################################################
    # STEP 1: RETURN A SORTED COPY FOR STABLE DISPLAY#
    ####################################################
    return sorted(SUPPORTED_FORMATS)


def is_format_supported(file_path: str) -> bool:
    """Return True if the file extension is supported for conversion."""
    ####################################################
    # STEP 1: CHECK ONLY THE FILE EXTENSION, NOT THE FILE CONTENT#
    ####################################################
    # This helper answers a quick capability question.
    return Path(file_path).suffix.lower() in SUPPORTED_FORMATS


if __name__ == "__main__":
    # Run as a module so src.* imports resolve:
    #   python -m src.tools.engines.document_ingestion.document_conversion <path>
    import sys

    file_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "/home/shubham_singh/Downloads/Shubham_Resume_2026_April_version2.pdf"
    )
    print(f"Converting: {file_path}\n")
    markdown = convert_document_to_markdown(file_path)
    print(markdown)
