"""
Resume Parser Tool
--------------------

This module defines a CrewAI tool for extracting structured information from a
resume file. It serves as the primary mechanism for the `Resume Content Extractor`
agent to ingest and understand a candidate's career history.

TOOL DESIGN & PRINCIPLES:
- Two-Step Extraction Process:
  1. Document Conversion: The tool first uses the `document_converter` utility
     to convert any supported file format (PDF, DOCX, etc.) into clean,
     standardized Markdown. This step handles the complexity of different
     file types and ensures the next step has a consistent input format.
  2. LLM-Ready Output: The tool's primary output is this clean Markdown. It
     does NOT perform the structured extraction itself. Instead, it prepares
     the data for the agent's LLM, which is much more powerful and flexible at
     parsing natural language and mapping it to a structured schema (our
     `Resume` Pydantic model).
- Simplicity & Focus: The tool's responsibility is narrow and well-defined:
  get clean text from a file. This makes it simple, reliable, and easy to test.
- Professional & Documented: The tool is decorated with `@tool` and has a
  comprehensive docstring that explains its purpose, usage, and what it returns.
  This documentation is automatically used by the agent to understand how to
  use the tool.
"""

from crewai.tools import tool

# Handle imports for both package usage and direct script execution
try:
    from src.core.logger import get_logger
    from src.tools.document_converter import convert_document_to_markdown
except ImportError:
    # Fallback for when running this file directly for testing
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.core.logger import get_logger
    from src.tools.document_converter import convert_document_to_markdown

logger = get_logger(__name__)


def _parse_resume_impl(file_path: str) -> str:
    """
    Core implementation of resume parsing logic.
    
    This function contains the actual parsing logic and can be tested independently.
    The @tool decorator wraps this function for CrewAI agent usage.
    
    Args:
        file_path: Path to the resume file.
    
    Returns:
        Clean Markdown content or error message string.
    """
    logger.info(f"Starting resume parsing for file: {file_path}")
    
    try:
        # Step 1: Use the document converter to get clean Markdown.
        # This handles all the complexity of different file formats and parsing.
        markdown_content = convert_document_to_markdown(file_path)
        
        logger.info(f"Successfully converted resume to {len(markdown_content)} characters of Markdown.")
        
        # Step 2: Return the clean Markdown.
        # The agent will now take this text and use its LLM to populate the
        # `Resume` Pydantic model.
        return markdown_content

    except FileNotFoundError as e:
        error_message = f"ERROR: Resume file not found at path: {file_path}. {e}"
        logger.error(error_message)
        return error_message
        
    except ValueError as e:
        error_msg = str(e)
        # Handle Unicode encoding issues in error messages for Windows console
        try:
            error_msg = error_msg.encode('utf-8', errors='replace').decode('utf-8')
        except Exception:
            error_msg = repr(e)
        error_message = f"ERROR: Unsupported file format for resume. {error_msg}"
        logger.error(error_message)
        return error_message
        
    except IOError as e:
        error_message = f"ERROR: Failed to read or process the resume file. The file might be corrupted. {e}"
        logger.error(error_message)
        return error_message
        
    except Exception as e:
        # Catch-all for any other unexpected errors during parsing.
        error_message = f"ERROR: An unexpected error occurred while parsing the resume: {e}"
        logger.error(error_message, exc_info=True)
        return error_message


@tool("Resume Parser Tool")
def parse_resume(file_path: str) -> str:
    """
    Parses a resume file and extracts its content into clean Markdown text.

    This tool is the first step in the resume analysis process. It takes the path
    to a resume file (PDF, DOCX, MD, or TXT), reads it, and converts its content
    into a clean, standardized Markdown format.

    The resulting Markdown is then passed to an agent's LLM, which performs the
    actual structured extraction into the `Resume` data model. This approach is
    highly robust as it separates the file parsing complexity from the language
    understanding task.

    Args:
        file_path (str): The absolute or relative path to the resume document.
                         Supported formats: .pdf, .docx, .md, .txt.

    Returns:
        str: A string containing the clean Markdown content extracted from the
             resume. If an error occurs (e.g., file not found, unsupported format),
             it returns a descriptive error message string.
    
    Example:
        >>> resume_markdown = parse_resume("path/to/my_resume.pdf")
        >>> if "ERROR:" not in resume_markdown:
        ...     print("Successfully parsed resume.")
        ... else:
        ...     print(f"Failed to parse resume: {resume_markdown}")
    """
    return _parse_resume_impl(file_path)
