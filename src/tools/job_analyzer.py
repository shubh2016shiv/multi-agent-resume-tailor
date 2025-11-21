"""
Job Analyzer Tool
-----------------

This module defines a CrewAI tool for extracting structured information from a
job description file. This tool is essential for the `Job Description Analyst`
agent to understand the requirements and context of a potential role.

TOOL DESIGN & PRINCIPLES:
- Consistent Two-Step Process: Like the resume parser, this tool first uses the
  `document_converter` to standardize the input (PDF, DOCX, etc.) into clean
  Markdown. This ensures reliability and consistency.
- LLM-Ready Output: The tool's job is to provide clean text. The agent's LLM is
  responsible for the complex task of understanding this text and structuring it
  into the `JobDescription` Pydantic model. This separation of concerns makes
  the system more robust and easier to debug.
- Professional Tool Definition: The function is decorated with `@tool` and
  features a comprehensive docstring, which CrewAI uses to inform the agent on
  how to use its capabilities.
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


@tool("Job Analyzer Tool")
def parse_job_description(file_path: str) -> str:
    """
    Parses a job description file and extracts its content into clean Markdown text.

    This tool is the first step in the job analysis workflow. It takes the path
    to a job description file (PDF, DOCX, MD, or TXT), reads it, and converts the
    content into a clean, standardized Markdown format.

    The resulting Markdown is then passed to an agent's LLM, which performs the
    actual structured extraction of job requirements, skills, company details, etc.,
    into the `JobDescription` data model.

    Args:
        file_path (str): The absolute or relative path to the job description document.
                         Supported formats: .pdf, .docx, .md, .txt.

    Returns:
        str: A string containing the clean Markdown content extracted from the
             job description. If an error occurs, it returns a descriptive error
             message string.
    
    Example:
        >>> job_markdown = parse_job_description("path/to/job_posting.docx")
        >>> if "ERROR:" not in job_markdown:
        ...     print("Successfully parsed job description.")
        ... else:
        ...     print(f"Failed to parse job description: {job_markdown}")
    """
    logger.info(f"Starting job description parsing for file: {file_path}")
    
    try:
        # Step 1: Use the document converter for robust file handling.
        markdown_content = convert_document_to_markdown(file_path)
        
        logger.info(f"Successfully converted job description to {len(markdown_content)} characters of Markdown.")
        
        # Step 2: Return the clean Markdown for the agent's LLM to analyze.
        return markdown_content

    except FileNotFoundError as e:
        error_message = f"ERROR: Job description file not found at path: {file_path}. {e}"
        logger.error(error_message)
        return error_message
        
    except ValueError as e:
        error_message = f"ERROR: Unsupported file format for job description. {e}"
        logger.error(error_message)
        return error_message
        
    except IOError as e:
        error_message = f"ERROR: Failed to read or process the job description file. It may be corrupted. {e}"
        logger.error(error_message)
        return error_message
        
    except Exception as e:
        # Catch-all for any other unexpected errors.
        error_message = f"ERROR: An unexpected error occurred while parsing the job description: {e}"
        logger.error(error_message, exc_info=True)
        return error_message
