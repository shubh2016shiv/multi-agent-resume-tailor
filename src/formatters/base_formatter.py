"""
Base Formatter Module
=====================

Provides core formatting utilities including TOON (Token-Oriented Object Notation)
and Markdown conversion functions. Supports multiple output formats:

- TOON: Reduces token usage by ~46% compared to JSON
- Markdown: Human-readable format with headings, sections, and tables

TOON Format Rules:
- Objects: Use indentation, no braces
- Arrays: Use `-` prefix, no brackets
- Strings: No quotes unless needed (spaces/special chars)
- Numbers: Direct representation
- Nested structures: Indent with 2 spaces per level

Markdown Format Rules:
- Use headings (##, ###) for sections
- Use tables for structured data
- Use lists for arrays
- Use code blocks for nested structures
- Include descriptions at the top of sections
"""

import json
import re
from typing import Any, Literal

from src.core.logger import get_logger

logger = get_logger(__name__)

# Format type constants
FormatType = Literal["toon", "markdown"]


def _needs_quotes(value: str) -> bool:
    """
    Determine if a string value needs quotes in TOON format.

    Args:
        value: String value to check

    Returns:
        True if quotes are needed (contains spaces, special chars, or is empty)
    """
    if not value:
        return True
    # Check for spaces, special characters, or starts with number
    if " " in value or "\n" in value or "\t" in value:
        return True
    if re.match(r"^[0-9]", value):
        return True
    # Check for special characters that might confuse parsing
    if re.search(r"[{}[\]()<>:,\-]", value):
        return True
    return False


def _to_toon_value(value: Any, indent_level: int = 0) -> str:
    """
    Convert a Python value to TOON format string.

    Args:
        value: Python value to convert (dict, list, str, int, float, bool, None)
        indent_level: Current indentation level (for nested structures)

    Returns:
        TOON-formatted string representation
    """
    indent = "  " * indent_level

    if value is None:
        return "null"
    elif isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, int | float):
        return str(value)
    elif isinstance(value, str):
        if _needs_quotes(value):
            # Escape quotes in the string
            escaped = value.replace('"', '\\"')
            return f'"{escaped}"'
        return value
    elif isinstance(value, dict):
        if not value:
            return "{}"
        lines = []
        for key, val in value.items():
            key_str = key if not _needs_quotes(key) else f'"{key}"'
            val_str = _to_toon_value(val, indent_level + 1)
            lines.append(f"{indent}  {key_str}: {val_str}")
        return "\n".join(lines)
    elif isinstance(value, list):
        if not value:
            return "[]"
        lines = []
        for item in value:
            item_str = _to_toon_value(item, indent_level + 1)
            # For list items, indent the dash
            lines.append(f"{indent}- {item_str}")
        return "\n".join(lines)
    else:
        # Fallback: convert to string
        str_value = str(value)
        if _needs_quotes(str_value):
            return f'"{str_value}"'
        return str_value


def to_toon(data: dict[str, Any]) -> str:
    """
    Convert a Python dictionary to TOON (Token-Oriented Object Notation) format.

    TOON format reduces token usage by ~46% compared to JSON by:
    - Using indentation instead of braces/brackets
    - Removing quotes where not needed
    - Minimizing punctuation overhead

    Args:
        data: Python dictionary to convert

    Returns:
        TOON-formatted string

    Example:
        >>> data = {"name": "Alice", "age": 30, "city": "New York"}
        >>> to_toon(data)
        'name: Alice\\nage: 30\\ncity: "New York"'

    Raises:
        ValueError: If data cannot be converted to TOON format
    """
    try:
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict, got {type(data).__name__}")

        if not data:
            return "{}"

        result_lines = []
        for key, value in data.items():
            key_str = key if not _needs_quotes(key) else f'"{key}"'
            value_str = _to_toon_value(value, indent_level=0)
            result_lines.append(f"{key_str}: {value_str}")

        return "\n".join(result_lines)

    except Exception as e:
        logger.error(f"Failed to convert data to TOON format: {e}", exc_info=True)
        # Fallback to JSON if TOON conversion fails
        logger.warning("Falling back to JSON format")
        return json.dumps(data, indent=2)


def filter_nested_dict(data: dict[str, Any], allowed_fields: set[str]) -> dict[str, Any]:
    """
    Recursively filter a dictionary to only include specified fields.

    This function performs a deep filter, keeping only the fields specified in
    allowed_fields at each level of nesting.

    Args:
        data: Dictionary to filter
        allowed_fields: Set of field names to keep

    Returns:
        Filtered dictionary containing only allowed fields

    Example:
        >>> data = {"name": "Alice", "age": 30, "city": "New York", "metadata": {"id": 1}}
        >>> filter_nested_dict(data, {"name", "age"})
        {"name": "Alice", "age": 30}
    """
    if not isinstance(data, dict):
        return data

    filtered = {}
    for key, value in data.items():
        if key in allowed_fields:
            if isinstance(value, dict):
                # Recursively filter nested dictionaries
                filtered[key] = filter_nested_dict(value, allowed_fields)
            elif isinstance(value, list):
                # Filter list items if they are dictionaries
                filtered[key] = [
                    filter_nested_dict(item, allowed_fields) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                filtered[key] = value

    return filtered


def _to_markdown_value(value: Any, key: str = "", indent_level: int = 0) -> str:
    """
    Convert a Python value to Markdown format string.

    Args:
        value: Python value to convert (dict, list, str, int, float, bool, None)
        key: Key name (used for section headings)
        indent_level: Current indentation level (for nested structures)

    Returns:
        Markdown-formatted string representation
    """
    indent = "  " * indent_level

    if value is None:
        return "*None*"
    elif isinstance(value, bool):
        return "**True**" if value else "**False**"
    elif isinstance(value, int | float):
        return f"`{value}`"
    elif isinstance(value, str):
        # For multiline strings, use code blocks
        if "\n" in value:
            return f"```\n{value}\n```"
        return value
    elif isinstance(value, dict):
        if not value:
            return "*Empty*"
        lines = []
        # Use table format for dicts with simple values, sections for complex
        simple_values = all(
            not isinstance(v, dict | list) or (isinstance(v, list) and len(v) == 0)
            for v in value.values()
        )
        if simple_values and len(value) <= 5:
            # Use table format for simple key-value pairs
            lines.append(f"{indent}| Field | Value |")
            lines.append(f"{indent}|-------|-------|")
            for k, v in value.items():
                v_str = str(v) if not isinstance(v, str) else v
                lines.append(f"{indent}| {k} | {v_str} |")
        else:
            # Use sections for complex structures
            for k, v in value.items():
                heading_level = min(indent_level + 2, 6)  # Max heading level is 6
                heading_prefix = "#" * heading_level
                if isinstance(v, dict | list) and v:
                    lines.append(f"{indent}{heading_prefix} {k.replace('_', ' ').title()}")
                    lines.append("")
                    lines.append(_to_markdown_value(v, k, indent_level + 1))
                else:
                    v_str = _to_markdown_value(v, k, indent_level)
                    lines.append(f"{indent}**{k.replace('_', ' ').title()}**: {v_str}")
        return "\n".join(lines)
    elif isinstance(value, list):
        if not value:
            return "*Empty list*"
        lines = []
        # Check if list items are simple (strings, numbers) or complex (dicts)
        if all(not isinstance(item, dict | list) for item in value):
            # Simple list - use bullet points
            for item in value:
                item_str = str(item) if not isinstance(item, str) else item
                lines.append(f"{indent}- {item_str}")
        else:
            # Complex list - use numbered list with details
            for idx, item in enumerate(value, 1):
                item_str = _to_markdown_value(item, "", indent_level + 1)
                # For complex items, add proper indentation
                if "\n" in item_str:
                    # Indent all lines except the first
                    item_lines = item_str.split("\n")
                    indented_lines = [item_lines[0]]
                    for line in item_lines[1:]:
                        indented_lines.append(f"{indent}   {line}" if line else line)
                    item_str = "\n".join(indented_lines)
                lines.append(f"{indent}{idx}. {item_str}")
        return "\n".join(lines)
    else:
        return f"`{str(value)}`"


def to_markdown(data: dict[str, Any], description: str = "") -> str:
    """
    Convert a Python dictionary to Markdown format.

    Markdown format provides human-readable structure with:
    - Section headings (##, ###)
    - Tables for structured data
    - Lists for arrays
    - Code blocks for nested structures

    Args:
        data: Python dictionary to convert
        description: Optional description text to include at the top

    Returns:
        Markdown-formatted string

    Example:
        >>> data = {"name": "Alice", "age": 30, "experience": [{"role": "Engineer"}]}
        >>> to_markdown(data, "Candidate Information")
        '## Candidate Information\\n\\n**Name**: Alice\\n**Age**: `30`\\n...'
    """
    try:
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict, got {type(data).__name__}")

        lines = []

        # Add description as heading if provided
        if description:
            lines.append(f"## {description}")
            lines.append("")

        # Convert the data
        content = _to_markdown_value(data, "", indent_level=0)
        lines.append(content)

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"Failed to convert data to Markdown format: {e}", exc_info=True)
        # Fallback to JSON if Markdown conversion fails
        logger.warning("Falling back to JSON format")
        return json.dumps(data, indent=2)


def format_data(
    data: dict[str, Any], format_type: FormatType = "toon", description: str = ""
) -> str:
    """
    Format data in the specified format (TOON or Markdown).

    Args:
        data: Python dictionary to format
        format_type: Format to use ("toon" or "markdown")
        description: Optional description for markdown format

    Returns:
        Formatted string in the requested format

    Raises:
        ValueError: If format_type is not "toon" or "markdown"
    """
    if format_type == "toon":
        return to_toon(data)
    elif format_type == "markdown":
        return to_markdown(data, description)
    else:
        raise ValueError(f"Invalid format_type: {format_type}. Must be 'toon' or 'markdown'")


def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a text string.

    This is a rough estimation using a simple heuristic:
    - Average English word: ~1.3 tokens
    - Punctuation and whitespace: ~0.5 tokens per character
    - More accurate for longer texts

    Args:
        text: Text string to estimate tokens for

    Returns:
        Estimated number of tokens

    Note:
        This is an approximation. For accurate token counting, use the actual
        tokenizer from the LLM provider (e.g., tiktoken for OpenAI).
    """
    if not text:
        return 0

    # Rough estimation: ~4 characters per token on average
    # This is a conservative estimate that works reasonably well for English text
    char_count = len(text)
    estimated_tokens = char_count // 4

    # Add some overhead for special characters and formatting
    special_chars = len(re.findall(r'[{}[\]()<>:,\-"\']', text))
    estimated_tokens += special_chars // 2

    return max(estimated_tokens, 1)  # At least 1 token
