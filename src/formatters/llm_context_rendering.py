"""Render formatter payloads into compact LLM context strings.

This module is called by the formatter entrypoints in `src/formatters/`.
Those formatter modules first choose the small set of fields one agent needs,
then pass that filtered dictionary here for rendering.

This module owns only shared rendering:
- `render_toon(...)` for compact runtime context
- `render_markdown(...)` for optional review/debug output
- `render_context_data(...)` as the shared entrypoint

Toy example:
    >>> render_context_data({"candidate_name": "Alice"}, format_type="toon")
    'candidate_name: Alice'
"""

import re
from typing import Any, Literal

OutputFormat = Literal["toon", "markdown"]

TOON_INDENT = "  "
TOON_SPECIAL_CHARACTER_PATTERN = re.compile(r'[{}[\]()<>:,\-"\']')


def toon_string_needs_quotes(text: str) -> bool:
    """Return whether a TOON string needs quotes to stay unambiguous."""
    if not text:
        return True
    if " " in text or "\n" in text or "\t" in text:
        return True
    if re.match(r"^[0-9]", text):
        return True
    return bool(TOON_SPECIAL_CHARACTER_PATTERN.search(text))


def quote_toon_string(text: str) -> str:
    """Wrap a TOON string in quotes only when the content needs it."""
    if not toon_string_needs_quotes(text):
        return text
    escaped_text = text.replace('"', '\\"')
    return f'"{escaped_text}"'


def render_toon_value(value: Any, indent_level: int = 0) -> str:
    """Render one Python value into TOON text."""
    indentation = TOON_INDENT * indent_level

    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, str):
        return quote_toon_string(value)
    if isinstance(value, dict):
        if not value:
            return "{}"
        dictionary_lines: list[str] = []
        for key, nested_value in value.items():
            dictionary_lines.append(render_toon_key_value_line(str(key), nested_value, indent_level))
        return "\n".join(dictionary_lines)
    if isinstance(value, list):
        if not value:
            return "[]"
        list_lines: list[str] = []
        for item in value:
            rendered_item = render_toon_value(item, indent_level + 1)
            list_lines.append(f"{indentation}- {rendered_item}")
        return "\n".join(list_lines)
    return quote_toon_string(str(value))


def render_toon_key_value_line(key: str, value: Any, indent_level: int) -> str:
    """Render one TOON key/value line with readable multiline layout."""
    indentation = TOON_INDENT * indent_level
    rendered_key = quote_toon_string(key)
    rendered_value = render_toon_value(value, indent_level + 1)

    if not isinstance(value, dict | list) or not value:
        return f"{indentation}{rendered_key}: {rendered_value}"
    return f"{indentation}{rendered_key}:\n{rendered_value}"


def render_toon(data: dict[str, Any]) -> str:
    """Render a filtered payload into TOON text."""
    ####################################################
    # STEP 1: FAIL FAST WHEN THE SHARED RENDERER RECEIVES THE WRONG TYPE#
    ####################################################
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict, got {type(data).__name__}")

    ####################################################
    # STEP 2: HANDLE THE EMPTY PAYLOAD EXPLICITLY#
    ####################################################
    if not data:
        return "{}"

    ####################################################
    # STEP 3: RENDER EACH TOP-LEVEL FIELD IN ORDER#
    ####################################################
    rendered_lines = [render_toon_key_value_line(key, value, indent_level=0) for key, value in data.items()]
    return "\n".join(rendered_lines)


def render_markdown_value(value: Any, indent_level: int = 0) -> str:
    """Render one Python value into Markdown text."""
    indentation = TOON_INDENT * indent_level

    if value is None:
        return "*None*"
    if isinstance(value, bool):
        return "**True**" if value else "**False**"
    if isinstance(value, int | float):
        return f"`{value}`"
    if isinstance(value, str):
        if "\n" in value:
            return f"```\n{value}\n```"
        return value
    if isinstance(value, dict):
        if not value:
            return "*Empty*"
        rendered_sections: list[str] = []
        for key, nested_value in value.items():
            heading_level = min(indent_level + 3, 6)
            heading_prefix = "#" * heading_level
            readable_key = key.replace("_", " ").title()
            if isinstance(nested_value, dict | list) and nested_value:
                rendered_sections.append(f"{indentation}{heading_prefix} {readable_key}")
                rendered_sections.append("")
                rendered_sections.append(render_markdown_value(nested_value, indent_level + 1))
            else:
                rendered_value = render_markdown_value(nested_value, indent_level)
                rendered_sections.append(f"{indentation}**{readable_key}**: {rendered_value}")
        return "\n".join(rendered_sections)
    if isinstance(value, list):
        if not value:
            return "*Empty list*"
        if all(not isinstance(item, dict | list) for item in value):
            return "\n".join(f"{indentation}- {item}" for item in value)
        rendered_items: list[str] = []
        for index, item in enumerate(value, start=1):
            rendered_item = render_markdown_value(item, indent_level + 1)
            rendered_items.append(f"{indentation}{index}. {rendered_item}")
        return "\n".join(rendered_items)
    return f"`{value}`"


def render_markdown(data: dict[str, Any], description: str = "") -> str:
    """Render a filtered payload into Markdown text."""
    ####################################################
    # STEP 1: FAIL FAST WHEN THE SHARED RENDERER RECEIVES THE WRONG TYPE#
    ####################################################
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict, got {type(data).__name__}")

    ####################################################
    # STEP 2: START WITH AN OPTIONAL TITLE FOR HUMAN REVIEW#
    ####################################################
    rendered_sections: list[str] = []
    if description:
        rendered_sections.append(f"## {description}")
        rendered_sections.append("")

    ####################################################
    # STEP 3: RENDER THE PAYLOAD BODY#
    ####################################################
    rendered_sections.append(render_markdown_value(data, indent_level=0))
    return "\n".join(rendered_sections)


def render_context_data(
    data: dict[str, Any],
    format_type: OutputFormat = "toon",
    description: str = "",
) -> str:
    """Render a filtered formatter payload in the requested output format."""
    ####################################################
    # STEP 1: SEND THE PAYLOAD TO THE REQUESTED RENDERER#
    ####################################################
    if format_type == "toon":
        return render_toon(data)
    if format_type == "markdown":
        return render_markdown(data, description)
    raise ValueError(f"Invalid format_type: {format_type}. Must be 'toon' or 'markdown'")
