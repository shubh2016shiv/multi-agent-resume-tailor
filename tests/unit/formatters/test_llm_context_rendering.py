"""Unit tests for src/formatters/llm_context_rendering.py.

Contract under test: the shared renderer turns filtered dictionaries into TOON
or Markdown output, and rejects unsupported output formats.
"""

import pytest

from src.formatters.llm_context_rendering import render_context_data


class TestRenderContextData:
    """Tests for the shared formatter renderer."""

    def test_render_context_data_with_default_format_returns_toon_text(self):
        """
        Contract: TOON is the default output format.
        Expected value comes from the renderer contract, not by restating the implementation.
        """
        payload = {"candidate_name": "Jane Doe", "skills": ["Python", "SQL"]}

        result = render_context_data(payload)

        assert result == 'candidate_name: "Jane Doe"\nskills:\n  - Python\n  - SQL'

    def test_render_context_data_with_markdown_format_returns_readable_sections(self):
        """Contract: Markdown output includes the optional description heading and readable fields."""
        payload = {"candidate_name": "Jane Doe", "skills": ["Python", "SQL"]}

        result = render_context_data(
            payload,
            format_type="markdown",
            description="Candidate Snapshot",
        )

        assert "## Candidate Snapshot" in result
        assert "**Candidate Name**: Jane Doe" in result
        assert "- Python" in result
        assert "- SQL" in result

    def test_render_context_data_with_invalid_format_raises_value_error(self):
        """Contract: unsupported format names are rejected explicitly."""
        payload = {"candidate_name": "Jane Doe"}

        with pytest.raises(ValueError, match="Invalid format_type"):
            render_context_data(payload, format_type="json")  # type: ignore[arg-type]
