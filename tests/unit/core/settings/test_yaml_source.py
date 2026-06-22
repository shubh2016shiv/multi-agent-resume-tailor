"""Unit tests for src/core/settings/yaml_source.py."""

from pathlib import Path

import pytest

from src.core.settings.exceptions import ConfigurationError
from src.core.settings.yaml_source import read_yaml_mapping


class TestReadYamlMapping:
    """Tests for low-level YAML mapping loading."""

    def test_read_yaml_mapping_returns_empty_mapping_when_file_is_absent(self, tmp_path: Path):
        """Contract: missing YAML files are treated as no overrides."""
        result = read_yaml_mapping(tmp_path / "missing.yaml", "settings")

        assert result == {}

    def test_read_yaml_mapping_returns_a_mapping_when_yaml_is_valid(self, tmp_path: Path):
        """Contract: valid YAML mappings load into plain Python dictionaries."""
        yaml_path = tmp_path / "settings.yaml"
        yaml_path.write_text("llm:\n  provider: openai\n", encoding="utf-8")

        result = read_yaml_mapping(yaml_path, "settings")

        assert result == {"llm": {"provider": "openai"}}

    def test_read_yaml_mapping_raises_for_non_mapping_yaml(self, tmp_path: Path):
        """Contract: top-level YAML lists are rejected explicitly."""
        yaml_path = tmp_path / "settings.yaml"
        yaml_path.write_text("- one\n- two\n", encoding="utf-8")

        with pytest.raises(ConfigurationError, match="must be a YAML mapping"):
            read_yaml_mapping(yaml_path, "settings")

    def test_read_yaml_mapping_raises_for_invalid_yaml(self, tmp_path: Path):
        """Contract: invalid YAML raises ConfigurationError with file context."""
        yaml_path = tmp_path / "settings.yaml"
        yaml_path.write_text("llm: [broken\n", encoding="utf-8")

        with pytest.raises(ConfigurationError, match="Could not load settings configuration"):
            read_yaml_mapping(yaml_path, "settings")
