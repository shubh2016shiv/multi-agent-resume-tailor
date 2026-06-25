"""Unit tests for src/core/settings/agent_task_catalog.py."""

from pathlib import Path

import pytest

from src.core.settings.agent_task_catalog import get_agents_config, get_tasks_config


class TestGetAgentsConfig:
    """Tests for merged agent catalog loading."""

    def test_get_agents_config_merges_sorted_yaml_files(self, monkeypatch, tmp_path: Path):
        """Contract: every agent YAML file contributes its top-level keys to one catalog."""
        first_file = tmp_path / "a_agent.yaml"
        second_file = tmp_path / "b_agent.yaml"
        first_file.write_text("first_agent:\n  role: first\n", encoding="utf-8")
        second_file.write_text("second_agent:\n  role: second\n", encoding="utf-8")

        monkeypatch.setattr("src.core.settings.agent_task_catalog.AGENTS_CONFIG_DIR", tmp_path)
        get_agents_config.cache_clear()

        result = get_agents_config()

        assert result == {
            "first_agent": {"role": "first"},
            "second_agent": {"role": "second"},
        }
        get_agents_config.cache_clear()

    def test_get_agents_config_raises_when_directory_is_missing(self, monkeypatch, tmp_path: Path):
        """Contract: a missing agent catalog directory is a hard error."""
        missing_dir = tmp_path / "missing_agents"
        monkeypatch.setattr("src.core.settings.agent_task_catalog.AGENTS_CONFIG_DIR", missing_dir)
        get_agents_config.cache_clear()

        with pytest.raises(FileNotFoundError, match="Agent configuration directory not found"):
            get_agents_config()

        get_agents_config.cache_clear()


class TestGetTasksConfig:
    """Tests for merged task catalog loading."""

    def test_get_tasks_config_merges_sorted_yaml_files(self, monkeypatch, tmp_path: Path):
        """Contract: every task YAML file contributes its top-level keys to one catalog."""
        first_file = tmp_path / "a_task.yaml"
        second_file = tmp_path / "b_task.yaml"
        first_file.write_text("first_task:\n  description: first\n", encoding="utf-8")
        second_file.write_text("second_task:\n  description: second\n", encoding="utf-8")

        monkeypatch.setattr("src.core.settings.agent_task_catalog.TASKS_CONFIG_DIR", tmp_path)
        get_tasks_config.cache_clear()

        result = get_tasks_config()

        assert result == {
            "first_task": {"description": "first"},
            "second_task": {"description": "second"},
        }
        get_tasks_config.cache_clear()

    def test_get_tasks_config_raises_when_directory_is_missing(self, monkeypatch, tmp_path: Path):
        """Contract: a missing task catalog directory is a hard error."""
        missing_dir = tmp_path / "missing_tasks"
        monkeypatch.setattr("src.core.settings.agent_task_catalog.TASKS_CONFIG_DIR", missing_dir)
        get_tasks_config.cache_clear()

        with pytest.raises(FileNotFoundError, match="Task configuration directory not found"):
            get_tasks_config()

        get_tasks_config.cache_clear()
