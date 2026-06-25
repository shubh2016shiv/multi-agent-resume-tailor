"""Unit tests for src/agents/agent_config.py

Contract under test: load_agent_config(name) returns the config dict for the
named agent, or raises RuntimeError if any of the four required fields is absent.
"""

from unittest.mock import patch

import pytest

from src.agents.agent_config import load_agent_config


class TestLoadAgentConfig:
    """Tests for load_agent_config — the shared config loader for all agent factories."""

    def test_load_agent_config_with_all_required_fields_returns_config_dict(self):
        """
        Contract: returns the full config dict when all four required fields are present.
        Mocking get_agents_config because it reads agents.yaml from disk.
        Everything else here uses the real implementation.
        """
        # Arrange
        complete_config = {
            "role": "Resume Tailoring Specialist",
            "goal": "Produce an ATS-optimised resume.",
            "backstory": "Expert in resume tailoring.",
            "llm": "anthropic/claude-sonnet-4-5",
            "temperature": 0.2,
        }
        with patch("src.agents.agent_config.get_agents_config") as mock_get:
            mock_get.return_value = {"my_agent": complete_config}

            # Act
            result = load_agent_config("my_agent")

        # Assert
        assert result == complete_config

    def test_load_agent_config_with_missing_llm_field_raises_runtime_error(self):
        """
        Contract: raises RuntimeError when 'llm' is absent — every field is required.
        Expected value derived from docstring: "Raises: RuntimeError if any required field is missing."
        """
        # Arrange
        incomplete_config = {
            "role": "Specialist",
            "goal": "Do something.",
            "backstory": "Expert.",
            # 'llm' deliberately absent
        }
        with patch("src.agents.agent_config.get_agents_config") as mock_get:
            mock_get.return_value = {"my_agent": incomplete_config}

            # Act / Assert
            with pytest.raises(RuntimeError, match="llm"):
                load_agent_config("my_agent")

    def test_load_agent_config_with_missing_role_field_raises_runtime_error(self):
        """Contract: raises RuntimeError when 'role' is absent."""
        incomplete_config = {
            "goal": "Do something.",
            "backstory": "Expert.",
            "llm": "anthropic/claude-sonnet-4-5",
        }
        with patch("src.agents.agent_config.get_agents_config") as mock_get:
            mock_get.return_value = {"my_agent": incomplete_config}

            with pytest.raises(RuntimeError, match="role"):
                load_agent_config("my_agent")

    def test_load_agent_config_with_unknown_agent_name_raises_runtime_error(self):
        """
        Contract: raises RuntimeError when the agent name is not found in agents.yaml.
        get_agents_config().get(name, {}) returns {}, all required fields are missing.
        """
        with patch("src.agents.agent_config.get_agents_config") as mock_get:
            mock_get.return_value = {}  # no agents defined

            with pytest.raises(RuntimeError, match="role"):
                load_agent_config("nonexistent_agent")

    def test_load_agent_config_with_multiple_missing_fields_names_all_in_error(self):
        """
        Contract: the RuntimeError message lists every missing field, not just the first.
        Derived from: the error message includes the full 'missing' list.
        """
        empty_config = {}  # all four fields absent
        with patch("src.agents.agent_config.get_agents_config") as mock_get:
            mock_get.return_value = {"my_agent": empty_config}

            with pytest.raises(RuntimeError) as exc_info:
                load_agent_config("my_agent")

        error_message = str(exc_info.value)
        assert "role" in error_message
        assert "goal" in error_message
        assert "backstory" in error_message
        assert "llm" in error_message

    @pytest.mark.parametrize(
        "required_field",
        ["role", "goal", "backstory", "llm"],
    )
    def test_load_agent_config_with_any_single_missing_field_raises_runtime_error(
        self, required_field
    ):
        """
        Contract: every one of the four required fields is mandatory on its own.
        Parametrised to avoid four near-identical test bodies.
        """
        # Build a complete config then remove exactly one field
        complete_config = {
            "role": "Specialist",
            "goal": "Do something.",
            "backstory": "Expert.",
            "llm": "anthropic/claude-sonnet-4-5",
        }
        del complete_config[required_field]

        with patch("src.agents.agent_config.get_agents_config") as mock_get:
            mock_get.return_value = {"my_agent": complete_config}

            with pytest.raises(RuntimeError, match=required_field):
                load_agent_config("my_agent")
