"""Unit tests for src/observability/langsmith_backend.py

Tests verify that LangSmith initialization correctly handles API keys,
configuration flags, and library dependencies without crashing when they're missing.
"""

import os

import pytest

from src.observability.langsmith_backend import init_observability, is_observability_enabled


class TestInitObservability:
    """Tests for init_observability function."""

    def test_init_observability_already_initialized_returns_true(
        self, mock_config, reset_observability_state
    ):
        """
        Contract: When already initialized, return True without re-initializing.
        Mocking get_config because it reads from app settings.
        Everything else uses real implementation.
        """
        # Arrange: initialize once
        init_observability("test-project")
        # Act: call again
        result = init_observability("test-project")

        # Assert: returns True (already initialized)
        assert result is True

    def test_init_observability_disabled_by_config_returns_false(
        self, mock_config_disabled, reset_observability_state
    ):
        """
        Contract: When config.observability.enabled is False,
        return False without initializing.
        """
        # Arrange
        # mock_config_disabled already sets enabled=False

        # Act
        result = init_observability("test-project")

        # Assert
        assert result is False

    def test_init_observability_disabled_by_caller_returns_false(
        self, mock_config, reset_observability_state
    ):
        """
        Contract: When caller passes enabled=False,
        return False regardless of config.
        """
        # Arrange
        # mock_config has enabled=True

        # Act
        result = init_observability("test-project", enabled=False)

        # Assert
        assert result is False

    def test_init_observability_missing_api_key_returns_false_and_logs_warning(
        self, mock_config_no_api_key, reset_observability_state
    ):
        """
        Contract: When API key is missing, return False
        and log a warning (don't crash pipeline).
        Mocking get_config to control API key availability.
        Everything else uses real implementation.
        """
        # Arrange
        # mock_config_no_api_key has langsmith_api_key=None

        # Act
        result = init_observability("test-project", enabled=True)

        # Assert
        assert result is False

    def test_init_observability_litellm_import_fails_returns_false(
        self, mock_config, reset_observability_state, monkeypatch
    ):
        """
        Contract: When litellm library is unavailable,
        return False and log warning (don't crash).
        Mocking sys.modules to simulate ImportError.
        Everything else uses real implementation.
        """
        # Arrange: make litellm import fail
        import sys
        monkeypatch.setitem(sys.modules, "litellm", None)

        # Act
        result = init_observability("test-project", enabled=True)

        # Assert
        assert result is False

    def test_init_observability_success_sets_env_vars_and_returns_true(
        self,
        mock_config,
        reset_observability_state,
        mock_environment,
        mock_litellm_import,
    ):
        """
        Contract: When all conditions are met (config enabled, API key present,
        litellm available), initialize and return True.
        Also verify environment variables are set for third-party libs.
        Mocking config, environment, and litellm.
        Everything else uses real implementation.
        """
        # Arrange
        # All mocks are set up correctly

        # Act
        result = init_observability("my-project", enabled=True)

        # Assert
        assert result is True
        # Verify env vars were set
        assert os.environ.get("LANGSMITH_TRACING") == "true"
        assert os.environ.get("LANGSMITH_API_KEY") == "test-api-key-12345"
        assert os.environ.get("LANGSMITH_PROJECT") == "my-project"
        assert os.environ.get("LANGSMITH_ENDPOINT") == "https://api.smith.langchain.com"

    def test_init_observability_idempotent_second_call_returns_immediately(
        self, mock_config, reset_observability_state, mock_environment, mock_litellm_import
    ):
        """
        Contract: init_observability is idempotent — calling twice doesn't re-process.
        """
        # Arrange
        # Call once successfully
        result1 = init_observability("project1")
        assert result1 is True

        # Act: call again with different project name
        result2 = init_observability("project2")

        # Assert: still returns True, but project env var wasn't updated
        assert result2 is True
        assert os.environ.get("LANGSMITH_PROJECT") == "project1"  # Not "project2"

    def test_init_observability_uses_config_project_when_caller_provides_none(
        self,
        mock_config,
        reset_observability_state,
        mock_environment,
        mock_litellm_import,
    ):
        """
        Contract: When caller doesn't provide project_name,
        use the one from config.observability.project.
        """
        # Arrange
        # mock_config has project="test-project"

        # Act
        result = init_observability(project_name=None)

        # Assert
        assert result is True
        assert os.environ.get("LANGSMITH_PROJECT") == "test-project"


class TestIsObservabilityEnabled:
    """Tests for is_observability_enabled function."""

    def test_is_observability_enabled_returns_false_initially(self, reset_observability_state):
        """
        Contract: Before init_observability is called, returns False.
        """
        # Arrange: state is reset

        # Act
        result = is_observability_enabled()

        # Assert
        assert result is False

    def test_is_observability_enabled_returns_true_after_init(
        self,
        mock_config,
        reset_observability_state,
        mock_environment,
        mock_litellm_import,
    ):
        """
        Contract: After successful init_observability, returns True.
        """
        # Arrange: initialize
        init_observability("test-project")

        # Act
        result = is_observability_enabled()

        # Assert
        assert result is True
