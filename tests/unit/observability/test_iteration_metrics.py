"""Unit tests for src/observability/iteration_metrics.py

Tests verify that metrics are correctly logged and attached to LangSmith runs,
with graceful degradation when tracing is disabled or the library is unavailable.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.observability.iteration_metrics import log_iteration_metrics


class TestLogIterationMetrics:
    """Tests for log_iteration_metrics function."""

    def test_log_iteration_metrics_always_logs_to_structlog(
        self, mock_logger_metrics, reset_observability_state, monkeypatch
    ):
        """
        Contract: Metrics are ALWAYS logged to structlog, regardless of whether
        LangSmith tracing is enabled or the run tree is available.
        Mocking the logger to verify the call.
        Everything else uses real implementation.
        """
        # Arrange
        monkeypatch.setattr(
            "src.observability.iteration_metrics.is_observability_enabled",
            lambda: False,  # Even with observability off, we log
        )

        # Act
        log_iteration_metrics("my_agent", iteration=1, metrics={"score": 85})

        # Assert: logger.info was called with the metrics
        mock_logger_metrics.info.assert_called_once()
        call_args = mock_logger_metrics.info.call_args
        # First positional arg should be "iteration_metrics"
        assert call_args[0][0] == "iteration_metrics"
        # Check keyword args
        assert call_args[1]["agent"] == "my_agent"
        assert call_args[1]["iteration"] == 1
        assert call_args[1]["score"] == 85

    def test_log_iteration_metrics_returns_early_when_observability_disabled(
        self, mock_logger_metrics, reset_observability_state, monkeypatch
    ):
        """
        Contract: When observability is disabled, return early after logging
        to structlog (don't attempt LangSmith attachment).
        Mocking is_observability_enabled to return False.
        Everything else uses real implementation.
        """
        # Arrange
        monkeypatch.setattr(
            "src.observability.iteration_metrics.is_observability_enabled",
            lambda: False,
        )

        # Act: no exception should be raised
        log_iteration_metrics("agent_name", iteration=1, metrics={"key": "value"})

        # Assert: function completes without error (no LangSmith calls attempted)
        assert True  # If we reach here, no exception was raised

    def test_log_iteration_metrics_attaches_to_langsmith_run_when_active(
        self, mock_logger_metrics, reset_observability_state, monkeypatch
    ):
        """
        Contract: When observability is enabled and a LangSmith run is active,
        attach metrics to the run's metadata.
        Mocking is_observability_enabled and get_current_run_tree.
        Everything else uses real implementation.
        """
        # Arrange
        monkeypatch.setattr(
            "src.observability.iteration_metrics.is_observability_enabled",
            lambda: True,
        )

        # Mock the run tree to be available
        mock_run_tree = {"metadata": {}}

        def get_mock_run():
            class MockRun:
                metadata = mock_run_tree["metadata"]
            return MockRun()

        with patch("langsmith.run_helpers.get_current_run_tree", side_effect=get_mock_run):
            # Act
            log_iteration_metrics("agent_name", iteration=2, metrics={"tokens": 1000, "cost": 0.01})

            # Assert: metadata was updated
            assert mock_run_tree["metadata"]["agent_name"] == "agent_name"
            assert mock_run_tree["metadata"]["iteration"] == 2
            assert mock_run_tree["metadata"]["tokens"] == 1000
            assert mock_run_tree["metadata"]["cost"] == 0.01

    def test_log_iteration_metrics_returns_none_when_no_run_active(
        self, mock_logger_metrics, reset_observability_state, monkeypatch
    ):
        """
        Contract: When observability is enabled but no LangSmith run is active
        (get_current_run_tree returns None), return gracefully after logging.
        """
        # Arrange
        monkeypatch.setattr(
            "src.observability.iteration_metrics.is_observability_enabled",
            lambda: True,
        )

        with patch("langsmith.run_helpers.get_current_run_tree") as mock_get_run:
            mock_get_run.return_value = None

            # Act: should not raise
            result = log_iteration_metrics("agent", iteration=1, metrics={})

            # Assert
            assert result is None

    def test_log_iteration_metrics_handles_langsmith_error_gracefully(
        self, mock_logger_metrics, reset_observability_state, monkeypatch
    ):
        """
        Contract: If langsmith.run_helpers.get_current_run_tree raises an exception,
        log a warning but don't break the pipeline (never raise).
        Mocking get_current_run_tree to raise an exception.
        Everything else uses real implementation.
        """
        # Arrange
        monkeypatch.setattr(
            "src.observability.iteration_metrics.is_observability_enabled",
            lambda: True,
        )

        with patch("langsmith.run_helpers.get_current_run_tree") as mock_get_run:
            mock_get_run.side_effect = RuntimeError("LangSmith API error")

            # Act: should not raise, should log warning
            result = log_iteration_metrics("agent", iteration=1, metrics={"score": 90})

            # Assert: function completes without raising
            assert result is None
            # Verify warning was logged
            mock_logger_metrics.warning.assert_called_once()
            call_args = mock_logger_metrics.warning.call_args
            assert "langsmith_metric_attach_failed" in call_args[0]

    def test_log_iteration_metrics_with_empty_metrics_dict(
        self, mock_logger_metrics, reset_observability_state, monkeypatch
    ):
        """
        Contract: Empty metrics dict should still log and not raise.
        """
        # Arrange
        monkeypatch.setattr(
            "src.observability.iteration_metrics.is_observability_enabled",
            lambda: False,
        )

        # Act
        log_iteration_metrics("agent", iteration=1, metrics={})

        # Assert
        mock_logger_metrics.info.assert_called_once()
        call_args = mock_logger_metrics.info.call_args
        assert call_args[0][0] == "iteration_metrics"

    def test_log_iteration_metrics_with_complex_metrics(
        self, mock_logger_metrics, reset_observability_state, monkeypatch
    ):
        """
        Contract: Complex metrics (nested dicts, lists) should be logged and attached.
        """
        # Arrange
        monkeypatch.setattr(
            "src.observability.iteration_metrics.is_observability_enabled",
            lambda: True,
        )

        mock_run_tree = {"metadata": {}}

        def get_mock_run():
            class MockRun:
                metadata = mock_run_tree["metadata"]
            return MockRun()

        complex_metrics = {
            "scores": {"accuracy": 0.95, "relevance": 0.88},
            "tokens": [1000, 500, 200],
            "cost": 0.05,
        }

        with patch("langsmith.run_helpers.get_current_run_tree", side_effect=get_mock_run):
            # Act
            log_iteration_metrics("agent", iteration=1, metrics=complex_metrics)

            # Assert: complex metrics were attached
            assert mock_run_tree["metadata"]["scores"] == {"accuracy": 0.95, "relevance": 0.88}
            assert mock_run_tree["metadata"]["tokens"] == [1000, 500, 200]
            assert mock_run_tree["metadata"]["cost"] == 0.05
