"""Unit tests for src/observability/tracing.py

Tests verify that trace decorators correctly wrap functions when tracing is enabled,
and pass through unchanged when tracing is disabled or langsmith is unavailable.
"""

import pytest

from src.observability.tracing import build_traced_function, trace_agent, trace_tool


def dummy_function(x):
    """A simple function for testing decoration."""
    return x * 2


class TestBuildTracedFunction:
    """Tests for build_traced_function helper."""

    def test_build_traced_function_observability_disabled_returns_unchanged(
        self, monkeypatch, reset_observability_state
    ):
        """
        Contract: When observability is disabled, return the function exactly as-is,
        without wrapping or importing langsmith.
        Mocking is_observability_enabled to return False.
        Everything else uses real implementation.
        """
        # Arrange: make is_observability_enabled return False
        monkeypatch.setattr(
            "src.observability.tracing.is_observability_enabled",
            lambda: False,
        )

        # Act
        result = build_traced_function("chain", dummy_function)

        # Assert: function is returned unchanged
        assert result is dummy_function
        assert result(5) == 10  # Original behavior

    def test_build_traced_function_langsmith_import_fails_returns_unchanged(
        self, monkeypatch, reset_observability_state
    ):
        """
        Contract: When langsmith library is unavailable,
        return function unchanged and log warning.
        Mocking is_observability_enabled to return True, but make langsmith import fail.
        Everything else uses real implementation.
        """
        # Arrange: enable observability but make langsmith import fail
        monkeypatch.setattr(
            "src.observability.tracing.is_observability_enabled",
            lambda: True,
        )
        # Make langsmith import raise ImportError
        import sys
        original_langsmith = sys.modules.get("langsmith")
        sys.modules["langsmith"] = None

        try:
            # Act
            result = build_traced_function("chain", dummy_function)

            # Assert: function is returned unchanged
            assert result is dummy_function
            assert result(5) == 10  # Original behavior
        finally:
            # Cleanup
            if original_langsmith is None:
                sys.modules.pop("langsmith", None)
            else:
                sys.modules["langsmith"] = original_langsmith

    def test_build_traced_function_langsmith_available_returns_wrapped(
        self, monkeypatch, reset_observability_state, mock_langsmith_traceable
    ):
        """
        Contract: When observability is enabled and langsmith is available,
        wrap the function with langsmith.traceable and return it.
        Mocking is_observability_enabled and langsmith.traceable.
        Everything else uses real implementation.
        """
        # Arrange: enable observability
        monkeypatch.setattr(
            "src.observability.tracing.is_observability_enabled",
            lambda: True,
        )

        # Act
        result = build_traced_function("chain", dummy_function)

        # Assert: function is wrapped (different object), but still callable
        assert result is not dummy_function
        assert result(5) == 10  # Wrapped function still works

    def test_build_traced_function_wraps_with_both_run_types(
        self, monkeypatch, reset_observability_state, mock_langsmith_traceable
    ):
        """
        Contract: build_traced_function successfully wraps both "chain" and "tool" run types.
        """
        # Arrange
        monkeypatch.setattr(
            "src.observability.tracing.is_observability_enabled",
            lambda: True,
        )

        # Act: wrap with both run types
        result_chain = build_traced_function("chain", dummy_function)
        result_tool = build_traced_function("tool", dummy_function)

        # Assert: both return wrapped functions (different from original)
        assert result_chain is not dummy_function
        assert result_tool is not dummy_function
        # But they still work
        assert result_chain(5) == 10
        assert result_tool(5) == 10


class TestTraceAgent:
    """Tests for trace_agent decorator."""

    def test_trace_agent_disables_function_wrapping_when_observability_off(
        self, monkeypatch, reset_observability_state
    ):
        """
        Contract: When observability is off, trace_agent returns the function unchanged.
        Mocking is_observability_enabled to return False.
        Everything else uses real implementation.
        """
        # Arrange
        monkeypatch.setattr(
            "src.observability.tracing.is_observability_enabled",
            lambda: False,
        )

        # Act
        result = trace_agent(dummy_function)

        # Assert
        assert result is dummy_function

    def test_trace_agent_calls_build_traced_function_with_chain(
        self, monkeypatch, reset_observability_state
    ):
        """
        Contract: trace_agent calls build_traced_function with run_type="chain".
        """
        # Arrange: capture build_traced_function calls
        calls = []

        def mock_build_traced(run_type, func):
            calls.append({"run_type": run_type, "func": func})
            return func

        monkeypatch.setattr(
            "src.observability.tracing.build_traced_function",
            mock_build_traced,
        )

        # Act
        result = trace_agent(dummy_function)

        # Assert
        assert len(calls) == 1
        assert calls[0]["run_type"] == "chain"
        assert calls[0]["func"] is dummy_function


class TestTraceTool:
    """Tests for trace_tool decorator."""

    def test_trace_tool_disables_function_wrapping_when_observability_off(
        self, monkeypatch, reset_observability_state
    ):
        """
        Contract: When observability is off, trace_tool returns the function unchanged.
        Mocking is_observability_enabled to return False.
        Everything else uses real implementation.
        """
        # Arrange
        monkeypatch.setattr(
            "src.observability.tracing.is_observability_enabled",
            lambda: False,
        )

        # Act
        result = trace_tool(dummy_function)

        # Assert
        assert result is dummy_function

    def test_trace_tool_calls_build_traced_function_with_tool(
        self, monkeypatch, reset_observability_state
    ):
        """
        Contract: trace_tool calls build_traced_function with run_type="tool".
        """
        # Arrange: capture build_traced_function calls
        calls = []

        def mock_build_traced(run_type, func):
            calls.append({"run_type": run_type, "func": func})
            return func

        monkeypatch.setattr(
            "src.observability.tracing.build_traced_function",
            mock_build_traced,
        )

        # Act
        result = trace_tool(dummy_function)

        # Assert
        assert len(calls) == 1
        assert calls[0]["run_type"] == "tool"
        assert calls[0]["func"] is dummy_function
