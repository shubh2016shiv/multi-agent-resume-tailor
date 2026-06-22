"""Unit tests for src/core/llm_token_tracker/tracking_context.py."""

import logging
from types import SimpleNamespace

import pytest

from src.core import logger as logger_module
from src.core.llm_token_tracker import tracking_context as tracking_context_module
from src.core.llm_token_tracker.counter import get_token_counter
from src.core.llm_token_tracker.tracking_context import track_agent_tokens


class TestTrackAgentTokens:
    """Tests for token-tracking execution context."""

    def _configure_debug_logging(self, monkeypatch) -> None:
        """Enable observable debug logging with the real structlog pipeline."""
        debug_config = SimpleNamespace(
            logging=SimpleNamespace(level="DEBUG", format="console", log_file=None),
            application=SimpleNamespace(environment="development"),
        )
        monkeypatch.setattr("src.core.logger.get_config", lambda: debug_config)
        logger_module.configure_structlog()
        monkeypatch.setattr(
            tracking_context_module,
            "logger",
            logger_module.get_logger(tracking_context_module.__name__),
        )

    def test_track_agent_tokens_yields_the_shared_counter_instance(self, monkeypatch):
        """Contract: the context yields the shared token counter for in-block use."""
        monkeypatch.setattr(
            "src.core.llm_token_tracker.counter.token_counter",
            lambda *, model, messages: 9,
        )
        monkeypatch.setattr(
            "src.core.llm_token_tracker.counter.cost_per_token",
            lambda *, model, prompt_tokens, completion_tokens: (0.0, 0.0),
        )
        get_token_counter.cache_clear()

        with track_agent_tokens("summary_writer", "gpt-4o", "write a summary") as counter:
            shared_counter = get_token_counter()

        assert counter is shared_counter
        get_token_counter.cache_clear()

    def test_track_agent_tokens_logs_start_event_with_estimated_input_tokens(
        self, monkeypatch, caplog
    ):
        """Contract: entering the context logs the agent start event and token estimate."""
        monkeypatch.setattr(
            "src.core.llm_token_tracker.counter.token_counter",
            lambda *, model, messages: 9,
        )
        monkeypatch.setattr(
            "src.core.llm_token_tracker.counter.cost_per_token",
            lambda *, model, prompt_tokens, completion_tokens: (0.0, 0.0),
        )
        get_token_counter.cache_clear()
        caplog.set_level(logging.INFO)

        with track_agent_tokens("summary_writer", "gpt-4o", "write a summary"):
            pass

        assert "agent_execution_started" in caplog.text
        assert "estimated_input_tokens=9" in caplog.text
        assert "agent=summary_writer" in caplog.text
        get_token_counter.cache_clear()

    def test_track_agent_tokens_logs_completion_even_when_wrapped_block_raises(
        self, monkeypatch, capsys
    ):
        """Contract: completion logging happens even when the wrapped block fails."""
        monkeypatch.setattr(
            "src.core.llm_token_tracker.counter.token_counter",
            lambda *, model, messages: 9,
        )
        monkeypatch.setattr(
            "src.core.llm_token_tracker.counter.cost_per_token",
            lambda *, model, prompt_tokens, completion_tokens: (0.0, 0.0),
        )
        get_token_counter.cache_clear()
        self._configure_debug_logging(monkeypatch)

        with pytest.raises(RuntimeError, match="boom"):
            with track_agent_tokens("summary_writer", "gpt-4o", "write a summary"):
                raise RuntimeError("boom")

        captured = capsys.readouterr()
        assert "agent_execution_completed" in captured.out
        assert "summary_writer" in captured.out
        get_token_counter.cache_clear()
