"""Fixtures for observability module tests."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_config():
    """Mock the get_config() function with observability settings."""
    with patch("src.observability.langsmith_backend.get_config") as mock:
        config = MagicMock()
        config.observability.enabled = True
        config.observability.project = "test-project"
        config.observability.endpoint = "https://api.smith.langchain.com"
        config.langsmith_api_key = "test-api-key-12345"
        mock.return_value = config
        yield mock


@pytest.fixture
def mock_config_disabled():
    """Mock get_config with observability disabled."""
    with patch("src.observability.langsmith_backend.get_config") as mock:
        config = MagicMock()
        config.observability.enabled = False
        config.langsmith_api_key = "test-api-key-12345"
        mock.return_value = config
        yield mock


@pytest.fixture
def mock_config_no_api_key():
    """Mock get_config with missing API key."""
    with patch("src.observability.langsmith_backend.get_config") as mock:
        config = MagicMock()
        config.observability.enabled = True
        config.langsmith_api_key = None
        mock.return_value = config
        yield mock


@pytest.fixture
def mock_logger():
    """Mock the structlog logger."""
    with patch("src.observability.langsmith_backend.get_logger") as mock:
        logger = MagicMock()
        mock.return_value = logger
        yield logger


@pytest.fixture
def mock_logger_tracing():
    """Mock the logger in tracing module."""
    with patch("src.observability.tracing.get_logger") as mock:
        logger = MagicMock()
        mock.return_value = logger
        yield logger


@pytest.fixture
def mock_logger_metrics():
    """Mock the logger in iteration_metrics module."""
    with patch("src.observability.iteration_metrics.logger") as mock_logger:
        yield mock_logger


@pytest.fixture
def mock_environment(monkeypatch):
    """Fixture to manage environment variables during tests."""
    # Clear LangSmith env vars before test
    for key in ["LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT", "LANGSMITH_ENDPOINT"]:
        monkeypatch.delenv(key, raising=False)
    yield monkeypatch


@pytest.fixture
def reset_observability_state():
    """Reset the _is_initialized state before and after each test.

    The langsmith_backend module maintains module-level state that persists
    across tests. This fixture ensures clean state.
    """
    # Reset before
    import src.observability.langsmith_backend as backend
    original_state = backend._is_initialized
    backend._is_initialized = False

    yield

    # Reset after
    backend._is_initialized = original_state


@pytest.fixture
def mock_litellm_import():
    """Mock successful litellm import."""
    mock_litellm = MagicMock()
    mock_litellm.callbacks = []

    with patch.dict("sys.modules", {"litellm": mock_litellm}):
        yield mock_litellm


@pytest.fixture
def mock_langsmith_traceable():
    """Mock langsmith.traceable decorator."""
    def traceable_decorator(run_type: str, name: str):
        def decorator(func):
            # Return a wrapped function that works like the original
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            wrapper.__name__ = func.__name__
            return wrapper
        return decorator

    mock_langsmith = MagicMock()
    mock_langsmith.traceable = traceable_decorator

    with patch.dict("sys.modules", {"langsmith": mock_langsmith}):
        yield mock_langsmith


@pytest.fixture
def mock_langsmith_run_helpers():
    """Mock langsmith.run_helpers.get_current_run_tree."""
    mock_run_tree = MagicMock()
    mock_run_tree.metadata = {}

    with patch("src.observability.iteration_metrics.get_current_run_tree") as mock:
        mock.return_value = mock_run_tree
        yield mock
