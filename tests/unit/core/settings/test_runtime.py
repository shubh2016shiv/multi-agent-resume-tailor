"""Unit tests for src/core/settings/runtime.py."""

from src.core.settings.runtime import Settings, get_config


class TestGetConfig:
    """Tests for cached settings access."""

    def test_get_config_returns_the_process_wide_cached_settings_instance(self):
        """Contract: repeated access returns the same cached Settings object."""
        get_config.cache_clear()

        first_config = get_config()
        second_config = get_config()

        assert first_config is second_config
        get_config.cache_clear()


def test_settings_loads_deepseek_key_from_environment(monkeypatch):
    """DeepSeek credentials are environment-only settings."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-secret")

    settings = Settings()  # pyright: ignore[reportCallIssue]

    assert settings.deepseek_api_key == "test-secret"
