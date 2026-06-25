"""Unit tests for src/core/settings/runtime.py."""

from src.core.settings.runtime import get_config


class TestGetConfig:
    """Tests for cached settings access."""

    def test_get_config_returns_the_process_wide_cached_settings_instance(self):
        """Contract: repeated access returns the same cached Settings object."""
        get_config.cache_clear()

        first_config = get_config()
        second_config = get_config()

        assert first_config is second_config
        get_config.cache_clear()
