"""Unit tests for src/core/llm_cache.py."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from src.core import llm_cache as llm_cache_module


class TestConfigureLlmCache:
    """Tests for synchronizing LiteLLM cache state with the feature flag."""

    def test_configure_llm_cache_enables_disk_cache_when_feature_flag_is_true(
        self, monkeypatch
    ) -> None:
        """Contract: enabling the flag installs LiteLLM's disk cache once."""
        fake_cache_factory = MagicMock(return_value="disk-cache")
        fake_litellm = SimpleNamespace(cache=None, Cache=fake_cache_factory)

        monkeypatch.setattr(
            llm_cache_module,
            "get_config",
            lambda: SimpleNamespace(feature_flags=SimpleNamespace(enable_cache=True)),
        )
        monkeypatch.setattr(llm_cache_module, "litellm", fake_litellm)
        monkeypatch.setattr(llm_cache_module, "logger", MagicMock())
        monkeypatch.setattr(llm_cache_module, "_configured_cache_enabled", None)

        llm_cache_module.configure_llm_cache()

        fake_cache_factory.assert_called_once_with(type="disk", disk_cache_dir=".litellm_cache")
        assert fake_litellm.cache == "disk-cache"
        assert llm_cache_module._configured_cache_enabled is True

    def test_configure_llm_cache_disables_existing_cache_when_feature_flag_is_false(
        self, monkeypatch
    ) -> None:
        """Contract: disabling the flag clears any previously configured LiteLLM cache."""
        fake_cache_factory = MagicMock()
        fake_litellm = SimpleNamespace(cache="existing-cache", Cache=fake_cache_factory)

        monkeypatch.setattr(
            llm_cache_module,
            "get_config",
            lambda: SimpleNamespace(feature_flags=SimpleNamespace(enable_cache=False)),
        )
        monkeypatch.setattr(llm_cache_module, "litellm", fake_litellm)
        monkeypatch.setattr(llm_cache_module, "logger", MagicMock())
        monkeypatch.setattr(llm_cache_module, "_configured_cache_enabled", True)

        llm_cache_module.configure_llm_cache()

        fake_cache_factory.assert_not_called()
        assert fake_litellm.cache is None
        assert llm_cache_module._configured_cache_enabled is False

    def test_configure_llm_cache_reconfigures_when_the_feature_flag_changes(
        self, monkeypatch
    ) -> None:
        """Contract: a later flag change is applied instead of being blocked by stale process state."""
        cache_enabled = False
        fake_cache_factory = MagicMock(return_value="disk-cache")
        fake_litellm = SimpleNamespace(cache=None, Cache=fake_cache_factory)

        def get_config():
            return SimpleNamespace(feature_flags=SimpleNamespace(enable_cache=cache_enabled))

        monkeypatch.setattr(llm_cache_module, "get_config", get_config)
        monkeypatch.setattr(llm_cache_module, "litellm", fake_litellm)
        monkeypatch.setattr(llm_cache_module, "logger", MagicMock())
        monkeypatch.setattr(llm_cache_module, "_configured_cache_enabled", None)

        llm_cache_module.configure_llm_cache()
        assert fake_litellm.cache is None

        cache_enabled = True
        llm_cache_module.configure_llm_cache()

        fake_cache_factory.assert_called_once_with(type="disk", disk_cache_dir=".litellm_cache")
        assert fake_litellm.cache == "disk-cache"
        assert llm_cache_module._configured_cache_enabled is True
