"""Unit tests for provider-aware CrewAI LLM construction."""

from types import SimpleNamespace
from unittest.mock import patch

import litellm
from pydantic import BaseModel

from src.core.llm_factory import build_llm, structured_response_format


class ExampleOutput(BaseModel):
    """Minimal structured-output contract used by these tests."""

    value: str


def _settings() -> SimpleNamespace:
    """Return isolated shared defaults without loading project configuration."""
    llm = SimpleNamespace(
        model_dump=lambda: {
            "model": "deepseek/deepseek-v4-flash",
            "api_base": "https://api.deepseek.com",
            "temperature": 0.3,
            "thinking": "disabled",
            "reasoning_effort": None,
            "max_tokens": 4096,
            "timeout": 120,
        }
    )
    return SimpleNamespace(
        llm=llm,
        deepseek_api_key="deepseek-secret",
        openai_api_key="openai-secret",
    )


def test_build_llm_passes_deepseek_high_reasoning_without_temperature():
    """Thinking requests preserve effort and omit ineffective sampling controls."""
    config = {
        "llm": "deepseek/deepseek-v4-pro",
        "thinking": "enabled",
        "reasoning_effort": "high",
        "temperature": 0.9,
        "max_tokens": 6000,
    }
    with patch("src.core.llm_factory.get_config", return_value=_settings()):
        with patch("src.core.llm_factory.LLM") as llm_class:
            build_llm(config)

    kwargs = llm_class.call_args.kwargs
    assert "temperature" not in kwargs
    assert kwargs["reasoning_effort"] == "high"
    assert kwargs["allowed_openai_params"] == ["reasoning_effort"]
    assert kwargs["extra_body"] == {"thinking": {"type": "enabled"}}
    assert kwargs["max_tokens"] == 6000


def test_build_llm_keeps_temperature_when_deepseek_thinking_is_disabled():
    """Non-thinking DeepSeek requests retain configured creative sampling."""
    config = {
        "llm": "deepseek/deepseek-v4-pro",
        "thinking": "disabled",
        "temperature": 0.5,
    }
    with patch("src.core.llm_factory.get_config", return_value=_settings()):
        with patch("src.core.llm_factory.LLM") as llm_class:
            build_llm(config)

    kwargs = llm_class.call_args.kwargs
    assert kwargs["temperature"] == 0.5
    assert "reasoning_effort" not in kwargs
    assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}


def test_build_llm_uses_openai_key_for_gpt_structured_output():
    """Strict GPT contract calls use the OpenAI key, not the DeepSeek key."""
    with patch("src.core.llm_factory.get_config", return_value=_settings()):
        with patch("src.core.llm_factory.LLM") as llm_class:
            build_llm({"model": "gpt-4o"}, ExampleOutput)

    kwargs = llm_class.call_args.kwargs
    assert kwargs["api_key"] == "openai-secret"
    assert kwargs["response_format"] is ExampleOutput
    assert "extra_body" not in kwargs


def test_deepseek_structured_output_uses_json_object_mode():
    """DeepSeek Chat Completions never receives unsupported JSON Schema mode."""
    result = structured_response_format("deepseek/deepseek-v4-flash", ExampleOutput)

    assert result == {"type": "json_object"}


def test_installed_litellm_preserves_allowlisted_deepseek_effort():
    """The local LiteLLM adapter keeps both thinking and the requested effort."""
    optional = litellm.get_optional_params(  # pyright: ignore[reportPrivateImportUsage]
        model="deepseek-v4-pro",
        custom_llm_provider="deepseek",
        drop_params=True,
        reasoning_effort="high",
        allowed_openai_params=["reasoning_effort"],
        extra_body={"thinking": {"type": "enabled"}},
    )

    assert optional["reasoning_effort"] == "high"
    assert optional["extra_body"]["thinking"] == {"type": "enabled"}
