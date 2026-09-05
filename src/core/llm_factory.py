"""Provider-aware CrewAI LLM construction."""

from collections.abc import Mapping
from typing import Any

from crewai import LLM
from pydantic import BaseModel

from src.core.settings import get_config

DEEPSEEK_MODEL_PREFIX = "deepseek/"
DEEPSEEK_REASONING_EFFORTS = {"low", "high", "max"}
DEEPSEEK_THINKING_MODES = {"enabled", "disabled"}


def is_deepseek_model(model: str) -> bool:
    """Return whether a LiteLLM model identifier selects DeepSeek."""
    return model.startswith(DEEPSEEK_MODEL_PREFIX)


def structured_response_format(model: str, output_model: type[BaseModel]) -> Any:
    """Return a provider-supported structured response format."""
    if is_deepseek_model(model):
        return {"type": "json_object"}
    return output_model


def build_llm(model_config: Mapping[str, Any], output_model: type[BaseModel] | None = None) -> LLM:
    """Build a CrewAI LLM from shared defaults plus per-call overrides."""
    settings = get_config()
    merged = {**settings.llm.model_dump(), **model_config}
    model = str(merged.get("llm") or merged["model"])
    merged["api_key"] = _api_key_for_model(model, settings)
    params = _base_params(model, merged, output_model)
    if is_deepseek_model(model):
        params.update(_deepseek_params(merged))
    return LLM(**{key: value for key, value in params.items() if value is not None})


def _api_key_for_model(model: str, settings: Any) -> str | None:
    """Return the environment-only API key associated with the selected model."""
    if is_deepseek_model(model):
        return settings.deepseek_api_key
    if model.startswith("openai/") or model.startswith("gpt-"):
        return settings.openai_api_key
    return None


def _base_params(
    model: str, config: Mapping[str, Any], output_model: type[BaseModel] | None
) -> dict[str, Any]:
    """Return provider-neutral CrewAI LLM parameters."""
    response_format = None
    if output_model is not None:
        response_format = structured_response_format(model, output_model)
    return {
        "model": model,
        "api_key": config.get("api_key"),
        "temperature": config.get("temperature"),
        "timeout": config.get("timeout"),
        "max_tokens": config.get("max_tokens"),
        "response_format": response_format,
    }


def _deepseek_params(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return validated DeepSeek OpenAI-compatible request parameters."""
    thinking = str(config.get("thinking", "disabled"))
    if thinking not in DEEPSEEK_THINKING_MODES:
        raise ValueError(f"Unsupported DeepSeek thinking mode: {thinking}")
    params: dict[str, Any] = {
        "api_base": config.get("api_base"),
        "api_key": config.get("api_key"),
        "extra_body": {"thinking": {"type": thinking}},
    }
    if thinking == "enabled":
        params.update(_reasoning_params(config))
        params["temperature"] = None
    return params


def _reasoning_params(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return an allowlisted DeepSeek reasoning-effort parameter."""
    effort = str(config.get("reasoning_effort", "high"))
    if effort not in DEEPSEEK_REASONING_EFFORTS:
        raise ValueError(f"Unsupported DeepSeek reasoning effort: {effort}")
    return {
        "reasoning_effort": effort,
        "allowed_openai_params": ["reasoning_effort"],
    }
