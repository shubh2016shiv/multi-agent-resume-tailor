"""Build ready-to-use LLM clients for agents and tools.

Agents and tools need a CrewAI ``LLM`` object (the client that talks to a
model). Different providers need different settings — for example DeepSeek
wants JSON mode one way, OpenAI another; API keys also differ.

This file is the single place that:

1. Takes shared defaults from settings
2. Applies any one-off overrides for this call
3. Picks the right API key
4. Adds provider-specific options when needed
5. Returns a configured ``LLM``

Who calls it
------------
- Agent factories under ``src/agents/*/agent.py`` — each agent gets its LLM here
- ``src/tools/llm_gateway/structured_output.py`` — tools that need JSON shaped
  like a Pydantic model

Helpers ``is_deepseek_model`` and ``structured_response_format`` are also used
by ``src/orchestration/crew_task_execution.py`` when an agent already has an
LLM and we only need to set how structured answers should look.
"""

from collections.abc import Mapping
from typing import Any

from crewai import LLM
from pydantic import BaseModel

from src.core.settings import get_config

DEEPSEEK_MODEL_PREFIX = "deepseek/"
DEEPSEEK_REASONING_EFFORTS = {"low", "high", "max"}
DEEPSEEK_THINKING_MODES = {"enabled", "disabled"}


def is_deepseek_model(model: str) -> bool:
    """Return True if this model name is a DeepSeek model.

    LiteLLM (the library that routes model calls) uses names like
    ``deepseek/deepseek-chat``. Anything starting with ``deepseek/`` counts.
    """
    ####################################################
    # STEP 1: TREAT THE LITELLM PREFIX AS THE PROVIDER SIGNAL
    ####################################################
    # Other helpers in this file branch on this one check.
    return model.startswith(DEEPSEEK_MODEL_PREFIX)


def structured_response_format(model: str, output_model: type[BaseModel]) -> Any:
    """Choose how to ask the model for structured (JSON) output.

    Args:
        model: Model name (e.g. ``gpt-4o`` or ``deepseek/deepseek-chat``).
        output_model: Pydantic class describing the JSON shape we want back.

    Returns:
        - For DeepSeek: ``{"type": "json_object"}`` (what that provider accepts)
        - For others: the Pydantic class itself (what those providers accept)

    Why this exists: not every provider accepts a Pydantic class as
    ``response_format``. DeepSeek needs the simpler JSON-object flag instead.
    """
    ####################################################
    # STEP 1: PICK THE FORMAT SHAPE THE PROVIDER ACTUALLY SUPPORTS
    ####################################################
    if is_deepseek_model(model):
        return {"type": "json_object"}
    return output_model


def build_llm(model_config: Mapping[str, Any], output_model: type[BaseModel] | None = None) -> LLM:
    """Create a CrewAI ``LLM`` from defaults plus optional overrides.

    Args:
        model_config: Extra settings for this call only (model name,
            temperature, etc.). These override the shared ``settings.llm``
            values when both define the same key.
        output_model: Optional Pydantic class. Pass it when you need the
            model to return JSON matching that shape. Omit it for normal
            free-text agent replies.

    Returns:
        A configured CrewAI ``LLM`` ready to use.

    Example:
        Agent code often passes its YAML/config dict::

            llm = build_llm(agent_config)

        Structured tools pass a Pydantic model too::

            llm = build_llm({"model": "..."}, output_model=Resume)
    """
    ####################################################
    # STEP 1: MERGE PROCESS DEFAULTS WITH THIS CALL'S OVERRIDES
    ####################################################
    # Shared defaults first; model_config wins on conflicts
    # (e.g. temperature=0.0 for a strict gate decision).
    settings = get_config()
    merged = {**settings.llm.model_dump(), **model_config}

    ####################################################
    # STEP 2: RESOLVE THE CANONICAL MODEL IDENTIFIER
    ####################################################
    # Older override dicts used the key "llm"; settings use "model".
    # Prefer "llm" when present so both styles keep working.
    model = str(merged.get("llm") or merged["model"])

    ####################################################
    # STEP 3: ATTACH THE ENVIRONMENT-ONLY API KEY FOR THAT MODEL
    ####################################################
    # API keys come from the environment (via settings), never from YAML.
    merged["api_key"] = _api_key_for_model(model, settings)

    ####################################################
    # STEP 4: BUILD THE PROVIDER-NEUTRAL CREWAI PARAMETER SET
    ####################################################
    params = _base_params(model, merged, output_model)

    ####################################################
    # STEP 5: LAYER DEEPSEEK-ONLY KNOBS WHEN THE MODEL IS DEEPSEEK
    ####################################################
    # Thinking mode, api_base, and reasoning_effort only apply to DeepSeek.
    if is_deepseek_model(model):
        params.update(_deepseek_params(merged))

    ####################################################
    # STEP 6: DROP NONE VALUES AND CONSTRUCT THE CREWAI LLM
    ####################################################
    # Passing temperature=None is not the same as omitting temperature for
    # some CrewAI/DeepSeek paths. Drop Nones so "unset" stays unset.
    return LLM(**{key: value for key, value in params.items() if value is not None})


def _api_key_for_model(model: str, settings: Any) -> str | None:
    """Pick the API key that matches this model family.

    DeepSeek models use ``settings.deepseek_api_key``.
    OpenAI-style names (``openai/...`` or ``gpt-...``) use
    ``settings.openai_api_key``.
    Anything else returns ``None`` — we do not invent a key here.
    """
    ####################################################
    # STEP 1: MAP MODEL FAMILY → THE MATCHING SETTINGS KEY
    ####################################################
    if is_deepseek_model(model):
        return settings.deepseek_api_key
    if model.startswith("openai/") or model.startswith("gpt-"):
        return settings.openai_api_key
    ####################################################
    # STEP 2: LEAVE THE KEY UNSET FOR UNRECOGNIZED FAMILIES
    ####################################################
    # CrewAI/LiteLLM may still find a key in the environment on their own.
    return None


def _base_params(
    model: str, config: Mapping[str, Any], output_model: type[BaseModel] | None
) -> dict[str, Any]:
    """Build the common LLM settings every provider understands.

    Includes model name, API key, temperature, timeout, max tokens, and
    optional structured-output format. DeepSeek-only options are added later.
    """
    ####################################################
    # STEP 1: DECIDE WHETHER THIS CALL REQUESTS STRUCTURED OUTPUT
    ####################################################
    response_format = None
    if output_model is not None:
        response_format = structured_response_format(model, output_model)

    ####################################################
    # STEP 2: ASSEMBLE THE SHARED PARAMETER DICT
    ####################################################
    return {
        "model": model,
        "api_key": config.get("api_key"),
        "temperature": config.get("temperature"),
        "timeout": config.get("timeout"),
        "max_tokens": config.get("max_tokens"),
        "response_format": response_format,
    }


def _deepseek_params(config: Mapping[str, Any]) -> dict[str, Any]:
    """Build DeepSeek-only options (thinking mode, API base, reasoning).

    Raises:
        ValueError: If ``thinking`` is not ``enabled`` or ``disabled``.
    """
    ####################################################
    # STEP 1: VALIDATE THINKING MODE AGAINST THE ALLOWLIST
    ####################################################
    thinking = str(config.get("thinking", "disabled"))
    if thinking not in DEEPSEEK_THINKING_MODES:
        raise ValueError(f"Unsupported DeepSeek thinking mode: {thinking}")

    ####################################################
    # STEP 2: SET API BASE, KEY, AND THE THINKING EXTRA BODY
    ####################################################
    params: dict[str, Any] = {
        "api_base": config.get("api_base"),
        "api_key": config.get("api_key"),
        "extra_body": {"thinking": {"type": thinking}},
    }

    ####################################################
    # STEP 3: WHEN THINKING IS ON, ADD REASONING EFFORT AND CLEAR TEMPERATURE
    ####################################################
    # With thinking enabled, DeepSeek controls sampling; keeping an explicit
    # temperature can fight that path, so we clear it (build_llm drops None).
    if thinking == "enabled":
        params.update(_reasoning_params(config))
        params["temperature"] = None
    return params


def _reasoning_params(config: Mapping[str, Any]) -> dict[str, Any]:
    """Build DeepSeek reasoning-effort settings when thinking is enabled.

    Allowed efforts: ``low``, ``high``, ``max`` (default ``high``).

    Raises:
        ValueError: If the effort value is not in that allowlist.
    """
    ####################################################
    # STEP 1: VALIDATE REASONING EFFORT AGAINST THE ALLOWLIST
    ####################################################
    effort = str(config.get("reasoning_effort", "high"))
    if effort not in DEEPSEEK_REASONING_EFFORTS:
        raise ValueError(f"Unsupported DeepSeek reasoning effort: {effort}")

    ####################################################
    # STEP 2: EXPOSE EFFORT THROUGH CREWAI'S OPENAI-COMPATIBLE ESCAPE HATCH
    ####################################################
    # allowed_openai_params tells CrewAI/LiteLLM to keep reasoning_effort
    # instead of stripping it as an unknown argument.
    return {
        "reasoning_effort": effort,
        "allowed_openai_params": ["reasoning_effort"],
    }
