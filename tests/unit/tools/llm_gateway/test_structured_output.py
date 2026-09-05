"""Unit tests for the provider-aware structured-output gateway."""

from types import SimpleNamespace
from unittest.mock import patch

from pydantic import BaseModel

# Initialize agent exports first; src.tools currently has eager package exports that
# otherwise form a pre-existing import cycle during isolated gateway collection.
import src.agents  # noqa: F401
from src.tools.llm_gateway.structured_output import (
    add_json_contract,
    build_structured_llm,
    request_structured_output,
)


class ExampleOutput(BaseModel):
    """Minimal gateway output contract."""

    value: str


def test_build_structured_llm_delegates_to_provider_factory():
    """The gateway uses the centralized builder with its Pydantic contract."""
    with patch("src.tools.llm_gateway.structured_output.get_config") as config:
        config.return_value.llm.structured_model = "gpt-4o"
        with patch("src.tools.llm_gateway.structured_output.build_llm") as builder:
            build_structured_llm(ExampleOutput, temperature=0.0)

    builder.assert_called_once_with({"model": "gpt-4o", "temperature": 0.0}, ExampleOutput)


def test_add_json_contract_includes_schema_for_deepseek():
    """DeepSeek JSON mode receives explicit JSON and schema instructions."""
    result = add_json_contract("Review this.", ExampleOutput, "deepseek/deepseek-v4-flash")

    assert "Return only one valid JSON object" in result
    assert '"value"' in result


def test_token_budget_uses_the_structured_model():
    """The token guard counts against the model that receives the strict call."""
    llm_config = SimpleNamespace(structured_model="gpt-4o", structured_input_token_budget=1000)
    with patch("src.tools.llm_gateway.structured_output.configure_llm_cache"):
        with patch("src.tools.llm_gateway.structured_output.get_config") as config:
            with patch("src.tools.llm_gateway.structured_output.ensure_token_budget") as guard:
                with patch(
                    "src.tools.llm_gateway.structured_output._request_structured_output",
                    return_value=ExampleOutput(value="ok"),
                ):
                    config.return_value.llm = llm_config
                    request_structured_output(ExampleOutput, "Review this.", "input")

    guarded_text, guarded_model = guard.call_args.args[:2]
    assert guarded_text == "Review this.\n\ninput"
    assert guarded_model == "gpt-4o"
