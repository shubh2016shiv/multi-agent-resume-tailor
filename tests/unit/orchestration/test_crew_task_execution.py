"""Contracts for the CrewAI task-execution adapter.

Two behaviours are pinned here before crew_task_execution.py is simplified:

  * JSON extraction and validation, including which failure raises
    AgentOutputError (a formatting failure the user should retry) rather than
    a quality-gate error.
  * The response_format branch. This is the subtle one: a tool-carrying agent
    must NOT get a structured response_format, because that makes the provider
    answer in the first turn and skip the tool-call loop entirely, returning an
    empty schema skeleton. The planned refactor collapses the duplicated
    Task(...) construction in both branches, so the branch's actual effect
    needs a test that does not depend on how the Task is built.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from src.orchestration.crew_task_execution import (
    _extract_json_object,
    _validate_agent_output,
    run_agent_task,
)
from src.orchestration.exceptions import AgentOutputError

MODULE = "src.orchestration.crew_task_execution"


class _Output(BaseModel):
    name: str
    count: int


# --- JSON extraction ---------------------------------------------------------


def test_plain_json_object_is_extracted() -> None:
    assert _extract_json_object('{"name": "a"}') == '{"name": "a"}'


def test_fenced_json_is_extracted_without_the_fence() -> None:
    """LLMs commonly wrap JSON in a ```json fence; the braces are what matter."""
    raw = 'Here you go:\n```json\n{"name": "a", "count": 1}\n```\nHope that helps!'

    assert _extract_json_object(raw) == '{"name": "a", "count": 1}'


def test_nested_object_keeps_the_outermost_braces() -> None:
    raw = 'prose {"outer": {"inner": 1}} more prose'

    assert _extract_json_object(raw) == '{"outer": {"inner": 1}}'


def test_text_without_braces_extracts_nothing() -> None:
    assert _extract_json_object("I could not complete this task.") is None


def test_closing_brace_before_opening_brace_extracts_nothing() -> None:
    assert _extract_json_object("} {") is None


# --- validation --------------------------------------------------------------


def test_valid_output_is_returned_as_the_typed_model() -> None:
    validated = _validate_agent_output('{"name": "a", "count": 1}', _Output, "Role", "task")

    assert isinstance(validated, _Output)
    assert validated.name == "a"
    assert validated.count == 1


def test_output_without_json_raises_agent_output_error() -> None:
    """A formatting failure, not a content judgment -- so retry is the right advice."""
    with pytest.raises(AgentOutputError) as error:
        _validate_agent_output("no json here", _Output, "Role", "task")

    assert error.value.stage == "Role (task)"
    assert "no JSON object" in error.value.reason
    assert "running the pipeline again" in error.value.user_action


def test_output_that_does_not_match_the_schema_raises_agent_output_error() -> None:
    with pytest.raises(AgentOutputError) as error:
        _validate_agent_output('{"name": "a"}', _Output, "Role", "task")

    assert "did not match _Output" in error.value.reason


# --- the response_format branch ---------------------------------------------


def _agent(*, tools: list[Any]) -> MagicMock:
    agent = MagicMock()
    agent.role = "Test Role"
    agent.tools = tools
    agent.llm = MagicMock()
    agent.llm.model = "openai/gpt-4.1"
    return agent


def _run(agent: MagicMock, *, raw: str = '{"name": "a", "count": 1}') -> Any:
    """Run run_agent_task with every external boundary stubbed out."""
    crew = MagicMock()
    crew.return_value.kickoff.return_value = MagicMock(raw=raw)
    with (
        patch(f"{MODULE}.Crew", crew),
        # Task is a Pydantic model that rejects a MagicMock agent, and how the Task
        # is constructed is exactly what the refactor is free to change -- so these
        # tests assert on the response_format effect, not on the Task.
        patch(f"{MODULE}.Task"),
        patch(f"{MODULE}.configure_llm_cache"),
        patch(f"{MODULE}.get_tasks_config", return_value={}),
        patch(f"{MODULE}.get_config"),
        patch(f"{MODULE}.save_agent_input_checkpoint", return_value=None),
        patch(f"{MODULE}.save_agent_output_checkpoint"),
        patch(
            f"{MODULE}.add_json_contract", side_effect=lambda description, _model, _m: description
        ),
        patch(f"{MODULE}.structured_response_format", return_value={"type": "json_schema"}),
    ):
        return run_agent_task(
            agent=agent,
            task_name="some_task",
            context="CONTEXT",
            output_model=_Output,
            run_id="test-run",
        )


def test_a_tool_carrying_agent_gets_no_structured_response_format() -> None:
    """Setting response_format here would bypass the tool-call loop entirely."""
    agent = _agent(tools=[MagicMock()])

    result = _run(agent)

    assert agent.llm.response_format is None
    assert isinstance(result, _Output)


def test_a_toolless_agent_gets_the_structured_response_format() -> None:
    """With no tools to call, the provider can be asked for JSON in the first turn."""
    agent = _agent(tools=[])

    result = _run(agent)

    assert agent.llm.response_format == {"type": "json_schema"}
    assert isinstance(result, _Output)


def test_a_toolless_deepseek_agent_gets_its_provider_response_format() -> None:
    """DeepSeek JSON mode must not be disabled for a tool-free structured task."""
    agent = _agent(tools=[])
    agent.llm.model = "deepseek/deepseek-chat"

    result = _run(agent)

    assert agent.llm.response_format == {"type": "json_schema"}
    assert isinstance(result, _Output)


def test_unparseable_agent_output_raises_agent_output_error() -> None:
    """The raw output is validated by this module regardless of which branch ran."""
    with pytest.raises(AgentOutputError):
        _run(_agent(tools=[]), raw="the model apologised instead of answering")
