"""
Generic structured-output harness: call the LLM and get a validated Pydantic
model back. The single choke point where the tools layer calls an LLM.

It enforces the requested schema at the boundary: on a malformed response it
retries once, then raises, so callers never receive malformed data. Both the
review tools (ReviewResult) and the extraction tools (Resume, JobDescription)
build on this one function.
"""

from typing import TypeVar

from crewai import LLM
from pydantic import BaseModel

from src.core.config import get_config
from src.core.logger import get_logger

logger = get_logger(__name__)

OutputModel = TypeVar("OutputModel", bound=BaseModel)


def request_structured_output(
    output_model: type[OutputModel], system_prompt: str, user_content: str
) -> OutputModel:
    """Call the LLM and return a validated instance of output_model.

    Args:
        output_model: The Pydantic model the response must conform to.
        system_prompt: Instructions defining the task (rubric or extraction spec).
        user_content: The text to act on.

    Returns:
        A validated instance of output_model.

    Raises:
        RuntimeError: If the model returns malformed output twice in a row.
    """
    # TODO: Accept an optional model= override so bounded tool calls can use a
    #       cheaper model than the agent's. Deferred: no cost measurement yet.
    structured_llm = _build_structured_llm(output_model)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    # One retry: the model occasionally returns malformed JSON; a second attempt
    # usually succeeds. Beyond that we fail loudly rather than pass bad data on.
    for attempt in range(2):
        raw_output = structured_llm.call(messages)
        try:
            return _parse_into_model(raw_output, output_model)
        except ValueError as parse_error:
            logger.warning(
                f"{output_model.__name__}: malformed output (attempt {attempt + 1}): {parse_error}"
            )
    raise RuntimeError(f"{output_model.__name__}: model returned malformed output twice; aborting")


def _build_structured_llm(output_model: type[BaseModel]) -> LLM:
    """Construct an LLM that returns JSON shaped like output_model, using project config."""
    llm_config = get_config().llm
    return LLM(
        model=llm_config.model,
        temperature=llm_config.temperature,
        response_format=output_model,
    )


def _parse_into_model(raw_output: object, output_model: type[OutputModel]) -> OutputModel:
    """Validate the model's output into output_model.

    Accepts an already-parsed instance, a JSON string, or a dict, since CrewAI's
    response_format handling can return any of these. Raises ValueError on
    anything that does not satisfy the schema.
    """
    if isinstance(raw_output, output_model):
        return raw_output
    if isinstance(raw_output, str):
        return output_model.model_validate_json(raw_output)
    if isinstance(raw_output, dict):
        return output_model.model_validate(raw_output)
    raise ValueError(f"unexpected output type: {type(raw_output).__name__}")
