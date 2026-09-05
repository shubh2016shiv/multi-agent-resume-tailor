"""
Generic structured-output harness: call the LLM and get a validated Pydantic
model back. The single choke point where the tools layer calls an LLM.

It enforces the requested schema at the boundary: on a malformed response it
retries once, then raises, so callers never receive malformed data. Both the
review tools (ReviewResult) and the extraction tools (Resume, JobDescription)
build on this one function.
"""

import json
from typing import Any

from crewai import LLM
from pydantic import BaseModel

from src.core.llm_cache import configure_llm_cache
from src.core.llm_factory import build_llm, is_deepseek_model
from src.core.llm_token_tracker import ensure_token_budget
from src.core.logger import get_logger
from src.core.prompt_catalog import load_tool_prompt
from src.core.resiliency import resilient_llm_call
from src.core.settings import get_config

logger = get_logger(__name__)
JSON_CONTRACT_PROMPT = load_tool_prompt("structured_output/json_contract.md")


def build_structured_llm(output_model: type[BaseModel], temperature: float | None = None) -> LLM:
    """Construct an LLM that returns JSON shaped like output_model.

    Uses the configured default temperature unless an explicit temperature is given
    (e.g. 0.0 for deterministic gate decisions like the entailment judge).
    """
    llm_config = get_config().llm
    overrides: dict[str, Any] = {"model": llm_config.structured_model}
    if temperature is not None:
        overrides["temperature"] = temperature
    return build_llm(overrides, output_model)


def request_structured_output[OutputModel: BaseModel](
    output_model: type[OutputModel],
    system_prompt: str,
    user_content: str,
    temperature: float | None = None,
) -> OutputModel:
    """Call the LLM and return a validated instance of output_model.

    Args:
        output_model: The Pydantic model the response must conform to.
        system_prompt: Instructions defining the task (rubric or extraction spec).
        user_content: The text to act on.
        temperature: Optional override; pass 0.0 for deterministic gate decisions.
            Defaults to the configured agent temperature when None.

    Returns:
        A validated instance of output_model.

    Raises:
        TokenBudgetExceeded: If the combined system and user input exceeds the configured budget.
        RuntimeError: If the model returns malformed output twice in a row.
    """
    configure_llm_cache()
    # TODO: Accept an optional model= override so bounded tool calls can use a
    #       cheaper model than the agent's. Deferred: no cost measurement yet.
    llm_config = get_config().llm
    structured_model = llm_config.structured_model
    contracted_prompt = add_json_contract(system_prompt, output_model, structured_model)

    ####################################################
    # STEP 1: ENFORCE THE INPUT TOKEN BUDGET BEFORE ANY PROVIDER CALL#
    ####################################################
    ensure_token_budget(
        f"{contracted_prompt}\n\n{user_content}",
        structured_model,
        llm_config.structured_input_token_budget,
    )

    ####################################################
    # STEP 2: HAND THE VALIDATED INPUT TO THE RESILIENT CALL PATH#
    ####################################################
    return _request_structured_output(output_model, contracted_prompt, user_content, temperature)


def add_json_contract(system_prompt: str, output_model: type[BaseModel], model: str) -> str:
    """Append a schema instruction when DeepSeek JSON-object mode is used.

    Public: also used by `crew_task_execution.run_agent_task` so every CrewAI agent --
    not just this module's direct gateway calls -- tells the model the exact field
    names its target Pydantic model expects, instead of leaving the model to guess
    them from prose alone.
    """
    if not is_deepseek_model(model):
        return system_prompt
    schema = json.dumps(output_model.model_json_schema(), separators=(",", ":"))
    return f"{system_prompt}\n\n{JSON_CONTRACT_PROMPT.format(json_schema=schema)}"


@resilient_llm_call()
def _request_structured_output[OutputModel: BaseModel](
    output_model: type[OutputModel],
    system_prompt: str,
    user_content: str,
    temperature: float | None = None,
) -> OutputModel:
    """Call the configured LLM after input validation has already passed."""
    ####################################################
    # STEP 1: BUILD THE STRUCTURED LLM AND MESSAGE PAYLOAD#
    ####################################################
    structured_llm = build_structured_llm(output_model, temperature)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    ####################################################
    # STEP 2: RETRY ON MALFORMED STRUCTURED OUTPUT ONE TIME#
    ####################################################
    # The model occasionally returns malformed JSON; one extra attempt
    # usually succeeds. Beyond that we fail loudly.
    for attempt in range(2):
        raw_output = structured_llm.call(messages)
        try:
            return parse_structured_output(raw_output, output_model)
        except ValueError as parse_error:
            logger.warning(
                f"{output_model.__name__}: malformed output (attempt {attempt + 1}): {parse_error}"
            )
    raise RuntimeError(f"{output_model.__name__}: model returned malformed output twice; aborting")


def parse_structured_output[OutputModel: BaseModel](
    raw_output: object, output_model: type[OutputModel]
) -> OutputModel:
    """Validate the model's output into output_model.

    Accepts an already-parsed instance, a JSON string, or a dict, since CrewAI's
    response_format handling can return any of these. Raises ValueError on
    anything that does not satisfy the schema.
    """
    ####################################################
    # STEP 1: ACCEPT AN ALREADY-VALID MODEL INSTANCE#
    ####################################################
    if isinstance(raw_output, output_model):
        return raw_output

    ####################################################
    # STEP 2: ACCEPT JSON STRINGS OR PLAIN DICTS FROM CREWAI#
    ####################################################
    if isinstance(raw_output, str):
        return output_model.model_validate_json(raw_output)
    if isinstance(raw_output, dict):
        return output_model.model_validate(raw_output)

    ####################################################
    # STEP 3: REJECT ANY UNEXPECTED OUTPUT TYPE#
    ####################################################
    raise ValueError(f"unexpected output type: {type(raw_output).__name__}")
