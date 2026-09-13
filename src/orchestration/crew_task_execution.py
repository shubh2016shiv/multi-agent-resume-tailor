"""CrewAI task execution primitives for orchestration nodes.

This module is intentionally not named runner: runner.py owns the public
LangGraph pipeline entry point. This file only adapts one CrewAI Agent and one
configured task into one typed Pydantic result.
"""

import threading
import time
from typing import Any

from crewai import Agent, Crew, Process, Task
from pydantic import BaseModel

from src.checkpointing import save_agent_input_checkpoint, save_agent_output_checkpoint
from src.core.llm_cache import configure_llm_cache
from src.core.llm_factory import is_deepseek_model, structured_response_format
from src.core.logger import get_logger
from src.core.settings import get_config, get_tasks_config
from src.orchestration.exceptions import AgentOutputError
from src.tools.llm_gateway.structured_output import add_json_contract

logger = get_logger(__name__)

# What the CLI tells the user when an agent's output cannot be parsed at all.
AGENT_OUTPUT_ERROR_USER_ACTION = (
    "This is usually a transient formatting slip by the model, not a problem with "
    "your resume or job description -- running the pipeline again typically resolves "
    "it. If it keeps happening on the same input, please report it."
)


# Serializes every Crew.kickoff() process-wide. CrewAI 0.134 writes one shared SQLite
# file (latest_kickoff_task_outputs.db, same path for every Crew) with no busy_timeout,
# so two overlapping kickoffs fail instantly with "database is locked" -- and this
# pipeline overlaps them constantly (parallel Stage 1/Stage 3 nodes, plus the experience
# node's own thread pool). We do not own that connection and it cannot be disabled, so
# the only fix we control is to stop the overlap.
#
# Cost: kickoff() includes the LLM call, so this makes every agent call sequential.
# The full reasoning, the cost, and how to recognize the next case of this are in
# orchestration_architecture.md section 9b. Read that before removing this.
_KICKOFF_LOCK = threading.Lock()


def run_agent_task(
    agent: Agent,
    task_name: str,
    context: str,
    output_model: type[BaseModel],
    run_id: str,
) -> Any:
    """Run one CrewAI task and return the validated Pydantic output.

    Precondition: agent is configured, task_name exists in tasks.yaml, and
    context already contains the task-specific input.
    Args:
        agent: configured CrewAI Agent.
        task_name: key in tasks.yaml (e.g. "optimize_skills_section_task").
        context: the task-specific formatted context string.
        output_model: Pydantic model class the agent must return.
        run_id: pipeline run identifier forwarded from state["run_id"];
                used to namespace debug checkpoint files.
    Returns: a validated instance of output_model.
    Raises: AgentOutputError if CrewAI does not produce output_model.
    """
    start_time = time.monotonic()
    configure_llm_cache()
    logger.info(
        "agent_task_started",
        agent_role=agent.role,
        task_name=task_name,
        output_model=output_model.__name__,
        run_id=run_id,
    )
    tasks_config = get_tasks_config()
    task_config = tasks_config.get(task_name, {})
    task_description = task_config.get("description", "") + "\n\nCONTEXT:\n" + context
    task_expected_output = task_config.get("expected_output", "Structured output.")

    # Asking the provider for structured JSON makes it answer in the FIRST response,
    # which skips the tool-call loop entirely -- a tool-carrying agent then returns an
    # empty schema skeleton instead of calling its tools. So tool-carrying agents get
    # no response_format and are steered by the JSON contract in the prompt instead.
    # DeepSeek is excluded too: it does not advertise response_format support to this
    # CrewAI version. Either way the raw output is validated below, which also avoids
    # CrewAI's output_pydantic converter -- it cannot parse our PEP 604 "X | None".
    llm: Any = agent.llm
    task_description = add_json_contract(task_description, output_model, llm.model)
    llm.response_format = (
        None
        if agent.tools or is_deepseek_model(llm.model)
        else structured_response_format(llm.model, output_model)
    )
    task = Task(
        description=task_description,
        expected_output=task_expected_output,
        agent=agent,
    )
    # Save the full task_description to a file before the LLM call so we can
    # inspect exactly what the agent received. No-ops unless DEBUG_CHECKPOINTS=1.
    # SAVE CHECKPOINT INPUT CONTEXT
    checkpoint = save_agent_input_checkpoint(
        run_id=run_id,
        agent_role=agent.role,
        task_name=task_name,
        output_model_name=output_model.__name__,
        task_description=task_description,
    )

    # The only place a Crew is built, so AgentDefaults.verbose here is the single
    # on/off switch for CrewAI's rich-console output across the whole run.
    crew_verbose = get_config().llm.agent_defaults.verbose
    with _KICKOFF_LOCK:  # see the note on _KICKOFF_LOCK above
        result = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=crew_verbose,
        ).kickoff()

    validated = _validate_agent_output(result.raw, output_model, agent.role, task_name)

    # Save the LLM's raw + validated output right after the call resolves.
    # No-ops unless DEBUG_CHECKPOINTS=1.
    # SAVE CHECKPOINT OUTPUT CONTEXT
    save_agent_output_checkpoint(
        checkpoint=checkpoint,
        raw_output=result.raw,
        validated_output=validated,
    )

    duration_ms = round((time.monotonic() - start_time) * 1000)
    logger.info(
        "agent_task_completed",
        agent_role=agent.role,
        task_name=task_name,
        output_model=output_model.__name__,
        run_id=run_id,
        duration_ms=duration_ms,
    )
    return validated


def _validate_agent_output(
    raw_output: str, output_model: type[BaseModel], agent_role: str, task_name: str
) -> Any:
    """Validate an agent's raw text output into output_model.

    Agents are asked to emit JSON, but LLMs often wrap it in ```json fences or a line
    of prose. We take the outermost {...} block and validate that against the model.

    Raises: AgentOutputError if no JSON object is present or it does not satisfy
            output_model -- a formatting/schema failure, not a content quality
            judgment, so it carries different user-facing guidance than
            PipelineQualityGateError.
    """
    json_text = _extract_json_object(raw_output)
    if json_text is None:
        raise AgentOutputError(
            stage=f"{agent_role} ({task_name})",
            reason=f"returned no JSON object: {raw_output[:200]!r}",
            user_action=AGENT_OUTPUT_ERROR_USER_ACTION,
        )
    try:
        return output_model.model_validate_json(json_text)
    except ValueError as error:
        raise AgentOutputError(
            stage=f"{agent_role} ({task_name})",
            reason=f"output did not match {output_model.__name__}: {error}",
            user_action=AGENT_OUTPUT_ERROR_USER_ACTION,
        ) from error


def _extract_json_object(text: str) -> str | None:
    """Return the outermost {...} JSON object in text, ignoring code fences and prose.

    Returns None when no braces are present.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    return text[start : end + 1]
