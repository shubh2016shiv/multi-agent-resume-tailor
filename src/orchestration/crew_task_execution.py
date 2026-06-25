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

from src.core.llm_cache import configure_llm_cache
from src.core.logger import get_logger
from src.core.settings import get_tasks_config

logger = get_logger(__name__)

# -----------------------------------------------------------------------------
# WHY A THREAD LOCK LIVES HERE  (read before removing it -- it is not decoration)
#
# WHAT A LOCK IS FOR (the general principle)
#   A threading.Lock serializes access to a SHARED RESOURCE that cannot tolerate two
#   threads touching it at once. You need one when ALL THREE of these are true:
#     1. Concurrency: two+ threads can run the same code at the same time.
#     2. Shared mutable state: they touch one resource that is not their own private
#        copy -- a file, a DB connection/file, a global counter, a network handle.
#     3. The resource is not internally synchronized: it does not protect itself, so
#        overlapping access corrupts it, loses writes, or errors out.
#   If any one is false you do NOT need a lock: no concurrency (single thread), or
#   each thread has its own copy (no sharing), or the resource is already thread-safe
#   (e.g. an OS pipe, a queue.Queue, a DB with proper locking + busy-timeout).
#
# WHY IT WAS REQUIRED HERE (the specific trigger)
#   - Concurrency: the LangGraph graph fans out -- Stage 1 (extract + analyze) and
#     Stage 3 (summary + experience + skills) run nodes in parallel THREADS, and the
#     experience node adds its own ThreadPoolExecutor. Many run_agent_task() at once.
#   - Shared state: every CrewAI kickoff() writes ONE shared SQLite file
#     (latest_kickoff_task_outputs.db) -- the same path for every Crew in the process.
#   - Not synchronized: CrewAI 0.134 opens that file with sqlite3.connect() and no
#     busy_timeout, so the moment two writers overlap, one fails IMMEDIATELY with
#     "database is locked" instead of waiting its turn.
#   All three were true -> a real e2e run crashed in the experience stage. CrewAI gives
#   no way to disable the store or make it tolerant, and we do not own its connection,
#   so the only fix WE control is to stop the overlap: serialize kickoff() process-wide.
#
# WHERE TO APPLY THIS (how to recognize the next one -- you cannot know preemptively,
# but you can know the SHAPE)
#   Reach for a lock when concurrent code funnels through a single un-synchronized
#   shared resource: a library that writes a fixed file/db per process, a module-level
#   mutable global (dict/list/counter) mutated from threads, a non-thread-safe client
#   reused across threads. The tell at debug time is an error that is INTERMITTENT and
#   load-dependent ("locked", "busy", corrupted/lost writes, a counter that is wrong
#   only under load) -- the signature of a race, not a logic bug.
#   PREFER avoiding the lock when you can: give each thread its OWN resource (separate
#   file/dir/connection), use an already-thread-safe primitive (queue.Queue), or push
#   the work to separate processes. A lock is the right tool only when the shared,
#   un-synchronized resource is fixed by a dependency you do not control -- exactly
#   this case. Keep the critical section as SMALL as correctness allows; here it must
#   wrap the whole kickoff() because that is where CrewAI does the hidden write.
# -----------------------------------------------------------------------------
_KICKOFF_LOCK = threading.Lock()


def run_agent_task(
    agent: Agent,
    task_name: str,
    context: str,
    output_model: type[BaseModel],
) -> Any:
    """Run one CrewAI task and return the validated Pydantic output.

    Precondition: agent is configured, task_name exists in tasks.yaml, and
    context already contains the task-specific input.
    Returns: a validated instance of output_model.
    Raises: ValueError if CrewAI does not produce output_model.
    """
    start_time = time.monotonic()
    configure_llm_cache()
    logger.info(
        "agent_task_started",
        agent_role=agent.role,
        task_name=task_name,
        output_model=output_model.__name__,
    )
    tasks_config = get_tasks_config()
    task_config = tasks_config.get(task_name, {})
    task_description = task_config.get("description", "") + "\n\nCONTEXT:\n" + context
    task_expected_output = task_config.get("expected_output", "Structured output.")

    # Tool-using agents must call tools before they can produce the output. Setting
    # response_format on the LLM instructs the provider to return structured JSON in
    # the FIRST response, which bypasses the tool-call loop entirely -- the model
    # skips its tools and returns an empty schema skeleton instead. So for agents that
    # carry tools we leave response_format unset and let the tool chain run; we still
    # validate the final raw output ourselves below (same path, no output_pydantic
    # coercion that crashes on our PEP 604 "X | None" fields).
    # For tool-free agents, response_format is safe and guarantees well-formed JSON
    # on the first try without any coercion retry.
    if agent.tools:
        agent.llm.response_format = None
        # output_pydantic constrains the agent to return a JSON object matching
        # output_model, the same way the trigger scripts do. Without this, the agent
        # can decide to return a prose error string instead of JSON (e.g. when it sees
        # a quality-check WARNING and chooses to stop). We still validate result.raw
        # ourselves below so we never hit CrewAI's PEP 604 schema-parser path.
        task = Task(
            description=task_description,
            expected_output=task_expected_output,
            agent=agent,
            output_pydantic=output_model,
        )
    else:
        agent.llm.response_format = output_model
        task = Task(
            description=task_description,
            expected_output=task_expected_output,
            agent=agent,
        )
    # Serialize the kickoff so concurrent pipeline nodes never write CrewAI's shared
    # SQLite store at the same time (see _KICKOFF_LOCK above).
    with _KICKOFF_LOCK:
        result = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True,
        ).kickoff()

    validated = _validate_agent_output(result.raw, output_model, agent.role)
    duration_ms = round((time.monotonic() - start_time) * 1000)
    logger.info(
        "agent_task_completed",
        agent_role=agent.role,
        task_name=task_name,
        output_model=output_model.__name__,
        duration_ms=duration_ms,
    )
    return validated


def _validate_agent_output(raw_output: str, output_model: type[BaseModel], agent_role: str) -> Any:
    """Validate an agent's raw text output into output_model.

    Agents are asked to emit JSON, but LLMs often wrap it in ```json fences or a line
    of prose. We take the outermost {...} block and validate that against the model.

    Raises: ValueError if no JSON object is present or it does not satisfy output_model.
    """
    json_text = _extract_json_object(raw_output)
    if json_text is None:
        raise ValueError(f"Agent {agent_role} returned no JSON object: {raw_output[:200]!r}")
    try:
        return output_model.model_validate_json(json_text)
    except ValueError as error:
        raise ValueError(
            f"Agent {agent_role} output did not match {output_model.__name__}: {error}"
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
