"""CrewAI task execution primitives for orchestration nodes.

This module is intentionally not named runner: runner.py owns the public
LangGraph pipeline entry point. This file only adapts one CrewAI Agent and one
configured task into one typed Pydantic result.
"""

import threading
from typing import Any

from crewai import Agent, Crew, Process, Task
from pydantic import BaseModel

from src.core.settings import get_tasks_config

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
    tasks_config = get_tasks_config()
    task_config = tasks_config.get(task_name, {})
    task_description = task_config.get("description", "") + "\n\nCONTEXT:\n" + context
    task_expected_output = task_config.get("expected_output", "Structured output.")

    task = Task(
        description=task_description,
        expected_output=task_expected_output,
        agent=agent,
        output_pydantic=output_model,
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

    if hasattr(result, "pydantic") and result.pydantic:
        return result.pydantic
    raise ValueError(f"Agent {agent.role} did not return a valid {output_model.__name__} object.")
