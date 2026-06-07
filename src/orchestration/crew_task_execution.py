"""CrewAI task execution primitives for orchestration nodes.

This module is intentionally not named runner: runner.py owns the public
LangGraph pipeline entry point. This file only adapts one CrewAI Agent and one
configured task into one typed Pydantic result.
"""

from typing import Any

from crewai import Agent, Crew, Process, Task
from pydantic import BaseModel

from src.core.config import get_tasks_config


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
    result = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    ).kickoff()

    if hasattr(result, "pydantic") and result.pydantic:
        return result.pydantic
    raise ValueError(f"Agent {agent.role} did not return a valid {output_model.__name__} object.")
