"""Accessors for CrewAI YAML registries."""

from functools import lru_cache
from typing import Any

from src.core.settings.paths import AGENTS_YAML_PATH, TASKS_YAML_PATH
from src.core.settings.yaml_source import read_yaml_mapping


@lru_cache
def get_agents_config() -> dict[str, Any]:
    """Load and return agent definitions from agents.yaml."""
    if not AGENTS_YAML_PATH.exists():
        raise FileNotFoundError(f"Agent configuration not found at: {AGENTS_YAML_PATH}")
    return read_yaml_mapping(AGENTS_YAML_PATH, "agent")


@lru_cache
def get_tasks_config() -> dict[str, Any]:
    """Load and return task definitions from tasks.yaml."""
    if not TASKS_YAML_PATH.exists():
        raise FileNotFoundError(f"Task configuration not found at: {TASKS_YAML_PATH}")
    return read_yaml_mapping(TASKS_YAML_PATH, "task")
