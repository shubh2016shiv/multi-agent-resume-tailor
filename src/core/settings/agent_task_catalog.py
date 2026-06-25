"""Accessors for CrewAI YAML registries."""

from functools import lru_cache
from typing import Any

from src.core.settings.paths import AGENTS_CONFIG_DIR, TASKS_CONFIG_DIR
from src.core.settings.yaml_source import read_yaml_mapping


@lru_cache
def get_agents_config() -> dict[str, Any]:
    """Load and merge all per-agent YAML files from src/config/agents/.

    Each file in the directory contributes one top-level key (the agent name).
    Raises FileNotFoundError if the directory does not exist.
    Raises ConfigurationError if any individual YAML is malformed.
    """
    ####################################################
    # STEP 1: REQUIRE THE AGENT CATALOG DIRECTORY TO EXIST#
    ####################################################
    if not AGENTS_CONFIG_DIR.exists():
        raise FileNotFoundError(f"Agent configuration directory not found at: {AGENTS_CONFIG_DIR}")

    ####################################################
    # STEP 2: MERGE EACH YAML FILE INTO ONE IN-MEMORY CATALOG#
    ####################################################
    # Sorting makes the merge order deterministic for debugging and tests.
    merged: dict[str, Any] = {}
    for yaml_file in sorted(AGENTS_CONFIG_DIR.glob("*.yaml")):
        merged.update(read_yaml_mapping(yaml_file, f"agent[{yaml_file.stem}]"))

    ####################################################
    # STEP 3: CACHE AND RETURN THE MERGED AGENT CATALOG#
    ####################################################
    return merged


@lru_cache
def get_tasks_config() -> dict[str, Any]:
    """Load and merge all per-stage task YAML files from src/config/tasks/.

    Each file in the directory contributes one or more top-level task keys.
    Raises FileNotFoundError if the directory does not exist.
    Raises ConfigurationError if any individual YAML is malformed.
    """
    ####################################################
    # STEP 1: REQUIRE THE TASK CATALOG DIRECTORY TO EXIST#
    ####################################################
    if not TASKS_CONFIG_DIR.exists():
        raise FileNotFoundError(f"Task configuration directory not found at: {TASKS_CONFIG_DIR}")

    ####################################################
    # STEP 2: MERGE EACH YAML FILE INTO ONE IN-MEMORY CATALOG#
    ####################################################
    merged: dict[str, Any] = {}
    for yaml_file in sorted(TASKS_CONFIG_DIR.glob("*.yaml")):
        merged.update(read_yaml_mapping(yaml_file, f"task[{yaml_file.stem}]"))

    ####################################################
    # STEP 3: CACHE AND RETURN THE MERGED TASK CATALOG#
    ####################################################
    return merged
