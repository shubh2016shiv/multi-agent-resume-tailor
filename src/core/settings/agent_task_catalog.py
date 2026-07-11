"""Accessors for CrewAI YAML registries.

These are the "declarative catalog" half of Pattern 10 (name the two kinds of
config) — open-ended directories of YAML entries merged into a dict, as opposed
to the typed Settings object in runtime.py. Each accessor is also Pattern 6
(resolve once) via `@lru_cache`. See CONFIGURATION_PATTERNS.md.

get_agents_config() and get_tasks_config() are two thin, independently-cached
wrappers around the same merge logic, factored into _merge_yaml_directory()
below. They stay as two separate public functions (rather than one
parameterized one) because each needs its own `@lru_cache` identity — tests
call `get_agents_config.cache_clear()` and `get_tasks_config.cache_clear()`
independently, which only works if each is its own decorated function.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

from src.core.settings.paths import AGENTS_CONFIG_DIR, TASKS_CONFIG_DIR
from src.core.settings.yaml_source import read_yaml_mapping


def _merge_yaml_directory(directory: Path, label_prefix: str) -> dict[str, Any]:
    """Merge every *.yaml file in `directory` into one dict, sorted for determinism.

    Shared by get_agents_config() and get_tasks_config() below — the only
    difference between the two is which directory and which error label
    prefix they use.

    Raises:
        FileNotFoundError: If `directory` does not exist.
        ConfigurationError: If any individual YAML file is malformed
            (raised by read_yaml_mapping).
    """
    if not directory.exists():
        raise FileNotFoundError(f"{label_prefix} configuration directory not found at: {directory}")

    # Sorting makes the merge order deterministic for debugging and tests.
    merged: dict[str, Any] = {}
    for yaml_file in sorted(directory.glob("*.yaml")):
        merged.update(read_yaml_mapping(yaml_file, f"{label_prefix.lower()}[{yaml_file.stem}]"))
    return merged


@lru_cache
def get_agents_config() -> dict[str, Any]:
    """Load and merge all per-agent YAML files from src/config/agents/.

    Each file in the directory contributes one top-level key (the agent name).
    Raises FileNotFoundError if the directory does not exist.
    Raises ConfigurationError if any individual YAML is malformed.
    """
    return _merge_yaml_directory(AGENTS_CONFIG_DIR, "Agent")


@lru_cache
def get_tasks_config() -> dict[str, Any]:
    """Load and merge all per-stage task YAML files from src/config/tasks/.

    Each file in the directory contributes one or more top-level task keys.
    Raises FileNotFoundError if the directory does not exist.
    Raises ConfigurationError if any individual YAML is malformed.
    """
    return _merge_yaml_directory(TASKS_CONFIG_DIR, "Task")
