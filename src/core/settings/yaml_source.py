"""YAML loading and validation for project configuration."""

import logging
from pathlib import Path
from typing import Any

import yaml

from src.core.settings.exceptions import ConfigurationError
from src.core.settings.paths import SETTINGS_YAML_PATH

logger = logging.getLogger(__name__)


def read_yaml_mapping(path: Path, label: str) -> dict[str, Any]:
    """Load a YAML mapping from disk.

    Args:
        path: YAML file path to load.
        label: Human-readable configuration type for error messages.

    Returns:
        Parsed YAML mapping, or an empty mapping when the file is absent.

    Raises:
        ConfigurationError: If the YAML is invalid or not a mapping.
    """
    if not path.exists():
        return {}

    try:
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as exc:
        message = f"Could not load {label} configuration at {path}: {exc}"
        logger.warning(message)
        raise ConfigurationError(message) from exc

    if not isinstance(data, dict):
        message = f"{label} configuration at {path} must be a YAML mapping"
        logger.warning(message)
        raise ConfigurationError(message)

    return data


def yaml_config_settings_source() -> dict[str, Any]:
    """Load settings.yaml as a Pydantic settings source."""
    return read_yaml_mapping(SETTINGS_YAML_PATH, "settings")
