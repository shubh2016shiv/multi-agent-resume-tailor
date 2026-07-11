"""YAML loading and validation for project configuration.

This module intentionally uses stdlib logging instead of `src.core.logger`.
It sits on the bootstrap path for settings resolution, so depending on the
configured application logger here would create a circular "settings need
logger, logger needs settings" dependency.

This is a concrete instance of Pattern 9 (break the bootstrap dependency cycle)
— the lowest layer stays dependency-free. See CONFIGURATION_PATTERNS.md.
"""

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
    ####################################################
    # STEP 1: TREAT A MISSING FILE AS "NO OVERRIDES"#
    ####################################################
    # The higher-level settings/catalog loaders decide whether a missing file
    # is acceptable. This low-level helper only says "nothing was found here."
    if not path.exists():
        return {}

    ####################################################
    # STEP 2: LOAD THE YAML DOCUMENT FROM DISK
    ####################################################
    try:
        with path.open(encoding="utf-8") as yaml_file:
            data = yaml.safe_load(yaml_file) or {}
    except (OSError, yaml.YAMLError) as exc:
        message = f"Could not load {label} configuration at {path}: {exc}"
        logger.warning(message)
        raise ConfigurationError(message) from exc

    ####################################################
    # STEP 3: REQUIRE A TOP-LEVEL YAML MAPPING#
    ####################################################
    # All config entry points in this project expect key/value mappings, not
    # YAML lists or scalar values.
    if not isinstance(data, dict):
        message = f"{label} configuration at {path} must be a YAML mapping"
        logger.warning(message)
        raise ConfigurationError(message)

    return data


def yaml_config_settings_source() -> dict[str, Any]:
    """Load settings.yaml as a Pydantic settings source."""
    ####################################################
    # STEP 1: REUSE THE SHARED YAML-MAPPING LOADER FOR settings.yaml#
    ####################################################
    return read_yaml_mapping(SETTINGS_YAML_PATH, "settings")
