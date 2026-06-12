"""Filesystem locations for runtime settings and YAML catalogs."""

from pathlib import Path

# `src/` root. This file lives at `src/core/settings/paths.py`.
SRC_ROOT = Path(__file__).resolve().parents[2]

# Project root.
PROJECT_ROOT = SRC_ROOT.parent

# Runtime configuration files.
SETTINGS_YAML_PATH = SRC_ROOT / "config" / "settings.yaml"
AGENTS_CONFIG_DIR = SRC_ROOT / "config" / "agents"
TASKS_CONFIG_DIR = SRC_ROOT / "config" / "tasks"
