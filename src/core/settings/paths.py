"""Filesystem locations for runtime settings and YAML catalogs.

Supports Pattern 1 (separate the data from the loader): these constants point
the loader code at where the config *data* lives. See CONFIGURATION_PATTERNS.md.
"""

from pathlib import Path

ROOT_MARKER_FILE = "pyproject.toml"


def _find_project_root(start: Path, marker: str = ROOT_MARKER_FILE) -> Path:
    """Walk upward from `start` until a directory containing `marker` is found.

    Locating the project root by counting a fixed number of `.parent` hops from
    this file's own location (e.g. `Path(__file__).parents[2]`) is fragile: if
    this file is ever moved to a different directory depth, the hop count is
    silently wrong -- it still returns *a* path, just not the right one, and
    nothing fails until something downstream can't find its files. Walking up
    to a stable marker at the true project root is robust to this file moving.

    Raises:
        RuntimeError: if no directory from `start` up to the filesystem root
            contains `marker` -- fail fast with a clear cause instead of
            silently resolving to the wrong directory.
    """
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / marker).exists():
            return candidate
    raise RuntimeError(
        f"Could not locate the project root: no {marker!r} found in any "
        f"parent directory of {start}."
    )


# Project root, located by walking up to the directory containing pyproject.toml.
PROJECT_ROOT = _find_project_root(Path(__file__))

# `src/` root.
SRC_ROOT = PROJECT_ROOT / "src"

# Runtime configuration files.
SETTINGS_YAML_PATH = SRC_ROOT / "config" / "settings.yaml"
AGENTS_CONFIG_DIR = SRC_ROOT / "config" / "agents"
TASKS_CONFIG_DIR = SRC_ROOT / "config" / "tasks"
