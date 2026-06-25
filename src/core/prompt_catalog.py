"""Centralized loading of application-owned prompt files."""

from functools import cache
from pathlib import Path

from src.core.settings import get_config
from src.core.settings.paths import PROJECT_ROOT


@cache
def load_tool_prompt(relative_path: str) -> str:
    """Load a tool prompt from the centralized tool prompt catalog.

    Expects a relative POSIX-style path such as
    'resume_diagnostics/language_quality.md'.
    Returns the prompt text with surrounding whitespace removed.
    Raises ValueError if the path escapes the configured prompt directory.
    """
    ####################################################
    # STEP 1: RESOLVE THE CONFIGURED PROMPT ROOT FROM APP SETTINGS#
    ####################################################
    configured_root = Path(get_config().prompt_catalog.tool_prompts_dir)
    if not configured_root.is_absolute():
        configured_root = PROJECT_ROOT / configured_root
    prompt_root = configured_root.resolve()

    ####################################################
    # STEP 2: RESOLVE THE REQUESTED PROMPT PATH UNDER THAT ROOT#
    ####################################################
    prompt_path = (prompt_root / relative_path).resolve()

    ####################################################
    # STEP 3: REJECT ANY PATH THAT ESCAPES THE CONFIGURED ROOT#
    ####################################################
    if not is_path_inside_directory(prompt_path, prompt_root):
        raise ValueError(f"Tool prompt path escapes prompt directory: {relative_path}")

    ####################################################
    # STEP 4: READ AND RETURN THE PROMPT TEXT#
    ####################################################
    return prompt_path.read_text(encoding="utf-8").strip()


def is_path_inside_directory(path: Path, parent_directory: Path) -> bool:
    """Return True when path is inside parent_directory."""
    try:
        path.relative_to(parent_directory)
    except ValueError:
        return False
    return True
