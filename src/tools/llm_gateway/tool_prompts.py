"""Loader for tool-layer prompt files."""

from functools import cache
from pathlib import Path

from src.core.settings.paths import SRC_ROOT

_TOOL_PROMPTS_DIR = SRC_ROOT / "config" / "tool_prompts"


@cache
def load_tool_prompt(relative_path: str) -> str:
    """Load a tool prompt from src/config/tool_prompts.

    Expects a relative POSIX-style path such as
    'resume_diagnostics/language_quality.md'.
    Returns the prompt text with surrounding whitespace removed.
    Raises ValueError if the path escapes the prompt directory.
    """
    prompt_path = (_TOOL_PROMPTS_DIR / relative_path).resolve()
    if not _is_relative_to(prompt_path, _TOOL_PROMPTS_DIR.resolve()):
        raise ValueError(f"Tool prompt path escapes prompt directory: {relative_path}")
    return prompt_path.read_text(encoding="utf-8").strip()


def _is_relative_to(path: Path, parent: Path) -> bool:
    """Return True when path is inside parent."""
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True
