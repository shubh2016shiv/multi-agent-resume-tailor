"""Centralized loading of application-owned prompt files.

Every tool-engine module (src/tools/engines/**) calls load_tool_prompt() at
IMPORT TIME to assign the result to a module-level constant, e.g.:

    REQUIREMENTS_RUBRIC = load_tool_prompt("job_matching/requirements_matcher.md")

That constant is then reused across every call the engine makes. Combined with
the @cache below, this means a prompt file's contents are read from disk at
most once per process and NEVER re-read afterward. Editing a .md file under the
configured prompt directory has NO EFFECT on an already-running process — the
process must restart to pick up the change. There is no hot-reload here.

See src/core/settings/README.md ("change a tool prompt") for the full chain
from settings.yaml to the actual LLM call.
"""

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

    Cached for the life of the process (see module docstring) — call this at
    module import time to assign a module-level constant, not inside a
    per-request/per-call function body.
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
