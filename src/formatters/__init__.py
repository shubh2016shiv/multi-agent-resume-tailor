"""Public formatter entrypoints: turn pipeline objects into per-agent LLM context.

Each orchestration node hands its large domain objects (Resume, JobDescription,
AlignmentStrategy, agent outputs) to one `format_*_context(...)` function here,
and gets back one compact context string tailored to exactly what that agent
needs -- nothing more, so no token budget is spent on fields the agent ignores.

Shared shape (every formatter module follows the same three layers):
    select_*(...)  -> keep only the fields this agent needs, as a plain dict
    build_*(...)   -> assemble those dict slices into one labelled payload
    format_*(...)  -> render the payload via src/formatters/llm_context_rendering

Only the `format_*` layer is the public surface; `select_*`/`build_*` are the
building blocks each formatter composes internally.

SCOPE BOUNDARY -- read before assuming this facade is the only way in:
Most orchestration nodes today import the formatter they need directly from its
submodule (e.g. `from src.formatters.skills_optimizer_formatter import ...`)
rather than through this package. This `__init__` is the *intended* public
surface and lists every public entrypoint; it is not yet the *enforced* one.
Prefer importing from here in new code.
"""

from src.formatters.ats_optimization_formatter import format_ats_optimization_context
from src.formatters.experience_optimizer_formatter import format_experience_optimizer_context
from src.formatters.gap_analysis_formatter import format_gap_analysis_context
from src.formatters.professional_summary_formatter import format_professional_summary_context
from src.formatters.quality_feedback_formatter import format_quality_feedback_context

# Two entrypoints from the skills formatter: the STEP 2 initial optimization and
# the STEP 6 scoped rewrite/correction pass. Both are public.
from src.formatters.skills_optimizer_formatter import (
    format_skills_optimizer_context,
    format_skills_rewrite_context,
)

__all__ = [
    "format_ats_optimization_context",
    "format_experience_optimizer_context",
    "format_gap_analysis_context",
    "format_professional_summary_context",
    "format_quality_feedback_context",
    "format_skills_optimizer_context",
    "format_skills_rewrite_context",
]
