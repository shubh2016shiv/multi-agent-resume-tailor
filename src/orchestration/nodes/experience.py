"""Stage 3 professional experience prioritization flow.

The LLM may propose only an ordering of existing bullets. Code accepts the proposal
only when it is an exact permutation of the source bullets, then rebuilds the role
from the original typed object. New or altered claims can never enter pipeline state.
"""

import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from src.agents.professional_experience import create_professional_experience_agent
from src.agents.professional_experience.models import OptimizedExperienceSection
from src.core.logger import get_logger
from src.data_models.job import JobDescription
from src.data_models.resume import Experience, Resume
from src.data_models.strategy import AlignmentStrategy
from src.formatters.experience_optimizer_formatter import format_experience_optimizer_context
from src.orchestration.crew_task_execution import run_agent_task
from src.orchestration.state import ResumeEnhancementPipelineState

logger = get_logger(__name__)


def optimize_experience(state: ResumeEnhancementPipelineState) -> dict:
    """Prioritize existing work-experience bullets with one role-scoped call per entry.

    Reads: resume, job_description, and alignment_strategy.
    Writes: optimized_experience.
    Returns: partial state with merged OptimizedExperienceSection output.
    """
    start_time = time.monotonic()
    resume = state["resume"]
    job_description = state["job_description"]
    strategy = state["alignment_strategy"]
    logger.info(
        "pipeline_stage_started",
        stage="optimize_experience",
        run_id=state["run_id"],
    )
    if resume is None or job_description is None or strategy is None:
        raise ValueError("resume, job_description, and alignment_strategy must be set.")

    optimized_experience = _optimize_experience_entries(resume, job_description, strategy)
    duration_ms = round((time.monotonic() - start_time) * 1000)
    logger.info(
        "pipeline_stage_completed",
        stage="optimize_experience",
        run_id=state["run_id"],
        duration_ms=duration_ms,
    )
    return {"optimized_experience": optimized_experience}


def _optimize_experience_entries(
    resume: Resume,
    job_description: JobDescription,
    strategy: AlignmentStrategy,
) -> OptimizedExperienceSection:
    """Optimize every resume experience and merge the role-scoped sections.

    Expects resume.work_experience to contain at least one entry.
    Returns one OptimizedExperienceSection for downstream ATS assembly.
    """
    experiences = resume.work_experience
    if not experiences:
        raise ValueError("resume.work_experience must contain at least one entry.")

    sections = _run_experience_optimization_workers(
        resume=resume,
        job_description=job_description,
        strategy=strategy,
        experiences=experiences,
    )
    return _merge_optimized_experience_sections(sections)


def _build_resume_with_single_experience(resume: Resume, experience: Experience) -> Resume:
    """Copy resume context with only one work experience entry.

    Expects a validated Resume and one Experience from resume.work_experience.
    Returns a Resume copy whose work_experience list contains only that entry.
    """
    return resume.model_copy(update={"work_experience": [experience]})


def _require_single_optimized_experience(section: OptimizedExperienceSection) -> Experience:
    """Return the single optimized experience expected from a role-scoped call.

    Expects an OptimizedExperienceSection from one role-only CrewAI task.
    Raises ValueError when the LLM returns zero or multiple experiences.
    """
    if len(section.optimized_experiences) != 1:
        raise ValueError("Role-scoped optimization must return exactly one experience.")
    return section.optimized_experiences[0]


def _accept_bullet_order_or_preserve_source(
    section: OptimizedExperienceSection,
    original_experience: Experience,
) -> OptimizedExperienceSection:
    """Accept only an exact source-bullet permutation; otherwise preserve the source role.

    The accepted role is always rebuilt from original_experience, so metadata,
    descriptions, and skills_used cannot be changed by the LLM.
    """
    proposed = _require_single_optimized_experience(section)
    if Counter(proposed.achievements) != Counter(original_experience.achievements):
        return _source_preserved_section(original_experience)
    prioritized = original_experience.model_copy(update={"achievements": proposed.achievements})
    return OptimizedExperienceSection(
        optimized_experiences=[prioritized],
        optimization_notes="Reordered source bullets without changing their text.",
    )


def _source_preserved_section(experience: Experience) -> OptimizedExperienceSection:
    """Return the untouched source role after an invalid LLM proposal."""
    return OptimizedExperienceSection(
        optimized_experiences=[experience],
        optimization_notes="LLM proposal changed source claims; original role preserved.",
    )


def _write_experience_section(context: str) -> OptimizedExperienceSection:
    """Ask the professional experience agent to write one role.

    Expects TOON context for a single role.
    Returns an OptimizedExperienceSection validated by CrewAI.
    """
    return run_agent_task(
        agent=create_professional_experience_agent(),
        task_name="optimize_experience_section_task",
        context=context,
        output_model=OptimizedExperienceSection,
    )


def _run_single_experience_optimization(
    resume: Resume,
    job_description: JobDescription,
    strategy: AlignmentStrategy,
    experience: Experience,
) -> OptimizedExperienceSection:
    """Prioritize one role's existing bullets with a role-scoped CrewAI call.

    Expects job_description and strategy to be present in pipeline state.
    Returns an exact source role, optionally with its original bullets reordered.
    """
    role_resume = _build_resume_with_single_experience(resume, experience)
    context = format_experience_optimizer_context(
        resume=role_resume,
        job_description=job_description,
        strategy=strategy,
        format_type="toon",
    )
    proposal = _write_experience_section(context)
    return _accept_bullet_order_or_preserve_source(proposal, experience)


def _merge_optimized_experience_sections(
    sections: list[OptimizedExperienceSection],
) -> OptimizedExperienceSection:
    """Merge role-scoped optimization results into one section.

    Expects each section to contain at least one optimized experience.
    Returns one OptimizedExperienceSection for downstream ATS assembly.
    """
    optimized_experiences = []
    optimization_notes = []
    keywords_integrated = []
    relevance_scores = []

    for section in sections:
        optimized_experiences.extend(section.optimized_experiences)
        if section.optimization_notes:
            optimization_notes.append(section.optimization_notes)
        keywords_integrated.extend(section.keywords_integrated)
        relevance_scores.extend(section.relevance_scores)

    return OptimizedExperienceSection(
        optimized_experiences=optimized_experiences,
        optimization_notes="\n".join(optimization_notes),
        keywords_integrated=list(dict.fromkeys(keywords_integrated)),
        relevance_scores=relevance_scores,
    )


def _run_experience_optimization_workers(
    resume: Resume,
    job_description: JobDescription,
    strategy: AlignmentStrategy,
    experiences: list[Experience],
) -> list[OptimizedExperienceSection]:
    """Run role-scoped experience optimization calls in parallel.

    Expects a non-empty experiences list from resume.work_experience.
    Returns one OptimizedExperienceSection per input experience.
    """
    max_workers = min(len(experiences), 4)
    logger.debug(
        "experience_optimization_parallelism",
        experience_count=len(experiences),
        max_workers=max_workers,
    )
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(
            executor.map(
                lambda experience: _run_single_experience_optimization(
                    resume,
                    job_description,
                    strategy,
                    experience,
                ),
                experiences,
            )
        )
