"""Stage 3 professional experience optimization flow.

This module owns the role-scoped writer -> audit -> optional rewrite -> merge
flow. Keeping it separate makes the multi-agent boundary explicit: the LLM
writes experience content, while code owns typed audit decisions and ID repair.
"""

from concurrent.futures import ThreadPoolExecutor

from src.agents.professional_experience import create_professional_experience_agent
from src.agents.professional_experience.models import OptimizedExperienceSection
from src.data_models.job import JobDescription
from src.data_models.resume import Experience, Resume
from src.data_models.strategy import AlignmentStrategy
from src.formatters.experience_optimizer_formatter import format_experience_optimizer_context
from src.orchestration.crew_task_execution import run_agent_task
from src.orchestration.state import ResumeEnhancementPipelineState
from src.tools.resume_diagnostics import audit_experience_quality_for_experiences
from src.tools.review_contract.review_models import ReviewResult, Severity

_LANGUAGE_QUALITY_ENGINE_ID = "language_quality_auditor"
_BULLET_STRUCTURE_ENGINE_ID = "bullet_structure_auditor"


def optimize_experience(state: ResumeEnhancementPipelineState) -> dict:
    """Rewrite work experience bullets with one role-scoped call per entry.

    Reads: resume, job_description, and alignment_strategy.
    Writes: optimized_experience.
    Returns: partial state with merged OptimizedExperienceSection output.
    """
    resume = state["resume"]
    job_description = state["job_description"]
    strategy = state["alignment_strategy"]
    if resume is None or job_description is None or strategy is None:
        raise ValueError("resume, job_description, and alignment_strategy must be set.")

    optimized_experience = _optimize_experience_entries(resume, job_description, strategy)
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


def _restore_original_experience_id(
    section: OptimizedExperienceSection,
    original_experience: Experience,
) -> OptimizedExperienceSection:
    """Copy the code-owned experience_id onto the LLM-written experience.

    Expects one optimized experience and the original source Experience.
    Returns a section whose optimized experience keeps the original ID.
    """
    optimized = _require_single_optimized_experience(section)
    fixed = optimized.model_copy(update={"experience_id": original_experience.experience_id})
    return section.model_copy(update={"optimized_experiences": [fixed]})


def _experience_audit_needs_rewrite(audit_result: ReviewResult) -> bool:
    """Return True when audit findings are serious enough for one rewrite.

    Expects a ReviewResult from the code-owned diagnostics layer.
    Returns True for blocker/major, language-quality, bullet-count, or multiple
    non-trivial findings.
    """
    serious = {Severity.BLOCKER, Severity.MAJOR}
    non_trivial = {Severity.MINOR, Severity.SUGGESTION}
    comments = audit_result.comments

    if any(comment.severity in serious for comment in comments):
        return True
    if any(comment.engine_id == _LANGUAGE_QUALITY_ENGINE_ID for comment in comments):
        return True
    has_bullet_structure_issue = any(
        _is_bullet_structure_finding(comment.engine_id, comment.message)
        for comment in comments
    )
    if has_bullet_structure_issue:
        return True
    return sum(1 for comment in comments if comment.severity in non_trivial) >= 2


def _is_bullet_structure_finding(engine_id: str, message: str) -> bool:
    """Return True when an audit comment flags bullet structure.

    Expects a review engine ID and message string.
    Returns True for bullet-structure auditor comments mentioning bullets.
    """
    return engine_id == _BULLET_STRUCTURE_ENGINE_ID and "bullet" in message.lower()


def _render_experience_audit_feedback(audit_result: ReviewResult) -> str:
    """Render audit comments into compact feedback for one rewrite attempt.

    Expects a ReviewResult from the experience audit helper.
    Returns plain text suitable for a CrewAI task context.
    """
    lines = [audit_result.summary or "Experience audit found serious issues."]
    for comment in audit_result.comments:
        lines.append(f"- {comment.severity.value}: {comment.message}")
        lines.append(f"  advice: {comment.advice}")
    return "\n".join(lines)


def _build_experience_rewrite_context(
    original_context: str,
    section: OptimizedExperienceSection,
    audit_result: ReviewResult,
) -> str:
    """Add previous output and audit feedback to the original role context.

    Expects original TOON context, prior structured output, and audit result.
    Returns context for exactly one rewrite attempt.
    """
    return (
        f"{original_context}\n\n"
        f"PREVIOUS_OPTIMIZED_EXPERIENCE_JSON:\n{section.model_dump_json()}\n\n"
        f"EXPERIENCE_AUDIT_FEEDBACK:\n{_render_experience_audit_feedback(audit_result)}\n\n"
        "Rewrite once to address blocker or major audit findings. "
        "Return only OptimizedExperienceSection JSON."
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


def _audit_experience_section(section: OptimizedExperienceSection) -> ReviewResult:
    """Run code-owned quality checks on optimized experience output.

    Expects a role-scoped OptimizedExperienceSection.
    Returns the merged ReviewResult from the diagnostics layer.
    """
    return audit_experience_quality_for_experiences(section.optimized_experiences)


def _rewrite_experience_section_once(
    context: str,
    section: OptimizedExperienceSection,
    audit_result: ReviewResult,
) -> OptimizedExperienceSection:
    """Ask for one rewrite using the previous output and audit feedback.

    Expects a serious audit result for the first optimized section.
    Returns one rewritten OptimizedExperienceSection.
    """
    rewrite_context = _build_experience_rewrite_context(context, section, audit_result)
    return _write_experience_section(rewrite_context)


def _run_single_experience_optimization(
    resume: Resume,
    job_description: JobDescription,
    strategy: AlignmentStrategy,
    experience: Experience,
) -> OptimizedExperienceSection:
    """Optimize one experience entry with a role-scoped CrewAI call.

    Expects job_description and strategy to be present in pipeline state.
    Returns an OptimizedExperienceSection containing the optimized role entry.
    """
    role_resume = _build_resume_with_single_experience(resume, experience)
    context = format_experience_optimizer_context(
        resume=role_resume,
        job_description=job_description,
        strategy=strategy,
        format_type="toon",
    )
    optimized_section = _write_experience_section(context)
    optimized_section = _restore_original_experience_id(optimized_section, experience)
    audit_result = _audit_experience_section(optimized_section)

    if not _experience_audit_needs_rewrite(audit_result):
        return optimized_section

    rewritten_section = _rewrite_experience_section_once(context, optimized_section, audit_result)
    return _restore_original_experience_id(rewritten_section, experience)


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
    relevance_scores = {}

    for section in sections:
        optimized_experiences.extend(section.optimized_experiences)
        if section.optimization_notes:
            optimization_notes.append(section.optimization_notes)
        keywords_integrated.extend(section.keywords_integrated)
        relevance_scores.update(section.relevance_scores)

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
