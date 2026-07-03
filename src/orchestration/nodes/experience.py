"""Stage 3 professional experience rewrite flow.

The LLM rewrites one role's bullets 1:1 for recruiter readability; code owns the
gate. Each proposal must keep the source bullet count, introduce no figure the
source role lacks (claim-inflation check), and pass a truthfulness drift review.
A failing proposal gets exactly one repair attempt with the findings; if that
also fails, the original source bullets ship. Language-quality findings trigger
the same single repair but never force the source fallback -- the source is what
needed the language fix. Role metadata is always rebuilt from the source object,
so the LLM's writable surface is the achievements text only.
"""

import time
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
from src.tools.contracts import Confidence, ReviewComment
from src.tools.engines.resume_diagnostics import audit_language_quality_for_experiences
from src.tools.engines.truthfulness import detect_claim_inflation, detect_rewrite_drift

logger = get_logger(__name__)


def optimize_experience(state: ResumeEnhancementPipelineState) -> dict:
    """Rewrite work-experience bullets with one role-scoped call per entry.

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

    optimized_experience = _optimize_experience_entries(resume, job_description, strategy, state["run_id"])
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
    run_id: str = "unknown",
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
        run_id=run_id,
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


def _rebuild_role_from_source(
    section: OptimizedExperienceSection,
    original_experience: Experience,
) -> Experience:
    """Keep only the proposed achievements text; every other field comes from the source.

    This is the containment boundary: metadata, description, and skills_used can
    never be changed by the LLM, whatever the proposal contains.
    """
    proposed = _require_single_optimized_experience(section)
    return original_experience.model_copy(update={"achievements": list(proposed.achievements)})


def _truth_gate_failures(
    original_role_resume: Resume,
    revised_role: Experience,
    original_experience: Experience,
) -> list[str]:
    """Return truth-gate failure messages for one rewritten role (empty list = pass).

    Three checks, cheapest first, short-circuiting so a mechanically failed
    proposal never spends an LLM review call:
    1. Bullet count parity -- the 1:1 rewrite contract, a closed countable fact.
    2. detect_claim_inflation -- deterministic; any figure the source role lacks fails.
    3. detect_rewrite_drift -- LLM judgment; only HIGH-confidence drift blocks, per
       that engine's own documented gating (advisory drift must not force fallback).
    """
    source_count = len(original_experience.achievements)
    revised_count = len(revised_role.achievements)
    if revised_count != source_count:
        return [
            f"Bullet count changed: source role has {source_count} bullet(s), "
            f"rewrite has {revised_count}. Rewrite each source bullet one-for-one."
        ]

    revised_role_resume = original_role_resume.model_copy(
        update={"work_experience": [revised_role]}
    )
    inflation = detect_claim_inflation(original_role_resume, revised_role_resume)
    if inflation.comments:
        return [f"{comment.message}. {comment.advice}" for comment in inflation.comments]

    drift = detect_rewrite_drift(original_role_resume, revised_role_resume)
    blocking_drift = [
        comment for comment in drift.comments if comment.confidence is Confidence.HIGH
    ]
    return [f"{comment.message}. {comment.advice}" for comment in blocking_drift]


def _language_findings(revised_role: Experience) -> list[ReviewComment]:
    """Run the language-quality audit on one rewritten role's bullets."""
    return audit_language_quality_for_experiences([revised_role]).comments


def _accepted_section(revised_role: Experience, note: str) -> OptimizedExperienceSection:
    """Wrap an accepted (truth-verified) role for the downstream merge."""
    return OptimizedExperienceSection(
        optimized_experiences=[revised_role],
        optimization_notes=note,
    )


def _source_preserved_section(experience: Experience, reasons: list[str]) -> OptimizedExperienceSection:
    """Return the untouched source role after the rewrite failed the truth gate twice."""
    return OptimizedExperienceSection(
        optimized_experiences=[experience],
        optimization_notes=(
            "Rewrite failed the truthfulness gate after one repair; original bullets "
            "preserved. Findings: " + "; ".join(reasons)
        ),
    )


def _build_experience_repair_context(
    original_context: str,
    section: OptimizedExperienceSection,
    findings: list[str],
) -> str:
    """Add the previous proposal and the exact gate findings for the single repair call."""
    feedback = "\n".join(f"- {finding}" for finding in findings)
    return (
        f"{original_context}\n\n"
        f"PREVIOUS_OPTIMIZED_EXPERIENCE_JSON:\n{section.model_dump_json()}\n\n"
        f"EXPERIENCE_AUDIT_FEEDBACK:\n{feedback}\n\n"
        "Rewrite once more to fix every finding above by correcting the claim level or "
        "wording the finding names -- do NOT revert other bullets to the source's duty "
        "phrasing or hype; keep every improvement the findings did not flag. When a "
        "finding asks you to lower a claim, restate it at the accurate level in active "
        "phrasing ('Contributed to X', 'Supported Y') -- never copy the finding's or the "
        "source's own 'worked on'/'helped with'/'responsible for' wording. Keep the "
        "same bullet count and stay truthful to the source bullets: no figure, tool, "
        "scale, or outcome the source bullet does not state. Return only "
        "OptimizedExperienceSection JSON."
    )


def _write_experience_section(context: str, run_id: str = "unknown") -> OptimizedExperienceSection:
    """Ask the professional experience agent to write one role.

    Expects TOON context for a single role.
    Returns an OptimizedExperienceSection validated by CrewAI.
    """
    return run_agent_task(
        agent=create_professional_experience_agent(),
        task_name="optimize_experience_section_task",
        context=context,
        output_model=OptimizedExperienceSection,
        run_id=run_id,
    )


def _gate_rewrite_with_one_repair(
    proposal: OptimizedExperienceSection,
    context: str,
    original_role_resume: Resume,
    original_experience: Experience,
    run_id: str = "unknown",
) -> OptimizedExperienceSection:
    """Accept a truthful rewrite, spend the single repair on any finding, else keep the source.

    Consequences are asymmetric by design. A truth-gate failure (count mismatch,
    introduced figure, high-confidence drift) can never ship: it is repaired once,
    and a second failure falls back to the best truthful candidate -- the first
    proposal if it was truthful, otherwise the source bullets. Language findings
    trigger the same single repair but never force the source fallback, because
    the source bullets are the very text the language audit flagged.
    """
    ####################################################
    # STEP 1: GATE THE FIRST PROPOSAL#
    ####################################################
    revised_role = _rebuild_role_from_source(proposal, original_experience)
    truth_failures = _truth_gate_failures(original_role_resume, revised_role, original_experience)
    language_comments = [] if truth_failures else _language_findings(revised_role)
    if not truth_failures and not language_comments:
        return _accepted_section(
            revised_role, "Rewrote bullets 1:1; truthfulness and language checks passed."
        )

    ####################################################
    # STEP 2: SPEND THE SINGLE REPAIR ON THE EXACT FINDINGS#
    ####################################################
    # One repair only: enough to prove the audit can influence output while keeping
    # cost bounded and failure visible (professional_experience_architecture.md 6.3).
    findings = truth_failures + [
        f"{comment.message}. {comment.advice}" for comment in language_comments
    ]
    logger.info(
        "experience_rewrite_repair_requested",
        run_id=run_id,
        company=original_experience.company_name,
        truth_failures=len(truth_failures),
        language_findings=len(language_comments),
    )
    repair_context = _build_experience_repair_context(context, proposal, findings)
    repaired = _write_experience_section(repair_context, run_id=run_id)
    repaired_role = _rebuild_role_from_source(repaired, original_experience)

    ####################################################
    # STEP 3: RE-GATE THE REPAIR AND SHIP THE BEST TRUTHFUL CANDIDATE#
    ####################################################
    repaired_truth_failures = _truth_gate_failures(
        original_role_resume, repaired_role, original_experience
    )
    if not repaired_truth_failures:
        return _accepted_section(
            repaired_role, "Repaired once per audit feedback; truthfulness verified."
        )
    if not truth_failures:
        # The first rewrite was truthful; only its language was flagged. A failed
        # repair must not discard truthful improvement, so the first rewrite ships.
        return _accepted_section(
            revised_role,
            "Repair failed the truthfulness gate; kept the first truthful rewrite. "
            "Residual language findings: "
            + "; ".join(comment.message for comment in language_comments),
        )
    logger.warning(
        "experience_rewrite_source_fallback",
        run_id=run_id,
        company=original_experience.company_name,
        findings=repaired_truth_failures,
    )
    return _source_preserved_section(original_experience, repaired_truth_failures)


def _run_single_experience_optimization(
    resume: Resume,
    job_description: JobDescription,
    strategy: AlignmentStrategy,
    experience: Experience,
    run_id: str = "unknown",
) -> OptimizedExperienceSection:
    """Rewrite one role's bullets with a role-scoped CrewAI call plus a code-owned gate.

    Expects job_description and strategy to be present in pipeline state.
    Returns a truth-verified rewritten role, or the untouched source role when the
    rewrite fails the truthfulness gate twice.
    """
    role_resume = _build_resume_with_single_experience(resume, experience)
    context = format_experience_optimizer_context(
        resume=role_resume,
        job_description=job_description,
        strategy=strategy,
        format_type="toon",
    )
    proposal = _write_experience_section(context, run_id=run_id)
    return _gate_rewrite_with_one_repair(proposal, context, role_resume, experience, run_id=run_id)


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
    run_id: str = "unknown",
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
                    run_id,
                ),
                experiences,
            )
        )
