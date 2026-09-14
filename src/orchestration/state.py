"""
ResumeEnhancementPipelineState: the shared state every node in the graph reads and writes.

How state moves: a node receives the whole state, returns a dict holding only the
fields it produced, and LangGraph merges that dict into the state for the next node.
A field is None until its producing node has run, so a node may only read fields
produced upstream of it. The edges in graph.py are what guarantee that order.
"""

from typing import TypedDict

from src.agents.ats_optimizer.models import AtsOptimizedResume
from src.agents.professional_experience.models import OptimizedExperienceSection
from src.agents.professional_summary.models import ProfessionalSummary
from src.data_models.evaluation import RenderedStructureEvaluation, ResumeQualityReport
from src.data_models.job import JobDescription
from src.data_models.rendering import RenderedResumeArtifacts
from src.data_models.resume import OptimizedSkillsSection, Resume
from src.data_models.strategy import AlignmentStrategy
from src.hitl.professional_experience.models import (
    ExperienceBulletClarification,
)
from src.tools.contracts import ReviewResult


class ResumeEnhancementPipelineState(TypedDict):
    """All data in flight between pipeline nodes.

    Fields are grouped by the stage that produces them.
    A None value means the producing node has not yet run.
    """

    # --- inputs (set by the runner before graph.invoke()) ---
    run_id: str  # identifies this run; keys the Redis PII mapping (see pii_mapping_store)
    resume_path: str
    jd_path: str
    # Candidate's answered clarifications from a previous run's sheet (HITL loop).
    # None/empty on a first run; answers are folded into role evidence by the
    # experience node so candidate-provided facts and figures become usable.
    clarification_answers: list[ExperienceBulletClarification] | None

    # --- Stage 1: parallel ingestion ---
    resume: Resume | None
    job_description: JobDescription | None

    # --- Stage 2: sequential gap analysis ---
    requirement_match_report: ReviewResult | None  # code-computed; fed into the agent's context
    alignment_strategy: AlignmentStrategy | None

    # --- Stage 3: parallel content generation ---
    professional_summary: ProfessionalSummary | None
    optimized_experience: OptimizedExperienceSection | None
    optimized_skills: OptimizedSkillsSection | None
    # Questions for the candidate about bullets that shipped truthful but thin
    # (HITL loop). The runner writes these to an editable clarification sheet.
    experience_clarifications: list[ExperienceBulletClarification] | None

    # --- Stage 4: sequential ATS assembly ---
    optimized_resume: AtsOptimizedResume | None

    # --- Stage 5: sequential quality assurance ---
    quality_report: ResumeQualityReport | None
    # Code-owned rendered-ATS verdict (authoritative over the agent's self-cert).
    # A non-PASS status here forces quality_report.passes_quality_gate to False.
    rendered_structure_evaluation: RenderedStructureEvaluation | None

    # --- Stage 5b: conditional ATS section recovery (only when the ATS check FAILed) ---
    # Terminal disposition: True when the ATS failure is unrecoverable (the essential
    # section is empty upstream too) or the check was INCONCLUSIVE (nothing could be
    # built to inspect). The resume is not rendered; a human must review it.
    human_review_required: bool

    # --- Stage 6: conditional render (only when the QA gate passes) ---
    # Markdown always; PDF best-effort. None until the render node runs.
    rendered_artifacts: RenderedResumeArtifacts | None


def new_pipeline_state(
    run_id: str,
    resume_path: str,
    jd_path: str,
) -> ResumeEnhancementPipelineState:
    """Build the state a fresh run starts from: the three inputs, everything else empty.

    Every key must be present even when its value is None, so a new field added to
    the TypedDict must be added here too.

    Two fields start with a value rather than None, because they are read before any
    node could have produced them: human_review_required (False) is read by the final
    disposition, and clarification_answers (empty list) is read by the HITL router.
    """
    return {
        "run_id": run_id,
        "resume_path": resume_path,
        "jd_path": jd_path,
        "clarification_answers": [],
        "resume": None,
        "job_description": None,
        "requirement_match_report": None,
        "alignment_strategy": None,
        "professional_summary": None,
        "optimized_experience": None,
        "experience_clarifications": None,
        "optimized_skills": None,
        "optimized_resume": None,
        "quality_report": None,
        "rendered_structure_evaluation": None,
        "human_review_required": False,
        "rendered_artifacts": None,
    }


def require[T](value: T | None, field: str) -> T:
    """Return a state field that an upstream node must already have populated.

        resume = require(state["resume"], "resume")

    Narrows `X | None` to `X` for the type checker, and raises RuntimeError if the
    field is still None. A None here means the graph is wired wrong, not that the
    input was bad, so it is a programming error: the CLI does not catch it (see
    exceptions.py) and it is not an `assert`, which `python -O` would strip.
    """
    if value is None:
        raise RuntimeError(
            f"Pipeline state '{field}' is None; the node that produces it has not run. "
            "This is a graph wiring bug, not a bad input."
        )
    return value
