"""
ResumeEnhancementPipelineState: the shared state that flows through every node in the graph.

Every field starts as None. A node sets its output field(s) and returns a partial
dict -- LangGraph merges that dict back into the state. Downstream nodes must only
read a field after the node that produces it has run.

The graph topology in graph.py enforces the correct read order.
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
