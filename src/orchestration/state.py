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
from src.agents.summary_writer_agent import ProfessionalSummary
from src.data_models.evaluation import QualityReport
from src.data_models.job import JobDescription
from src.data_models.resume import OptimizedSkillsSection, Resume
from src.data_models.strategy import AlignmentStrategy


class ResumeEnhancementPipelineState(TypedDict):
    """All data in flight between pipeline nodes.

    Fields are grouped by the stage that produces them.
    A None value means the producing node has not yet run.
    """

    # --- inputs (set by the runner before graph.invoke()) ---
    resume_path: str
    jd_path: str

    # --- Stage 1: parallel ingestion ---
    resume: Resume | None
    job_description: JobDescription | None

    # --- Stage 2: sequential gap analysis ---
    alignment_strategy: AlignmentStrategy | None

    # --- Stage 3: parallel content generation ---
    professional_summary: ProfessionalSummary | None
    optimized_experience: OptimizedExperienceSection | None
    optimized_skills: OptimizedSkillsSection | None

    # --- Stage 4: sequential ATS assembly ---
    optimized_resume: AtsOptimizedResume | None

    # --- Stage 5: sequential quality assurance ---
    qa_report: QualityReport | None

    # --- Stage 6: conditional PDF render (only when the QA gate passes) ---
    rendered_resume_path: str | None
