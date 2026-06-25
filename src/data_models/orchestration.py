"""Final result contract for the resume enhancement pipeline.

This is the typed object the orchestration runner returns to its caller. It lived
in the monolithic orchestrator (src/agent_orchestrator.py) historically; it is
moved here so the LangGraph runner does not depend on the retired monolith.
"""

from pydantic import BaseModel, Field

from src.agents.ats_optimizer.models import AtsOptimizedResume
from src.data_models.evaluation import ResumeQualityReport
from src.data_models.job import JobDescription
from src.data_models.rendering import RenderedResumeArtifacts
from src.data_models.resume import Resume
from src.data_models.strategy import AlignmentStrategy


class OrchestrationResult(BaseModel):
    """The complete output of one resume tailoring run."""

    original_resume: Resume = Field(
        ..., description="The candidate's resume as parsed from the source document."
    )
    job_description: JobDescription = Field(
        ..., description="The structured job description the resume was tailored to."
    )
    strategy: AlignmentStrategy = Field(
        ..., description="The gap-analysis strategy that guided the optimization."
    )
    optimized_resume: AtsOptimizedResume = Field(
        ..., description="The assembled, ATS-aligned resume plus the optimizer's decision notes."
    )
    quality_report: ResumeQualityReport = Field(
        ..., description="The quality assessment, including the code-owned pass/fail gate."
    )
    rendered_artifacts: RenderedResumeArtifacts | None = Field(
        None,
        description=(
            "Paths to the produced resume files (Markdown always, PDF best-effort), set "
            "only when the quality gate passed and rendering ran. None when it did not pass."
        ),
    )
