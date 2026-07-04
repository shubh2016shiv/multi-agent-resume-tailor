"""Final result contract for the resume enhancement pipeline.

This is the typed object the orchestration runner returns to its caller. It lived
in the monolithic orchestrator (src/agent_orchestrator.py) historically; it is
moved here so the LangGraph runner does not depend on the retired monolith.
"""

from enum import Enum

from pydantic import BaseModel, Field

from src.agents.ats_optimizer.models import AtsOptimizedResume
from src.data_models.evaluation import ResumeQualityReport
from src.data_models.job import JobDescription
from src.data_models.rendering import RenderedResumeArtifacts
from src.data_models.resume import Resume
from src.data_models.strategy import AlignmentStrategy
from src.hitl.professional_experience.models import (
    ExperienceBulletClarification,
)


class RunDisposition(str, Enum):
    """The caller's next action after a run, one value per distinct action.

    Exactly one disposition per run, chosen by precedence (most blocking wins):
    NEEDS_HUMAN_REVIEW > QUALITY_GATE_FAILED > NEEDS_CANDIDATE_INPUT > RENDERED.
    A NEEDS_CANDIDATE_INPUT run may still carry rendered artifacts -- the resume
    is usable today, and answering the clarification sheet makes it better.
    """

    RENDERED = "rendered"
    NEEDS_CANDIDATE_INPUT = "needs_candidate_input"
    NEEDS_HUMAN_REVIEW = "needs_human_review"
    QUALITY_GATE_FAILED = "quality_gate_failed"


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
    optimized_resume: AtsOptimizedResume | None = Field(
        None,
        description=(
            "The assembled, ATS-aligned resume plus the optimizer's decision notes. "
            "Null when the run paused for candidate clarification before ATS assembly."
        ),
    )
    quality_report: ResumeQualityReport | None = Field(
        None,
        description=(
            "The quality assessment, including the code-owned pass/fail gate. "
            "Null when the run paused before ATS assembly and QA."
        ),
    )
    rendered_artifacts: RenderedResumeArtifacts | None = Field(
        None,
        description=(
            "Paths to the produced resume files (Markdown always, PDF best-effort), set "
            "only when the quality gate passed and rendering ran. None when it did not pass."
        ),
    )
    clarifications_requested: list[ExperienceBulletClarification] = Field(
        default_factory=list,
        description=(
            "Questions for the candidate about bullets that shipped truthful but thin "
            "(HITL loop). Also written to an editable clarifications_sheet.json next to "
            "the run output; answer them there and re-run with that sheet."
        ),
    )
    disposition: RunDisposition = Field(
        ...,
        description=(
            "What the caller should do next -- derived by the runner from the human-review "
            "flag, the quality gate, and any candidate questions (see "
            "derive_run_disposition in src/orchestration/human_review_policy.py)."
        ),
    )
    paused_run_path: str | None = Field(
        None,
        description=(
            "Path to the paused-run directory when disposition is needs_candidate_input; "
            "resume from this directory after answering clarifications_sheet.json."
        ),
    )
