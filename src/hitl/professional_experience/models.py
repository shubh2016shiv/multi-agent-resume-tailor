"""Contracts for professional-experience bullet clarification HITL.

These models are intentionally named around the product behavior: the pipeline
pauses only when an experience bullet needs a candidate-owned fact that the LLM
cannot invent safely.
"""

from enum import Enum

from pydantic import BaseModel, Field

from src.data_models.resume import Experience


class ExperienceBulletMissingFactCategory(str, Enum):
    """Candidate-owned fact types that can justify pausing the experience flow."""

    ARTIFACT = "artifact"  # What the candidate built, changed, shipped, or operated.
    RESULT = "result"  # What improved because of the work, with or without a metric.
    USER_SCOPE = "user_scope"  # Who or what the work served: users, teams, systems, clients.
    SCALE = "scale"  # Size/context such as data volume, request load, team size, or rollout scope.


class ExperienceBulletFactGapFinding(BaseModel):
    """Semantic decision on whether one shipped experience bullet needs candidate facts."""

    bullet_id: str = Field(
        ...,
        description="Stable ID for the source bullet being judged.",
    )
    current_bullet: str = Field(
        ...,
        description="The exact bullet text that would ship if no candidate input is collected.",
    )
    requires_candidate_input: bool = Field(
        ...,
        description=(
            "True only when the bullet is truthful but still missing candidate-owned "
            "artifact, result, user/scope, or scale facts the LLM cannot invent."
        ),
    )
    gap_category: ExperienceBulletMissingFactCategory | None = Field(
        default=None,
        description="The main missing fact type when requires_candidate_input is true.",
    )
    missing_fact_summary: str = Field(
        default="",
        description="Plain summary of the exact fact missing from the shipped bullet.",
    )
    why_candidate_input_is_needed: str = Field(
        default="",
        description="Brief reason the missing fact must come from the candidate.",
    )
    question: str | None = Field(
        default=None,
        description=(
            "The final candidate-facing question, set only when requires_candidate_input "
            "is true: one direct, concise, professional question asking for exactly the "
            "missing fact."
        ),
    )


class ExperienceBulletFactGapReview(BaseModel):
    """Structured semantic review of candidate-owned fact gaps in shipped bullets."""

    findings: list[ExperienceBulletFactGapFinding] = Field(
        default_factory=list,
        description="One fact-gap decision per shipped experience bullet.",
    )


class ExperienceClarificationPausedRunStatus(str, Enum):
    """Lifecycle state of a paused professional-experience clarification run."""

    WAITING_FOR_CANDIDATE = "waiting_for_candidate"


class ExperienceClarificationPausedRunManifest(BaseModel):
    """Local metadata required to resume a paused experience-clarification run."""

    run_id: str = Field(..., description="Pipeline run id and LangGraph thread id.")
    resume_path: str = Field(..., description="Original resume path for audit/debug use.")
    jd_path: str = Field(..., description="Original job-description path for audit/debug use.")
    status: ExperienceClarificationPausedRunStatus = Field(
        default=ExperienceClarificationPausedRunStatus.WAITING_FOR_CANDIDATE,
        description="Current paused-run lifecycle state.",
    )
    clarifications_filename: str = Field(
        default="clarifications_sheet.json",
        description="Sheet file the candidate edits before resume.",
    )
    checkpoint_db_filename: str = Field(
        default="checkpoints.sqlite3",
        description=(
            "SQLite database inside the paused-run directory holding the LangGraph "
            "checkpoint history for this run's thread; opened with SqliteSaver on resume."
        ),
    )


class ExperienceBulletClarification(BaseModel):
    """One question to the candidate about one experience bullet.

    Produced by the experience stage when a bullet ships truthful but thin.
    The same object carries the candidate's answer back on the next run.
    """

    company_name: str = Field(
        ...,
        description="Company of the role this bullet belongs to; used to route the answer back to the right role.",
    )
    bullet_id: str = Field(
        ...,
        description="Stable ID for the exact bullet this question belongs to; the primary resume-path routing key.",
    )
    job_title: str = Field(
        ...,
        description="Job title of the role this bullet belongs to; used together with company_name for routing.",
    )
    start_date: str = Field(
        default="",
        description=(
            "ISO start date of the role, the routing tiebreaker when the candidate held "
            "the same title at the same company twice. Empty on sheets written before "
            "this field existed; routing then falls back to company_name + job_title."
        ),
    )
    bullet: str = Field(
        ...,
        description="The bullet text as it shipped in this run -- shown so the candidate knows what to elaborate on.",
    )
    why_flagged: str = Field(
        ...,
        description="The reviewer's note explaining what the bullet is missing (e.g. no concrete system or result).",
    )
    gap_category: ExperienceBulletMissingFactCategory = Field(
        ...,
        description="The main kind of candidate-owned fact still missing from this bullet.",
    )
    missing_fact_summary: str = Field(
        ...,
        description="Short summary of the exact missing fact the candidate should provide.",
    )
    question: str = Field(
        ...,
        description="The targeted question the candidate should answer to make this bullet specific.",
    )
    answer: str = Field(
        default="",
        description="The candidate's answer in their own words. Empty until the candidate fills it in; only answered entries are used on the next run.",
    )


def build_experience_bullet_id(experience: Experience, bullet_index: int) -> str:
    """Return a stable id for one bullet inside one professional-experience role."""
    role_id = experience.experience_id or (
        f"{experience.company_name.strip().lower()}::{experience.job_title.strip().lower()}::"
        f"{experience.start_date.isoformat()}"
    )
    return f"{role_id}::bullet::{bullet_index}"
