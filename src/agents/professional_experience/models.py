"""
Output contracts for the Professional Experience agent.

Defines both:
- the internal rewrite proposal used during the role-scoped rewrite/review loop
- the final optimized experience section used downstream by ATS assembly
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.data_models.resume import Experience


class ExperienceBulletRewrite(BaseModel):
    """One rewritten bullet plus the role evidence that supports it."""

    bullet_id: str = Field(
        ...,
        description="Stable ID of the source bullet being rewritten; copied exactly from context.",
    )
    source_bullet: str = Field(
        ...,
        description="The original bullet from this same role that is being rewritten.",
    )
    rewritten_bullet: str = Field(
        ...,
        description="The rewritten bullet, kept truthful to the same role and contribution level.",
    )
    supporting_role_evidence: list[str] = Field(
        default_factory=list,
        description="Short exact phrases or facts from this role's description, skills_used, or source bullet that support the rewrite.",
    )
    ownership_level: Literal["owned", "led", "executed", "contributed", "supported"] = Field(
        ...,
        description="The contribution level preserved from the source bullet.",
    )
    clarifying_question: str | None = Field(
        default=None,
        description=(
            "A specific question for the candidate, set ONLY when this bullet still "
            "cannot state a concrete result or scope without inventing facts -- ask "
            "exactly for the missing piece (e.g. what changed, for whom, at what "
            "scale). None when the bullet is already fully grounded."
        ),
    )


class ExperienceRewriteProposal(BaseModel):
    """Internal proposal returned by the role-scoped rewrite task."""

    rewritten_bullets: list[ExperienceBulletRewrite] = Field(
        ...,
        min_length=1,
        description="One rewrite record per source bullet, preserving bullet count.",
    )
    optimization_notes: str = Field(
        default="",
        description="Short notes about rewrite choices or limits encountered.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "rewritten_bullets": [
                    {
                        "bullet_id": "exp_001::bullet::0",
                        "source_bullet": "Worked on backend services for claims operations.",
                        "rewritten_bullet": "Built Python backend services for claims operations workflows, improving reliability for internal adjudication tools.",
                        "supporting_role_evidence": [
                            "claims operations workflows",
                            "Python",
                            "internal adjudication tools",
                        ],
                        "ownership_level": "executed",
                        "clarifying_question": None,
                    }
                ],
                "optimization_notes": "Used the role description to name the workflow and system without adding unsupported scope.",
            }
        }
    )


class ExperienceRelevance(BaseModel):
    """Relevance of one experience entry to the target job, scored 0-100.

    A list of these replaces a free-form {company_name: score} dict: a dynamic-key
    dict is not expressible as a native structured-output (response_format) schema,
    whereas a list of fixed-field objects is.
    """

    company_name: str = Field(..., description="Company of the experience entry being scored.")
    relevance_score: float = Field(
        ..., ge=0, le=100, description="How relevant this entry is to the target job (0-100)."
    )


class OptimizedExperienceSection(BaseModel):
    """Optimized work experience entries with metadata."""

    optimized_experiences: list[Experience] = Field(
        ...,
        description="List of optimized experience entries from the resume",
        min_length=1,
    )
    optimization_notes: str = Field(
        default="",
        description="Overall notes about optimization decisions and trade-offs",
    )
    keywords_integrated: list[str] = Field(
        default_factory=list,
        description="Keywords from the strategy that were integrated into bullets",
    )
    relevance_scores: list[ExperienceRelevance] = Field(
        default_factory=list,
        description="Relevance score per experience entry (one entry per company).",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "optimized_experiences": [
                    {
                        "job_title": "Senior Software Engineer",
                        "company_name": "Tech Corp",
                        "start_date": "2020-01-15",
                        "end_date": None,
                        "is_current_position": True,
                        "location": "San Francisco, CA",
                        "description": "Led development of cloud-native microservices platform",
                        "achievements": [
                            "Architected scalable microservices infrastructure using Python and AWS, reducing deployment time by 65%",
                        ],
                        "skills_used": ["Python", "AWS", "Docker", "Kubernetes"],
                    }
                ],
                "optimization_notes": "Emphasized cloud and Python experience per strategy guidance",
                "keywords_integrated": ["Python", "AWS", "microservices", "Docker"],
                "relevance_scores": [{"company_name": "Tech Corp", "relevance_score": 95.0}],
            }
        }
    )
