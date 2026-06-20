"""
Output contract for the Professional Experience agent.

Defines what the agent produces when it rewrites work experience bullet points:
optimized entries with metadata for downstream assembly and QA.
"""

from pydantic import BaseModel, ConfigDict, Field

from src.data_models.resume import Experience


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
