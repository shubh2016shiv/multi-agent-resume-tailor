"""
Output contract for the Professional Experience agent.

Defines what the agent produces when it rewrites work experience bullet points:
optimized entries with metadata for downstream assembly and QA.
"""

from pydantic import BaseModel, ConfigDict, Field

from src.data_models.resume import Experience


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
    relevance_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Relevance score per experience entry (key: company_name, value: 0-100)",
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
                "relevance_scores": {"Tech Corp": 95.0},
            }
        }
    )
