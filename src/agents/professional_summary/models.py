"""
Output contracts for the Professional Summary agent.

These Pydantic models define what the agent produces when it generates
multiple summary drafts and recommends the best one.
"""

from pydantic import BaseModel, ConfigDict, Field


class SummaryDraft(BaseModel):
    """A single draft version of the professional summary."""

    version_name: str = Field(
        ...,
        description="Name of the version (e.g., 'Hook-Value-Future', 'Story Spine', 'ATS-Optimized')",
    )
    strategy_used: str = Field(
        ...,
        description="Brief description of the writing strategy or framework used for this draft",
    )
    content: str = Field(
        ...,
        description="The actual summary text",
        min_length=50,
        max_length=1000,
    )
    critique: str = Field(
        default="",
        description="Self-critique of this draft (strengths and weaknesses)",
    )
    score: int = Field(
        ...,
        description="Self-assigned confidence score for this draft (0-100)",
        ge=0,
        le=100,
    )


class ProfessionalSummary(BaseModel):
    """Structured output: multiple strategic drafts with a recommendation."""

    drafts: list[SummaryDraft] = Field(
        ...,
        description="List of generated summary drafts using different writing strategies",
        min_length=1,
    )
    recommended_version: str = Field(
        ...,
        description="The version_name of the draft the agent most strongly recommends",
    )
    writing_notes: str = Field(
        default="",
        description="Overall notes about the writing process and trade-offs between versions",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "drafts": [
                    {
                        "version_name": "Hook-Value-Future",
                        "strategy_used": "Classic 3-part structure: Hook -> Value -> Future",
                        "content": "Senior Machine Learning Engineer with 3.5+ years...",
                        "critique": "Strong hook, could use more specific metrics in the value section.",
                        "score": 85,
                    },
                ],
                "recommended_version": "Hook-Value-Future",
                "writing_notes": "Hook-Value-Future aligns best with the senior level of this role.",
            }
        }
    )
