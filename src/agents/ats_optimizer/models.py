"""
Output contract for the ATS Optimizer agent.

Defines what the agent produces when it assembles the optimized sections into a
single, ATS-aligned resume. The contract is deliberately slim: the agent emits
the assembled Resume plus its decision notes. It does NOT emit rendered markdown,
rendered JSON, or a self-scored validation report -- those are mechanical and are
computed code-side (see engines.check_ats_quality), never self-graded by the LLM.
"""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from src.data_models.resume import Resume  # domain entity the agent assembles into


class AtsOptimizedResume(BaseModel):
    """The assembled, ATS-aligned resume plus the agent's decision notes."""

    final_resume: Resume = Field(
        ...,
        description=(
            "The complete resume assembled from the optimized summary, experience, "
            "and skills, with contact info and education carried over from the "
            "original. Section content is ordered and worded for ATS parsing."
        ),
    )
    section_order: Annotated[
        list[str],
        Field(
            description=(
                "The standard section headers in the order the agent placed them "
                "(e.g. ['Professional Summary', 'Work Experience', 'Skills', "
                "'Education']). Each must be an ATS-recognized standard header."
            )
        ),
    ] = []
    optimization_summary: str = Field(
        default="",
        description="Plain-language summary of the assembly and ATS decisions the agent made.",
    )
    keyword_integration_notes: str = Field(
        default="",
        description=(
            "How the agent ensured must-have job keywords are present without "
            "stuffing, and which keywords it could not place truthfully."
        ),
    )
    unresolved_issues: Annotated[
        list[str],
        Field(
            description=(
                "ATS concerns the agent could not fix without inventing content "
                "(e.g. a required keyword with no supporting experience). Empty "
                "list means the agent resolved everything it found."
            )
        ),
    ] = []

    model_config = ConfigDict(
        # Domain-neutral example: [PLACEHOLDERS] stand for the candidate's real field.
        # keyword_integration_notes reports what the coverage tool found -- it does NOT
        # self-assert a density number (density/score are graded in code).
        json_schema_extra={
            "example": {
                "final_resume": {
                    "full_name": "[FULL_NAME]",
                    "email": "[EMAIL]",
                    "professional_summary": "[assembled summary text]",
                    "work_experience": [],
                    "education": [],
                    "skills": [],
                },
                "section_order": [
                    "Professional Summary",
                    "Work Experience",
                    "Skills",
                    "Education",
                ],
                "optimization_summary": "Assembled all sections, standardized headers, ordered summary first.",
                "keyword_integration_notes": "Coverage tool confirmed all must-have keywords present; [KEYWORD] had no truthful support and was left out.",
                "unresolved_issues": [],
            }
        }
    )
