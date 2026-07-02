"""
Output contracts for the Professional Summary agent.

These Pydantic models define what the agent produces when it generates
multiple summary drafts and recommends the best one.
"""

from pydantic import BaseModel, ConfigDict, Field


class SummaryDraft(BaseModel):
    """A single draft version of the professional summary."""

    # NOTE ON FIELD ORDER: structured-output models fill fields top-to-bottom, and
    # earlier fields condition later ones. The order below is deliberate so that
    # reasoning precedes the conclusion: name -> strategy -> EVIDENCE -> content ->
    # critique -> score. `score` is last so it is a verdict on what was written,
    # not a number the model picks first and then rationalizes.
    version_name: str = Field(
        ...,
        description="Name of this draft's angle. Use the angle names from the task (e.g. 'Problem-Solver Thesis' or 'Gravity & Scope').",
    )
    strategy_used: str = Field(
        ...,
        description="One sentence: the positioning thesis for this draft and why it suits this role.",
    )
    evidence_used: str = Field(
        ...,
        description=(
            "FILL THIS FIRST, before writing `content`. List the specific resume- or "
            "strategy-supported facts you will draw on (skills, tools, roles, metrics, "
            "years) and the role vocabulary or JD term each fact justifies. `content` may "
            "use ONLY what is enumerated here. This field forces evidence-gathering before "
            "drafting -- do not leave it generic."
        ),
    )
    content: str = Field(
        ...,
        description="The summary text. Must use only facts and defensible implications listed in `evidence_used`.",
        min_length=50,
        max_length=1000,
    )
    critique: str = Field(
        default="",
        description=(
            "Written AFTER `content`: strengths, weaknesses, and explicit confirmation "
            "that every claim traces back to `evidence_used`."
        ),
    )
    score: int = Field(
        ...,
        description="Confidence score (0-100), assigned LAST as a verdict on the finished draft.",
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
                # Domain-neutral example. Placeholders in [BRACKETS] stand for whatever
                # the candidate's actual field is (nursing, finance, logistics, law,
                # software, etc.). The example teaches the STRUCTURE -- evidence -> the
                # keyword it justifies, and exclusion of unsupported keywords -- never a
                # specific industry. Do NOT copy any domain term from here into output.
                "drafts": [
                    {
                        "version_name": "Problem-Solver Thesis",
                        "strategy_used": "Lead with the defining problem this candidate is trusted to solve.",
                        "evidence_used": (
                            "Resume '[did X with TOOL_A]' -> justifies JD keyword '[KEYWORD_A]'. "
                            "Resume '[N years in ROLE]' -> justifies years + '[KEYWORD_B]'. "
                            "JD asks for '[KEYWORD_C]': no resume evidence -> excluded."
                        ),
                        "content": "[Professional thesis]. Built [supported capability] using "
                        "[TOOL_A], bringing [defensible operational implication] to [supported context]...",
                        "critique": "Every claim maps to evidence_used; '[KEYWORD_C]' correctly "
                        "omitted as unsupported.",
                        "score": 85,
                    },
                ],
                "recommended_version": "Problem-Solver Thesis",
                "writing_notes": "Problem-Solver Thesis sounded more senior; Gravity & Scope kept stronger environmental stakes.",
            }
        }
    )
