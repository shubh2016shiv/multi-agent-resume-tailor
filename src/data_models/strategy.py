"""
Data Models for Resume-Job Alignment Strategy
---------------------------------------------

This module defines the Pydantic models that capture the strategic output
from the `Gap & Alignment Strategist` agent. These models serve as a
structured "plan of attack" for the content-generating agents.

WHY THESE MODELS?
- Actionable Guidance: Instead of a vague text blob, this structure provides
  specific, actionable instructions that the writing agents can follow.
- Clarity of Purpose: It clearly separates the "analysis" phase from the
  "content generation" phase. The strategy is the bridge between them.
- Debuggability: If the final resume is poor, you can inspect the strategy
  object to see if the plan was flawed from the start.
- Consistency: Ensures that all content-generating agents are working from the
  same strategic plan, leading to a coherent final document.
"""

from pydantic import BaseModel, Field

# ==============================================================================
# 1. Skill Match Model
# ==============================================================================
# Represents the identified alignment between a skill from the resume and a
# requirement from the job description.


class SkillMatch(BaseModel):
    """
    Represents a successfully identified match between a resume skill and a job requirement.
    """

    # The skill as it appears in the candidate's resume.
    resume_skill: str = Field(
        ...,
        description="The specific skill identified in the candidate's resume.",
    )

    # The corresponding requirement from the job description.
    job_requirement: str = Field(
        ...,
        description="The requirement from the job posting that this skill matches.",
    )

    # A score indicating the quality of the match. This helps prioritize which
    # skills to emphasize most prominently.
    match_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="A quantitative assessment of the skill alignment (0=low, 100=perfect).",
    )

    # A brief explanation of why this is considered a match. This is crucial for
    # debugging and for the writing agents to understand the context.
    justification: str = Field(
        ...,
        description="A brief explanation of why this is considered a match, especially if it's not a direct 1:1 mapping.",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "resume_skill": "Team Leadership",
                "job_requirement": "Ability to mentor junior engineers",
                "match_score": 90.0,
                "justification": "Leadership experience is directly transferable to mentorship.",
            }
        }


# ==============================================================================
# 2. Skill Gap Model
# ==============================================================================
# Represents a required skill from the job description that is missing or
# underrepresented in the resume.


class SkillGap(BaseModel):
    """
    Represents a required skill that is missing or underrepresented in the resume.
    """

    # The skill that the job requires but is not found in the resume.
    missing_skill: str = Field(
        ...,
        description="The skill required by the job but not found or emphasized in the resume.",
    )

    # The importance of the missing skill, taken from the job analysis.
    importance: str = Field(
        ...,
        description="The business criticality of the missing skill: 'must_have', 'should_have', or 'nice_to_have'.",
    )

    # An actionable suggestion for the content-writing agents on how to handle this gap.
    # This is the core of the "strategy".
    suggestion: str = Field(
        ...,
        description="An actionable recommendation on how to address this gap.",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "missing_skill": "[REQUIRED_SKILL]",
                "importance": "should_have",
                "suggestion": "Resume shows [related evidence]; prioritize the role/skill where it appears so it is visible. If truly absent, acknowledge honestly -- do not fabricate.",
            }
        }


# ==============================================================================
# 3. Alignment Strategy Model (Aggregator)
# ==============================================================================
# The main model that aggregates all strategic guidance into a single, cohesive plan.


class AlignmentStrategy(BaseModel):
    """
    A comprehensive, structured plan for tailoring the resume.

    This model is the final output of the `Gap & Alignment Strategist` agent and serves
    as the primary input for all subsequent content generation agents.

    Architecture: This model implements the **Blackboard Pattern** (Strategy variant —
    one writer, many readers). The Gap Analysis agent WRITES the strategy once; three
    content-generation agents (Summary Writer, Experience Optimizer, Skills Optimizer)
    READ their section-specific guidance fields in parallel. No downstream agent
    re-analyzes Resume vs Job — the analysis happens once, the results are distributed.

    See: docs/blackboard-pattern-multi-agent-systems.md
    """

    # FIELD ORDER IS DELIBERATE: structured-output models fill fields top-to-bottom and
    # earlier fields condition later ones. The analysis (matches, gaps, keywords) comes
    # FIRST so it is grounded in the supplied current_match_report; overall_fit_score is
    # the VERDICT derived from that analysis; the three guidance fields come LAST because
    # each depends on everything above. Downstream agents read by field name, so the
    # reorder is invisible to them.

    # Matches grounded in the candidate profile and the code-computed match report.
    identified_matches: list[SkillMatch] = Field(
        default_factory=list,
        description=(
            "Skills/requirements the candidate already satisfies. Must agree with the "
            "supplied current_match_report -- do not invent matches it did not find."
        ),
    )

    # Gaps grounded in the code-computed match report findings.
    identified_gaps: list[SkillGap] = Field(
        default_factory=list,
        description=(
            "Required skills the candidate is missing or under-evidences, taken from the "
            "current_match_report findings. Each suggestion must be a TRUTHFUL handling "
            "(surface existing evidence, or acknowledge honestly) -- never fabrication."
        ),
    )

    # Keywords to surface, derived from the JD and the match report's missing-keyword findings.
    keywords_to_integrate: list[str] = Field(
        default_factory=list,
        description=(
            "Prioritized JD keywords for downstream agents to surface (must-haves first). "
            "Cover every missing must-have keyword from current_match_report. Do not pad to "
            "a fixed count and do not include keywords the candidate cannot truthfully support."
        ),
    )

    # The verdict, anchored to the code-computed match score (do not recompute by hand).
    overall_fit_score: float = Field(
        ...,
        ge=0,
        le=100,
        description=(
            "Overall compatibility (0-100). Anchor this to current_match_report.score "
            "(the authoritative must-have coverage); adjust only slightly and only with a "
            "reason in summary_of_strategy. Do NOT recompute via a hand-weighted formula."
        ),
    )

    # The core strategy that ties matches, gaps, and the score together.
    summary_of_strategy: str = Field(
        ...,
        description="A concise summary of the tailoring strategy and any adjustment to the anchored score.",
    )

    # Guidance for the Professional Summary Writer (it evidence-gates keywords itself).
    professional_summary_guidance: str = Field(
        ...,
        description=(
            "Instructions for the Summary Writer: which truthful strengths to lead with and "
            "which supported keywords to prioritize. It will keyword-gate against resume "
            "evidence, so do not ask it to claim anything the resume does not support."
        ),
    )

    # Guidance for the Experience Optimizer -- which is REORDER-ONLY (verbatim bullets).
    experience_guidance: str = Field(
        ...,
        description=(
            "Instructions for the Experience Optimizer, which may ONLY reorder existing "
            "bullets verbatim -- it cannot rewrite, reframe, add metrics, or merge. So guide "
            "it on WHICH roles/bullets to prioritize for this job, never on rewording them."
        ),
    )

    # Guidance for the Skills Optimizer -- which reorders + adds evidenced skills, never drops.
    skills_guidance: str = Field(
        ...,
        description=(
            "Instructions for the Skills Optimizer, which reorders and groups skills and may "
            "add only evidence-backed ones -- it never drops a truthful skill. Guide it on "
            "ordering, grouping, and which supported JD terms to surface first."
        ),
    )

    class Config:
        # Domain-neutral example. [PLACEHOLDERS] stand for the candidate's actual field
        # (nursing, finance, logistics, law, software, ...). It teaches STRUCTURE, not a
        # domain, and experience_guidance shows REORDERING (not rewriting) on purpose.
        json_schema_extra = {
            "example": {
                "identified_matches": [],
                "identified_gaps": [],
                "keywords_to_integrate": ["[MUST_HAVE_KEYWORD_1]", "[MUST_HAVE_KEYWORD_2]"],
                "overall_fit_score": 78.5,
                "summary_of_strategy": "Lead with [strongest supported area]; honestly note the [gap] from the match report.",
                "professional_summary_guidance": "Open with [supported strength]; prioritize keywords [A], [B].",
                "experience_guidance": "Rank the [most relevant role]'s bullets about [topic] first; de-emphasize unrelated bullets by ordering, not removal.",
                "skills_guidance": "Order [must-have skill] first; group [related skills]; surface [supported JD term].",
            }
        }
