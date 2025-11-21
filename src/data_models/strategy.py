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
        examples=["Django Framework"],
    )

    # The corresponding requirement from the job description.
    job_requirement: str = Field(
        ...,
        description="The requirement from the job posting that this skill matches.",
        examples=["Experience with Python web frameworks"],
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
        examples=["Django is a popular Python web framework, directly fulfilling the requirement."],
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
        examples=["Kubernetes"],
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
        examples=[
            "Review project experience for any container orchestration work that can be highlighted as a transferable skill."
        ],
    )

    class Config:
        json_schema_extra = {
            "example": {
                "missing_skill": "Terraform",
                "importance": "should_have",
                "suggestion": "Since the resume mentions AWS CloudFormation, reframe one achievement to highlight 'Infrastructure as Code (IaC)' experience and mention CloudFormation as the tool used.",
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
    """

    # A high-level assessment of how well the candidate's profile matches the job.
    overall_fit_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="A 0-100 score representing the overall compatibility of the resume with the job description.",
    )

    # A summary of the core strategy, telling the other agents what the main focus should be.
    summary_of_strategy: str = Field(
        ...,
        description="A concise summary of the overall tailoring strategy.",
        examples=[
            "Emphasize leadership and cloud experience, while downplaying legacy system skills. Reframe project achievements to use keywords from the job description."
        ],
    )

    # A list of all identified skill matches.
    identified_matches: list[SkillMatch] = Field(
        default_factory=list,
        description="A list of skills from the resume that align well with the job requirements.",
    )

    # A list of all identified skill gaps and how to address them.
    identified_gaps: list[SkillGap] = Field(
        default_factory=list,
        description="A list of required skills missing from the resume, with suggestions on how to mitigate them.",
    )

    # A curated list of keywords that MUST be included in the final resume for ATS optimization.
    keywords_to_integrate: list[str] = Field(
        default_factory=list,
        description="A prioritized list of keywords that must be naturally integrated into the resume.",
    )

    # Specific guidance for the agent that writes the professional summary.
    professional_summary_guidance: str = Field(
        ...,
        description="Specific instructions for the Professional Summary Writer agent.",
        examples=[
            "Focus on the 5+ years of Python experience and the recent AWS certification. Mention leadership in the context of 'mentoring junior developers'."
        ],
    )

    # Specific guidance for the agent that optimizes the work experience section.
    experience_guidance: str = Field(
        ...,
        description="Specific instructions for the Experience Section Optimizer agent.",
        examples=[
            "For the 'Innovate Corp' role, rewrite the first bullet point to include the keyword 'microservices'. For the 'Old Tech Co' role, condense it to two bullet points as it is less relevant."
        ],
    )

    # Specific guidance for the agent that optimizes the skills section.
    skills_guidance: str = Field(
        ...,
        description="Specific instructions for the Skills Section Strategist agent.",
        examples=[
            "Create a 'Cloud Technologies' category and list AWS, Docker, and Kubernetes first. Remove 'Visual Basic' as it is not relevant."
        ],
    )

    class Config:
        json_schema_extra = {
            "example": {
                "overall_fit_score": 78.5,
                "summary_of_strategy": "Highlight cloud and IaC experience.",
                "identified_matches": [],
                "identified_gaps": [],
                "keywords_to_integrate": ["AWS", "Terraform", "CI/CD"],
                "professional_summary_guidance": "Start with 'Senior Cloud Engineer with 8 years of experience...'",
                "experience_guidance": "Reframe the project at TechCorp to emphasize cost savings.",
                "skills_guidance": "Prioritize AWS services in the skills list.",
            }
        }
