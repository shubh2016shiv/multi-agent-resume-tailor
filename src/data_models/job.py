"""
Data Models for Job Description Structure
-----------------------------------------

This module defines the Pydantic models for representing a job description in a
structured, machine-readable format. These models are crucial for the analytical
agents to understand the requirements, skills, and context of a job posting.

WHY STRUCTURED MODELS FOR JOBS?
- Consistency: Job descriptions from different sources can be normalized into a
  single, consistent format.
- Analysis: Agents can programmatically analyze requirements, identify keywords,
  and assess seniority levels.
- Prioritization: By using enums like `SkillImportance`, agents can differentiate
  between "must-have" and "nice-to-have" skills, which is critical for effective
  resume tailoring.
- Type Safety: Ensures that all parts of the system are working with the same
  validated data structure.
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# ==============================================================================
# 1. Enumerations for Standardization
# ==============================================================================
# Enums provide a controlled vocabulary for classifying job attributes.

class JobLevel(str, Enum):
    """
    Standardized job seniority levels for consistent classification.

    This enum helps agents understand the expected experience level for a role,
    which influences the tone and focus of the tailored resume.
    """
    ENTRY = "entry"
    JUNIOR = "junior"
    MID = "mid-level"
    SENIOR = "senior"
    LEAD = "lead"
    PRINCIPAL = "principal"
    MANAGER = "manager"
    DIRECTOR = "director"
    EXECUTIVE = "executive"
    UNSPECIFIED = "unspecified"

class SkillImportance(str, Enum):
    """
    Skill requirement criticality levels for prioritization.

    This enum is vital for the gap analysis agent to determine which skills are
    deal-breakers (`MUST_HAVE`) versus those that are simply advantageous.
    """
    MUST_HAVE = "must_have"        # A non-negotiable requirement.
    SHOULD_HAVE = "should_have"      # A strongly preferred skill.
    NICE_TO_HAVE = "nice_to_have"    # A bonus skill that provides an edge.


# ==============================================================================
# 2. Job Requirement Model
# ==============================================================================
# Represents a single skill or qualification mentioned in the job description.

class JobRequirement(BaseModel):
    """
    Represents a single, specific requirement from a job posting.
    
    Breaking down a job description into a list of these structured requirements
    allows agents to perform granular analysis and matching.
    """
    # The name of the skill or requirement.
    requirement: str = Field(
        ...,
        description="The specific skill, technology, or competency required (e.g., 'Python', 'Project Management').",
        examples=["Python"]
    )

    # Its importance level, as determined by the analysis agent.
    importance: SkillImportance = Field(
        default=SkillImportance.SHOULD_HAVE,
        description="The business criticality of the requirement."
    )

    # Optional: The minimum years of experience needed for this specific skill.
    years_required: Optional[int] = Field(
        None,
        description="Minimum years of experience required for this specific skill.",
        ge=0,
        le=30
    )

    class Config:
        json_schema_extra = {
            "example": {
                "requirement": "Experience with AWS",
                "importance": "must_have",
                "years_required": 5,
            }
        }


# ==============================================================================
# 3. Job Description Model (Aggregator)
# ==============================================================================
# The main model that aggregates all information about a job posting.

class JobDescription(BaseModel):
    """
    A canonical, structured representation of a complete job posting.

    This model serves as the comprehensive data structure that the agents use
    for all analysis and strategy generation.
    """
    job_title: str = Field(
        ...,
        description="The official job title.",
        examples=["Senior Backend Engineer"]
    )

    company_name: str = Field(
        ...,
        description="The name of the hiring company.",
        examples=["Cloud Solutions Inc."]
    )

    # Uses the `JobLevel` enum for standardized seniority assessment.
    job_level: JobLevel = Field(
        default=JobLevel.UNSPECIFIED,
        description="The standardized seniority level of the role."
    )

    location: Optional[str] = Field(
        None,
        description="Job location or work arrangement (e.g., 'San Francisco, CA', 'Remote')."
    )

    # A brief, high-level summary of the role.
    summary: str = Field(
        ...,
        description="A brief, 1-3 sentence overview of the position.",
        examples=["Seeking a skilled backend engineer to build and maintain our cloud infrastructure."]
    )
    
    # The full, raw text of the job description for context.
    full_text: str = Field(
        ...,
        description="The complete, original text of the job description."
    )

    # The structured list of requirements, extracted by the analysis agent.
    requirements: List[JobRequirement] = Field(
        default_factory=list,
        description="A structured list of skills and qualifications with their importance."
    )

    # A list of keywords that are critical for ATS (Applicant Tracking System) matching.
    ats_keywords: List[str] = Field(
        default_factory=list,
        description="A list of high-priority keywords identified for ATS optimization.",
        examples=["Python", "AWS", "Microservices", "API Design"]
    )

    @property
    def must_have_skills(self) -> List[str]:
        """
        A computed property that returns a list of all "must-have" skill names.
        
        This is useful for quick access during the gap analysis phase, allowing the
        strategy agent to immediately identify critical skill gaps.
        """
        return [
            req.requirement
            for req in self.requirements
            if req.importance == SkillImportance.MUST_HAVE
        ]

    @property
    def preferred_skills(self) -> List[str]:
        """
        A computed property that returns a list of "should-have" and "nice-to-have" skills.
        """
        return [
            req.requirement
            for req in self.requirements
            if req.importance in (SkillImportance.SHOULD_HAVE, SkillImportance.NICE_TO_HAVE)
        ]

    class Config:
        json_schema_extra = {
            "example": {
                "job_title": "Senior Backend Engineer",
                "company_name": "Cloud Solutions Inc.",
                "job_level": "senior",
                "location": "Remote",
                "summary": "Join our team to build the next generation of cloud services.",
                "full_text": "...",
                "requirements": [
                    {"requirement": "Python", "importance": "must_have", "years_required": 5},
                    {"requirement": "Go", "importance": "nice_to_have"}
                ],
                "ats_keywords": ["Python", "AWS", "API"]
            }
        }
