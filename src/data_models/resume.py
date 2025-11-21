"""
Data Models for Resume Structure
--------------------------------

This module defines the Pydantic data models that represent the canonical
structure of a professional resume. By using these models, we ensure that
resume data is always structured, validated, and type-safe throughout the
entire multi-agent workflow.

WHY PYDANTIC MODELS?
- Data Validation: Automatically validates data types and constraints (e.g., a
  year must be a valid number), preventing common errors.
- Self-Documenting: The models themselves serve as clear documentation for the
  expected data structure.
- IDE Support: Provides excellent autocompletion and type-checking in modern
  IDEs, making development faster and more reliable.
- Serialization: Easily convert data to and from JSON for APIs, storage, or
  agent-to-agent communication.

DESIGN RATIONALE:
- Granularity: The resume is broken down into logical components (`Skill`,
  `Experience`, `Education`) for easier processing and analysis by different agents.
- ATS Compatibility: The structure is designed to be compatible with common
  Applicant Tracking System (ATS) parsing logic.
- Extensibility: The models can be easily extended with new fields as the
  application's requirements grow.
"""

from datetime import date
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ==============================================================================
# 1. Skill Model
# ==============================================================================
# Represents a single skill, such as a programming language or a soft skill.

class Skill(BaseModel):
    """
    Represents a single skill entry in a resume.

    This model captures the essential attributes of a skill, allowing agents to
    categorize, prioritize, and match skills against job requirements.
    """
    # The name of the skill. This is the primary identifier.
    # It should be specific and recognizable (e.g., "Python", "Project Management").
    skill_name: str = Field(
        ...,
        description="The name of the skill (e.g., 'Python', 'Docker', 'Team Leadership').",
        examples=["Python", "Strategic Planning"]
    )

    # The category helps in grouping skills for better organization and display.
    # Agents can use this to differentiate between different types of skills.
    category: Optional[str] = Field(
        None,
        description="The category of the skill (e.g., 'Programming Language', 'Tool', 'Soft Skill').",
        examples=["Technical", "Soft Skill"]
    )

    # Proficiency level provides a qualitative measure of expertise.
    # This can be used by agents to assess the depth of a candidate's knowledge.
    proficiency_level: Optional[str] = Field(
        None,
        description="The self-assessed proficiency level (e.g., 'Beginner', 'Intermediate', 'Expert').",
        examples=["Expert"]
    )

    # Years of experience provides a quantitative measure.
    # This is crucial for matching against job requirements that specify a certain
    # number of years of experience with a technology.
    years_of_experience: Optional[int] = Field(
        None,
        description="The number of years of experience with this skill.",
        ge=0,  # Must be a non-negative number.
        le=50, # A reasonable upper limit.
        examples=[5]
    )
    
    # === AI-FIRST SKILL INFERENCE FIELDS ===
    # These optional fields support the new AI-first skill inference approach,
    # allowing agents to add skills with justification and evidence validation
    
    justification: Optional[str] = Field(
        None,
        description="Explanation for why this skill was added or inferred, based on domain expertise.",
        examples=["Inferred from Generative AI domain expertise - standard tool for building LLM applications"]
    )
    
    evidence: Optional[List[str]] = Field(
        None,
        description="Direct quotes or references from experience section supporting this skill inference.",
        examples=[["Built AI chatbots using Python", "Developed conversational AI assistants"]]
    )
    
    confidence_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence score (0-1) for inferred skills, based on evidence strength and domain relevance.",
        examples=[0.85]
    )

    class Config:
        # Pydantic configuration to provide an example for documentation generation.
        json_schema_extra = {
            "example": {
                "skill_name": "Python",
                "category": "Programming Language",
                "proficiency_level": "Expert",
                "years_of_experience": 8,
            }
        }



# ==============================================================================
# 2. Experience Model
# ==============================================================================
# Represents a single work experience entry (a job at a company).

class Experience(BaseModel):
    """
    Represents a single work experience entry in a resume.

    This model captures all relevant details about a specific job, including the
    role, company, duration, responsibilities, and key achievements.
    """
    job_title: str = Field(
        ...,
        description="The official job title or position held.",
        examples=["Senior Software Engineer"]
    )

    company_name: str = Field(
        ...,
        description="The name of the company.",
        examples=["Tech Innovations Inc."]
    )

    start_date: date = Field(
        ...,
        description="The start date of employment. Using a `date` object allows for precise duration calculations."
    )

    # An end date of `None` signifies that this is the current job.
    end_date: Optional[date] = Field(
        None,
        description="The end date of employment. If `null`, it is the current position."
    )

    # A flag for easier filtering and logic, although it can be inferred from `end_date`.
    is_current_position: bool = Field(
        default=False,
        description="A flag to indicate if this is the candidate's current job."
    )

    location: Optional[str] = Field(
        None,
        description="The location of the job (e.g., 'San Francisco, CA', 'Remote').",
        examples=["Remote"]
    )

    # A general description of the role and responsibilities.
    description: str = Field(
        ...,
        description="A summary of the role, responsibilities, and the team's scope.",
        examples=["Led a team of 5 backend engineers developing a new microservices architecture."]
    )

    # Achievements are the most critical part of an experience entry.
    # They should be quantifiable and demonstrate impact.
    achievements: List[str] = Field(
        default_factory=list,
        description="A list of specific, quantifiable achievements. Each item should be a strong, action-oriented statement.",
        examples=["Reduced API latency by 40% by optimizing database queries."]
    )

    # Linking skills to a specific experience helps agents verify the skills
    # listed in the skills section.
    skills_used: List[str] = Field(
        default_factory=list,
        description="A list of specific skills that were demonstrated or utilized in this role.",
        examples=["Python", "AWS", "FastAPI"]
    )

    @property
    def duration_in_years(self) -> float:
        """
        Calculates the duration of this work experience in years.

        This is a computed property, meaning it's calculated on-the-fly and not
        stored in the database or JSON. This avoids data duplication and ensures
        the duration is always up-to-date.
        """
        # If it's an ongoing job, calculate duration up to the current date.
        end = self.end_date or date.today()
        # Calculate the difference in days and convert to years.
        # Using 365.25 accounts for leap years.
        delta = end - self.start_date
        return round(delta.days / 365.25, 1)

    class Config:
        json_schema_extra = {
            "example": {
                "job_title": "Senior Software Engineer",
                "company_name": "Innovate Corp",
                "start_date": "2020-01-15",
                "end_date": None,
                "is_current_position": True,
                "location": "San Francisco, CA",
                "description": "Led the development of a new real-time analytics platform.",
                "achievements": [
                    "Architected a scalable microservices backend using FastAPI and Kafka, reducing data processing time by 60%.",
                    "Mentored 3 junior engineers, improving team productivity by 25%."
                ],
                "skills_used": ["Python", "Kafka", "AWS", "Docker"],
            }
        }


# ==============================================================================
# 3. Education Model
# ==============================================================================
# Represents an educational background entry (e.g., a university degree).

class Education(BaseModel):
    """
    Represents an educational background entry in a resume.
    """
    institution_name: str = Field(
        ...,
        description="The name of the educational institution (e.g., 'University of California, Berkeley')."
    )

    degree: str = Field(
        ...,
        description="The degree earned (e.g., 'Bachelor of Science', 'PhD').",
        examples=["Bachelor of Science"]
    )

    field_of_study: str = Field(
        ...,
        description="The academic field or major (e.g., 'Computer Science').",
        examples=["Computer Science"]
    )

    graduation_year: int = Field(
        ...,
        description="The year of graduation or expected graduation.",
        ge=1950,
        le=date.today().year + 5 # Allow for future graduation dates.
    )

    gpa: Optional[float] = Field(
        None,
        description="Grade Point Average, if applicable.",
        ge=0.0,
        le=10.0 # Accommodate different scales (4.0, 5.0, or 10.0 point scales).
    )

    honors: Optional[str] = Field(
        None,
        description="Any academic honors or distinctions (e.g., 'Summa Cum Laude').",
        examples=["Cum Laude"]
    )

    class Config:
        json_schema_extra = {
            "example": {
                "institution_name": "Massachusetts Institute of Technology",
                "degree": "Master of Science",
                "field_of_study": "Computer Science",
                "graduation_year": 2020,
                "gpa": 3.9,
                "honors": "Graduated with Distinction",
            }
        }


# ==============================================================================
# 5. Optimized Skills Section Model
# ==============================================================================
# Represents the output of the Skills Section Strategist agent.

class OptimizedSkillsSection(BaseModel):
    """
    Represents an optimized skills section for a resume.

    This model captures the output of the Skills Section Strategist agent,
    including the reordered skills, categorization, and metadata about
    optimizations made. It ensures all skill additions are truthful and
    domain-based.
    """
    optimized_skills: List[Skill] = Field(
        default_factory=list,
        description="The complete list of skills, reordered by priority and relevance to the job.",
    )

    skill_categories: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Skills grouped into logical categories (e.g., 'Programming Languages', 'Cloud Platforms').",
        examples=[{
            "Programming Languages": ["Python", "JavaScript"],
            "Cloud Platforms": ["AWS", "Azure"],
            "Frameworks & Libraries": ["React", "Django"]
        }]
    )

    added_skills: List[Skill] = Field(
        default_factory=list,
        description="Newly inferred skills added based on domain expertise, with justification.",
        examples=[{
            "skill_name": "LangChain",
            "category": "AI Frameworks",
            "proficiency_level": "intermediate",
            "justification": "Inferred from Generative AI experience and Python expertise"
        }]
    )

    removed_skills: List[str] = Field(
        default_factory=list,
        description="Skills removed from the original resume with reasons.",
        examples=["Visual Basic", "COBOL"]
    )

    optimization_notes: str = Field(
        default="",
        description="Summary of optimization decisions and rationale.",
        examples=["Prioritized AWS skills to match job requirements. Added LangChain based on Generative AI experience."]
    )

    ats_match_score: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="ATS compatibility score (0-100) based on keyword matching and optimization.",
    )

    @property
    def total_skills_count(self) -> int:
        """Returns the total number of skills in the optimized section."""
        return len(self.optimized_skills)

    @property
    def category_count(self) -> int:
        """Returns the number of skill categories."""
        return len(self.skill_categories)

    @property
    def added_skills_count(self) -> int:
        """Returns the number of skills added through inference."""
        return len(self.added_skills)

    class Config:
        json_schema_extra = {
            "example": {
                "optimized_skills": [
                    {"skill_name": "Python", "category": "Programming Languages", "proficiency_level": "Expert"},
                    {"skill_name": "AWS", "category": "Cloud Platforms", "proficiency_level": "Advanced"},
                    {"skill_name": "LangChain", "category": "AI Frameworks", "proficiency_level": "Intermediate"}
                ],
                "skill_categories": {
                    "Programming Languages": ["Python", "JavaScript"],
                    "Cloud Platforms": ["AWS", "Docker"],
                    "AI & ML": ["LangChain", "OpenAI API"]
                },
                "added_skills": [
                    {
                        "skill_name": "LangChain",
                        "category": "AI Frameworks",
                        "proficiency_level": "intermediate",
                        "justification": "Inferred from Generative AI project experience"
                    }
                ],
                "removed_skills": ["COBOL"],
                "optimization_notes": "Prioritized must-have AWS skills. Added LangChain based on Generative AI domain expertise.",
                "ats_match_score": 85.0
            }
        }


# ==============================================================================
# 6. Resume Model (Aggregator)
# ==============================================================================
# The main model that aggregates all other components into a complete resume.

class Resume(BaseModel):
    """
    The canonical data model for a complete professional resume.

    This model serves as the central data structure for all resume-related
    information, providing a standardized and validated representation that
    all agents can reliably work with.
    """
    full_name: str = Field(
        ...,
        description="The candidate's full legal name."
    )

    email: str = Field(
        ...,
        description="The primary email address for professional communication."
    )

    phone_number: Optional[str] = Field(
        None,
        description="Phone number, preferably with a country code."
    )

    location: Optional[str] = Field(
        None,
        description="Current city and state/country, or 'Remote'."
    )

    website_or_portfolio: Optional[str] = Field(
        None,
        description="URL to a personal website, portfolio, or professional profile (e.g., GitHub, LinkedIn)."
    )

    professional_summary: str = Field(
        ...,
        description="A 2-4 sentence professional summary that encapsulates the candidate's expertise and career goals."
    )

    work_experience: List[Experience] = Field(
        default_factory=list,
        description="A list of professional work experiences, typically sorted from most to least recent."
    )

    education: List[Education] = Field(
        default_factory=list,
        description="A list of educational qualifications."
    )

    skills: List[Skill] = Field(
        default_factory=list,
        description="A comprehensive list of the candidate's technical and soft skills."
    )

    certifications: List[str] = Field(
        default_factory=list,
        description="A list of professional certifications (e.g., 'AWS Certified Solutions Architect')."
    )

    languages: List[str] = Field(
        default_factory=list,
        description="Languages spoken, optionally with proficiency levels (e.g., 'Spanish (Conversational)')."
    )

    @property
    def total_years_of_experience(self) -> float:
        """
        Calculates the total years of professional experience from all jobs.

        This is a business logic method that provides a high-level summary of the
        candidate's experience, which is useful for seniority assessment.
        """
        if not self.work_experience:
            return 0.0
        
        # This simple sum works for non-overlapping experiences. For a more robust
        # calculation, one might need to handle overlapping job durations.
        return sum(exp.duration_in_years for exp in self.work_experience)

    @property
    def list_of_skill_names(self) -> List[str]:
        """Returns a simple list of all skill names for quick lookups."""
        return [skill.skill_name for skill in self.skills]

    class Config:
        json_schema_extra = {
            "example": {
                "full_name": "Jane Doe",
                "email": "jane.doe@example.com",
                "phone_number": "+1-555-0123",
                "location": "New York, NY",
                "website_or_portfolio": "https://github.com/janedoe",
                "professional_summary": "Experienced software engineer with over 10 years in cloud computing and distributed systems.",
                "work_experience": [], # Add examples of Experience model here
                "education": [],     # Add examples of Education model here
                "skills": [],        # Add examples of Skill model here
                "certifications": ["AWS Certified Developer - Associate"],
                "languages": ["English (Native)", "German (Basic)"],
            }
        }
