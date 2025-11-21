"""
Data Models for Resume Quality Evaluation
-----------------------------------------

This module defines the Pydantic models for the quantitative evaluation of a
tailored resume. The `Quality Assurance Reviewer` agent uses these models to
generate a structured, data-driven quality report.

WHY THESE MODELS?
- Objectivity: Replaces subjective feedback with measurable, quantitative scores,
  making the quality assessment process consistent and reliable.
- Multi-Dimensional Analysis: Breaks down "quality" into three distinct,
  understandable dimensions: Accuracy, Relevance, and ATS Optimization.
- Debuggability: If a resume fails the quality check, these models provide a
  clear, structured report detailing exactly what went wrong and why.
- Actionable Feedback: The structure of the models (e.g., lists of specific
  issues) provides a clear basis for the iterative refinement loop.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ==============================================================================
# 1. Accuracy Metrics Model
# ==============================================================================
# Measures the truthfulness of the tailored resume against the original.

class AccuracyMetrics(BaseModel):
    """
    Measures the truthfulness of the tailored resume by comparing it against the original.
    
    This is the most critical metric because it ensures the system does not invent or
    exaggerate the candidate's qualifications. Honesty is non-negotiable.
    """
    # A score from 0-100 representing the overall accuracy.
    accuracy_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="A 0-100 score for accuracy, where 100 is perfect alignment with the original resume."
    )

    # A list of specific claims in the tailored resume that appear to be exaggerated.
    exaggerated_claims: List[str] = Field(
        default_factory=list,
        description="A list of claims in the tailored resume that are not supported by or exaggerate the original resume.",
        examples=["Claiming 'expert' proficiency for a skill listed as 'intermediate' in the original."]
    )

    # A list of skills that were added to the tailored resume but were not present
    # in the original, which could be a form of fabrication.
    unsupported_skills: List[str] = Field(
        default_factory=list,
        description="A list of skills added to the tailored resume that could not be found or inferred from the original.",
        examples=["Added 'Kubernetes' to the skills list when it was not mentioned anywhere in the original resume."]
    )

    # A brief explanation of the accuracy findings.
    justification: str = Field(
        ...,
        description="A summary of the reasoning behind the accuracy score and findings."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "accuracy_score": 85.0,
                "exaggerated_claims": ["Changed 'managed 3 projects' to 'led 5 major projects'."],
                "unsupported_skills": [],
                "justification": "Score was reduced due to one exaggerated claim regarding project management scope."
            }
        }


# ==============================================================================
# 2. Relevance Metrics Model
# ==============================================================================
# Measures how well the tailored resume addresses the job description.

class RelevanceMetrics(BaseModel):
    """
    Measures how relevant the tailored resume is to the target job description.

    This metric ensures the resume is not just a generic document but is sharply
    focused on the specific role, addressing the employer's needs directly.
    """
    # A score from 0-100 representing the overall relevance.
    relevance_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="A 0-100 score for relevance, where 100 means all job requirements are perfectly addressed."
    )

    # The percentage of "must-have" skills from the job description that were
    # successfully addressed in the tailored resume.
    must_have_skills_coverage: float = Field(
        ...,
        ge=0,
        le=100,
        description="The percentage of 'must-have' job requirements that are addressed in the resume."
    )

    # A list of important requirements from the job that were not adequately addressed.
    # This is critical feedback for the next iteration.
    missed_requirements: List[str] = Field(
        default_factory=list,
        description="A list of important job requirements that were not adequately addressed in the tailored resume.",
        examples=["The job description asked for CI/CD experience, which was not highlighted."]
    )

    # A brief explanation of the relevance findings.
    justification: str = Field(
        ...,
        description="A summary of the reasoning behind the relevance score and findings."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "relevance_score": 90.0,
                "must_have_skills_coverage": 100.0,
                "missed_requirements": ["Did not emphasize experience with Agile methodologies, which was a preferred skill."],
                "justification": "Excellent coverage of all must-have skills, with a minor missed opportunity on a preferred skill."
            }
        }


# ==============================================================================
# 3. ATS Optimization Metrics Model
# ==============================================================================
# Measures the resume's compatibility with Applicant Tracking Systems.

class ATSMetrics(BaseModel):
    """
    Measures the resume's technical compatibility with Applicant Tracking Systems (ATS).

    This metric focuses on the structural and keyword-related aspects that determine
    whether a resume will be successfully parsed and ranked by automated systems.
    """
    # A score from 0-100 representing the overall ATS compatibility.
    ats_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="A 0-100 score for ATS compatibility, where 100 is perfectly optimized."
    )

    # The percentage of critical keywords from the job description that are present.
    keyword_coverage: float = Field(
        ...,
        ge=0,
        le=100,
        description="The percentage of essential keywords from the job description found in the resume."
    )

    # A list of any formatting issues that could hinder ATS parsing.
    formatting_issues: List[str] = Field(
        default_factory=list,
        description="A list of formatting problems that might cause issues for ATS parsers.",
        examples=["Use of tables or columns in the experience section.", "Uncommon section headers."]
    )
    
    # A brief explanation of the ATS findings.
    justification: str = Field(
        ...,
        description="A summary of the reasoning behind the ATS score and findings."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "ats_score": 95.0,
                "keyword_coverage": 100.0,
                "formatting_issues": [],
                "justification": "Strong keyword coverage and clean, parsable formatting."
            }
        }


# ==============================================================================
# 4. Quality Report Model (Aggregator)
# ==============================================================================
# The main model that aggregates all evaluation metrics into a final report.

class QualityReport(BaseModel):
    """
    A comprehensive, structured report on the quality of the tailored resume.

    This is the final output of the `Quality Assurance Reviewer` agent, providing a
    conclusive, data-driven assessment with a final score and actionable feedback.
    """
    # The final, weighted score for the resume's quality.
    overall_quality_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="The final, weighted quality score (0-100)."
    )
    
    # A boolean flag indicating if the resume met the required quality threshold.
    passed_quality_threshold: bool = Field(
        ...,
        description="Whether the resume's score meets the minimum required quality threshold."
    )

    # A high-level summary of the quality assessment.
    assessment_summary: str = Field(
        ...,
        description="A brief, high-level summary of the overall quality assessment."
    )

    # Detailed breakdown of the three core quality dimensions.
    accuracy: AccuracyMetrics
    relevance: RelevanceMetrics
    ats_optimization: ATSMetrics

    # If the quality check failed, this field contains specific, actionable
    # feedback for the next iteration of the workflow.
    feedback_for_improvement: Optional[str] = Field(
        None,
        description="If quality check failed, this provides specific, actionable feedback for the next iteration.",
        examples=["The accuracy score is too low. Re-run the Experience Optimizer agent with instructions to not exaggerate metrics."]
    )

    class Config:
        json_schema_extra = {
            "example": {
                "overall_quality_score": 88.0,
                "passed_quality_threshold": True,
                "assessment_summary": "The resume is of high quality, accurately reflects the candidate's skills, and is well-aligned with the job.",
                "accuracy": {"accuracy_score": 90.0, "exaggerated_claims": [], "unsupported_skills": [], "justification": "All claims are well-supported."},
                "relevance": {"relevance_score": 90.0, "must_have_skills_coverage": 100.0, "missed_requirements": [], "justification": "Excellent alignment with job needs."},
                "ats_optimization": {"ats_score": 85.0, "keyword_coverage": 95.0, "formatting_issues": [], "justification": "Good keyword density and clean formatting."},
                "feedback_for_improvement": None,
            }
        }


