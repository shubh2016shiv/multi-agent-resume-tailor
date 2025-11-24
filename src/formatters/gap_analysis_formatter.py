"""
Gap Analysis Agent Formatter
=============================
Formats and filters data specifically for the Gap Analysis Agent to reduce
token usage while preserving all information needed for strategic alignment analysis.

The Gap Analysis Agent compares resumes against job descriptions to identify:
- Skill matches and alignment scores
- Critical skill gaps and missing requirements
- Strategic optimization opportunities
- ATS keyword recommendations

This formatter:
1. Extracts essential resume fields for gap comparison (skills, experience, education, certifications)
2. Extracts essential job fields for gap comparison (requirements, ATS keywords, job metadata)
3. Converts data to TOON or Markdown format based on format_type parameter
4. Removes redundant metadata and nested structures that don't contribute to gap analysis

Expected token reduction: ~55-70% with TOON format compared to full JSON serialization.

Design Philosophy:
-----------------
- **Clear Separation of Concerns**: Each helper function handles one specific data extraction task
- **Descriptive Naming**: Function and variable names clearly indicate purpose and content
- **Reduced Cognitive Overload**: Complex models broken down into simple, focused transformations
- **Debuggability**: Extensive logging at each transformation step with token metrics
- **No Data Loss**: All fields required for gap analysis are preserved
"""

from typing import Any

from src.core.logger import get_logger
from src.data_models.job import JobDescription
from src.data_models.resume import Resume
from src.formatters.base_formatter import FormatType, estimate_tokens, format_data

logger = get_logger(__name__)


# ==============================================================================
# RESUME DATA EXTRACTION FOR GAP ANALYSIS
# ==============================================================================
# The Gap Analysis Agent needs to compare what the candidate HAS versus what
# the job REQUIRES. These functions extract only the relevant "what they have"
# information from the resume.
# ==============================================================================


def _extract_skills_for_comparison(resume: Resume) -> dict[str, Any]:
    """
    Extract structured skill data for gap comparison.

    The Gap Analysis Agent needs to:
    - Compare resume skills against job requirements
    - Identify skill gaps and matches
    - Calculate alignment scores

    We extract:
    - Skill names for direct comparison
    - Proficiency levels to assess strength of match
    - Skill categories for grouping and analysis

    Args:
        resume: The Resume object containing candidate skills

    Returns:
        Dictionary with structured skill data ready for gap analysis

    Example Output:
        {
            "skills": [
                {"name": "Python", "proficiency": "Expert", "category": "Programming"},
                {"name": "AWS", "proficiency": "Advanced", "category": "Cloud"}
            ]
        }
    """
    skills_list = []

    for skill in resume.skills:
        skill_entry = {
            "name": skill.skill_name,
            "proficiency": skill.proficiency_level if skill.proficiency_level else "Not Specified",
        }

        # Include category if available for better grouping
        if skill.category:
            skill_entry["category"] = skill.category

        skills_list.append(skill_entry)

    logger.debug(f"Extracted {len(skills_list)} skills from resume for gap analysis")

    return {"skills": skills_list}


def _extract_experience_summaries_for_comparison(resume: Resume) -> dict[str, Any]:
    """
    Extract work experience summaries for skill and domain alignment.

    The Gap Analysis Agent needs to:
    - Identify domain experience alignment with job requirements
    - Find implicit skills mentioned in experience descriptions
    - Assess years of experience and seniority level

    We extract:
    - Job titles and companies for context
    - Duration of each role for experience calculation
    - Key responsibilities/achievements (condensed to reduce tokens)
    - Technologies/skills mentioned (for implicit skill matching)

    Args:
        resume: The Resume object containing work experience

    Returns:
        Dictionary with condensed experience data focused on skills and domains

    Example Output:
        {
            "experience": [
                {
                    "title": "Senior Engineer",
                    "company": "Tech Corp",
                    "duration": "2020-01-15 - 2023-12-31",
                    "description": "Led development of analytics platform",
                    "achievements": ["Architected microservices backend", "Mentored 3 junior engineers"],
                    "skills_used": ["Python", "Kafka", "AWS"]
                }
            ]
        }
    """
    experience_list = []

    for exp in resume.work_experience:
        # Build duration string for experience calculation
        duration = f"{exp.start_date} - {exp.end_date if exp.end_date else 'Present'}"

        experience_entry = {
            "title": exp.job_title,
            "company": exp.company_name,
            "duration": duration,
        }

        # Include description (role summary) for context
        if exp.description:
            experience_entry["description"] = exp.description

        # Extract key achievements (limit to reduce tokens while preserving context)
        # The Gap Analysis Agent needs these to identify implicit skills and domain expertise
        if exp.achievements:
            # Take first 3 achievements or all if fewer - this gives enough context
            # for gap analysis without overwhelming the token budget
            key_achievements = exp.achievements[:3]
            experience_entry["achievements"] = key_achievements

        # Include skills_used if available (helps with skill matching)
        if exp.skills_used:
            experience_entry["skills_used"] = exp.skills_used

        experience_list.append(experience_entry)

    logger.debug(f"Extracted {len(experience_list)} experience entries for gap analysis")

    return {"experience": experience_list}


def _extract_education_and_certifications(resume: Resume) -> dict[str, Any]:
    """
    Extract education and certifications for requirement matching.

    The Gap Analysis Agent needs to:
    - Match education requirements (e.g., "Bachelor's in CS required")
    - Identify relevant certifications (e.g., AWS Certified, PMP)
    - Assess domain-specific qualifications

    We extract:
    - Degree types and fields of study
    - Institutions (sometimes required, e.g., "Tier 1 university")
    - Certification names and issuing organizations
    - Years/dates for recency assessment

    Args:
        resume: The Resume object containing education and certifications

    Returns:
        Dictionary with education and certification data for gap matching

    Example Output:
        {
            "education": [
                {
                    "degree": "Bachelor of Science",
                    "field": "Computer Science",
                    "institution": "MIT",
                    "graduation_year": 2020,
                    "gpa": 3.9
                }
            ],
            "certifications": [
                "AWS Certified Solutions Architect",
                "PMP Certification"
            ]
        }
    """
    result = {}

    # Extract education details
    if resume.education:
        education_list = []
        for edu in resume.education:
            education_entry = {
                "degree": edu.degree,
                "field": edu.field_of_study,
                "institution": edu.institution_name,
            }

            # Include graduation year (required field, useful for recency checks)
            education_entry["graduation_year"] = edu.graduation_year

            # Include GPA if available (some jobs have GPA requirements)
            if edu.gpa is not None:
                education_entry["gpa"] = edu.gpa

            # Include honors if available (distinguishes candidates)
            if edu.honors:
                education_entry["honors"] = edu.honors

            education_list.append(education_entry)

        result["education"] = education_list
        logger.debug(f"Extracted {len(education_list)} education entries")

    # Extract certifications
    # Note: certifications is a list[str], not a list of objects
    if resume.certifications:
        # Simply use the certification strings as-is
        result["certifications"] = resume.certifications
        logger.debug(f"Extracted {len(resume.certifications)} certifications")

    return result


def _extract_additional_resume_context(resume: Resume) -> dict[str, Any]:
    """
    Extract additional resume context that may influence gap analysis.

    The Gap Analysis Agent may need:
    - Languages (some jobs require multilingual capabilities)
    - Total years of experience (for seniority matching)
    - Current location (for remote/on-site alignment)

    We extract:
    - Languages spoken (if specified in resume)
    - Any other contextual fields that inform strategic decisions

    Args:
        resume: The Resume object

    Returns:
        Dictionary with additional contextual information

    Example Output:
        {
            "languages": ["English", "Spanish"],
            "location": "San Francisco, CA"
        }
    """
    additional_context = {}

    # Extract languages if available
    # Note: languages is a list[str], not a list of objects
    if resume.languages:
        additional_context["languages"] = resume.languages
        logger.debug(f"Extracted {len(resume.languages)} languages")

    # Extract location if available (some jobs have location requirements)
    if resume.location:
        additional_context["location"] = resume.location

    return additional_context


# ==============================================================================
# JOB DESCRIPTION DATA EXTRACTION FOR GAP ANALYSIS
# ==============================================================================
# The Gap Analysis Agent needs to know what the job REQUIRES to perform
# effective comparison. These functions extract only the requirement-related
# information from the job description.
# ==============================================================================


def _extract_job_requirements_for_comparison(job_description: JobDescription) -> dict[str, Any]:
    """
    Extract structured job requirements for gap identification.

    The Gap Analysis Agent needs to:
    - Compare required skills against candidate skills
    - Prioritize gaps based on requirement importance
    - Generate strategic recommendations for each gap

    We extract:
    - Requirement text/skill names
    - Importance levels (must_have, should_have, nice_to_have)
    - Category groupings if available

    Args:
        job_description: The JobDescription object with parsed requirements

    Returns:
        Dictionary with structured requirements for gap analysis

    Example Output:
        {
            "requirements": [
                {"requirement": "5+ years Python", "importance": "must_have", "category": "technical"},
                {"requirement": "Team leadership", "importance": "should_have", "category": "soft_skills"}
            ]
        }
    """
    requirements_list = []

    for req in job_description.requirements:
        requirement_entry = {
            "requirement": req.requirement,
            "importance": req.importance,
        }

        # Include category if available for better grouping in analysis
        if hasattr(req, "category") and req.category:
            requirement_entry["category"] = req.category

        requirements_list.append(requirement_entry)

    logger.debug(f"Extracted {len(requirements_list)} requirements from job description")

    return {"requirements": requirements_list}


def _extract_ats_keywords_for_optimization(job_description: JobDescription) -> dict[str, Any]:
    """
    Extract ATS keywords for resume optimization strategy.

    The Gap Analysis Agent needs to:
    - Identify which ATS keywords are already present in resume
    - Recommend strategic keyword integration
    - Prioritize keywords for ATS matching

    Note: ats_keywords is a list[str], not a list of objects.

    Args:
        job_description: The JobDescription object with parsed ATS keywords

    Returns:
        Dictionary with ATS keywords for strategic integration

    Example Output:
        {
            "ats_keywords": ["Python", "AWS", "Agile", "Docker"]
        }
    """
    # ats_keywords is a simple list of strings, not objects
    if job_description.ats_keywords:
        logger.debug(f"Extracted {len(job_description.ats_keywords)} ATS keywords for optimization")
        return {"ats_keywords": job_description.ats_keywords}

    return {"ats_keywords": []}


def _extract_job_metadata_for_context(job_description: JobDescription) -> dict[str, Any]:
    """
    Extract job metadata that provides context for gap analysis.

    The Gap Analysis Agent needs:
    - Job title (for seniority and role alignment)
    - Company name (for industry/domain context)
    - Job level (for experience matching)
    - Summary (for overall role understanding)

    We extract:
    - Core identifying information
    - Context that influences strategic decisions
    - NO lengthy descriptions that increase tokens without value

    Args:
        job_description: The JobDescription object

    Returns:
        Dictionary with essential job metadata

    Example Output:
        {
            "job_title": "Senior Software Engineer",
            "company_name": "Tech Innovators Inc",
            "job_level": "Senior",
            "summary": "Looking for experienced backend engineer..."
        }
    """
    job_metadata = {
        "job_title": job_description.job_title,
        "company_name": job_description.company_name,
    }

    # Include job level if available (critical for seniority matching)
    if hasattr(job_description, "job_level") and job_description.job_level:
        job_metadata["job_level"] = job_description.job_level

    # Include summary for overall context (keep it as-is, usually concise)
    if hasattr(job_description, "summary") and job_description.summary:
        job_metadata["summary"] = job_description.summary

    logger.debug("Extracted job metadata for gap analysis context")

    return job_metadata


# ==============================================================================
# MAIN FORMATTING FUNCTION
# ==============================================================================
# This is the primary entry point that orchestrates all extraction and
# formatting steps to produce the final context string for the Gap Analysis Agent.
# ==============================================================================


def format_gap_analysis_context(
    resume: Resume,
    job_description: JobDescription,
    format_type: FormatType = "toon",
) -> str:
    """
    Format and filter resume and job description data for the Gap Analysis Agent.

    This function orchestrates the complete data preparation pipeline:
    1. Extract all essential resume fields (skills, experience, education, certifications)
    2. Extract all essential job fields (requirements, ATS keywords, metadata)
    3. Combine into a structured dictionary
    4. Convert to TOON or Markdown format based on format_type
    5. Log token reduction metrics

    The Gap Analysis Agent receives a streamlined context containing:
    - All data needed for comprehensive gap identification
    - No redundant metadata or nested structures
    - Optimized format for LLM processing

    Args:
        resume: The Resume object containing candidate information
        job_description: The JobDescription object containing job requirements
        format_type: Output format ("toon" or "markdown"), defaults to "toon"

    Returns:
        Formatted string ready for Gap Analysis Agent consumption

    Token Optimization Strategy:
        - Remove all fields not directly used in gap comparison
        - Use TOON format to reduce JSON syntax overhead (~40-50% reduction)
        - Condense nested structures while preserving essential information
        - Expected reduction: 55-70% compared to full JSON serialization

    Example Usage:
        >>> resume = Resume(...)
        >>> job = JobDescription(...)
        >>> context = format_gap_analysis_context(resume, job, format_type="toon")
        >>> # Context now contains optimized data for gap analysis
    """
    logger.info("Starting Gap Analysis context formatting...")

    # ==============================================================================
    # STEP 1: Extract all resume data needed for gap analysis
    # ==============================================================================
    resume_data_for_gap_analysis = {}

    # Extract skills (primary comparison dimension)
    skills_data = _extract_skills_for_comparison(resume)
    resume_data_for_gap_analysis.update(skills_data)

    # Extract experience (implicit skills and domain matching)
    experience_data = _extract_experience_summaries_for_comparison(resume)
    resume_data_for_gap_analysis.update(experience_data)

    # Extract education and certifications (requirement matching)
    education_cert_data = _extract_education_and_certifications(resume)
    resume_data_for_gap_analysis.update(education_cert_data)

    # Extract additional context (languages, location, etc.)
    additional_context = _extract_additional_resume_context(resume)
    resume_data_for_gap_analysis.update(additional_context)

    logger.debug(f"Resume data sections extracted: {list(resume_data_for_gap_analysis.keys())}")

    # ==============================================================================
    # STEP 2: Extract all job description data needed for gap analysis
    # ==============================================================================
    job_data_for_gap_analysis = {}

    # Extract job metadata (title, company, level, summary)
    job_metadata = _extract_job_metadata_for_context(job_description)
    job_data_for_gap_analysis.update(job_metadata)

    # Extract requirements (primary comparison target)
    requirements_data = _extract_job_requirements_for_comparison(job_description)
    job_data_for_gap_analysis.update(requirements_data)

    # Extract ATS keywords (optimization targets)
    ats_keywords_data = _extract_ats_keywords_for_optimization(job_description)
    job_data_for_gap_analysis.update(ats_keywords_data)

    logger.debug(f"Job data sections extracted: {list(job_data_for_gap_analysis.keys())}")

    # ==============================================================================
    # STEP 3: Combine into structured input for Gap Analysis Agent
    # ==============================================================================
    combined_context = {
        "candidate_profile": resume_data_for_gap_analysis,
        "job_requirements": job_data_for_gap_analysis,
    }

    # ==============================================================================
    # STEP 4: Convert to specified format (TOON or Markdown)
    # ==============================================================================
    formatted_context = format_data(
        data=combined_context,
        format_type=format_type,
        description="Gap Analysis Context: Compare candidate profile against job requirements to identify matches, gaps, and optimization opportunities.",
    )

    # ==============================================================================
    # STEP 5: Calculate and log token reduction metrics
    # ==============================================================================
    # Calculate original size (if we had sent full JSON)
    original_resume_json = resume.model_dump_json()
    original_job_json = job_description.model_dump_json()
    original_combined = f"RESUME DATA:\n{original_resume_json}\n\nJOB DATA:\n{original_job_json}"
    original_tokens = estimate_tokens(original_combined)

    # Calculate optimized size
    optimized_tokens = estimate_tokens(formatted_context)

    # Calculate reduction percentage
    if original_tokens > 0:
        reduction_percentage = ((original_tokens - optimized_tokens) / original_tokens) * 100
    else:
        reduction_percentage = 0

    logger.info(
        f"Gap Analysis context formatting complete. "
        f"Original: ~{original_tokens} tokens, "
        f"Optimized: ~{optimized_tokens} tokens, "
        f"Reduction: {reduction_percentage:.1f}% ({format_type} format)"
    )

    return formatted_context
