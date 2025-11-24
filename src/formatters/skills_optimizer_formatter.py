"""
Skills Optimizer Agent Formatter
=================================

Formats and filters data specifically for the Skills Optimizer Agent to reduce
token usage while preserving all information needed for skills optimization.

The Skills Optimizer Agent optimizes the skills section by:
- Selecting skills that match job requirements
- Prioritizing skills based on importance and proficiency
- Organizing skills into relevant categories
- Ensuring ATS keyword coverage

This formatter:
1. Extracts skills from Resume (excludes work_experience, education, contact info, summary)
2. Extracts requirements and ats_keywords from Job (excludes full_text, summary, company details)
3. Extracts gap analysis results from Strategy (skill_gaps, matched_skills, recommendations)
4. Converts data to TOON or Markdown format based on format_type parameter

Expected token reduction: ~55-70% with TOON format compared to full JSON serialization.

Design Philosophy:
-----------------
- **Clear Separation of Concerns**: Each helper function handles one specific data extraction task
- **Descriptive Naming**: Function and variable names clearly indicate purpose and content
- **Reduced Cognitive Overload**: Complex models broken down into simple, focused transformations
- **Debuggability**: Extensive logging at each transformation step with token metrics
- **No Data Loss**: All fields required for skills optimization are preserved
"""

from typing import Any

from src.core.logger import get_logger
from src.data_models.job import JobDescription
from src.data_models.resume import Resume
from src.data_models.strategy import AlignmentStrategy
from src.formatters.base_formatter import FormatType, estimate_tokens, format_data

logger = get_logger(__name__)


# ==============================================================================
# RESUME DATA EXTRACTION FOR SKILLS OPTIMIZATION
# ==============================================================================


def _extract_skills_for_optimization(resume: Resume) -> dict[str, Any]:
    """
    Extract skills data for optimization and selection.

    The Skills Optimizer Agent needs:
    - Skill names for matching against job requirements
    - Proficiency levels to assess strength of skills
    - Categories for organizing skills section

    Args:
        resume: The Resume object containing skills

    Returns:
        Dictionary with skills data ready for optimization

    Example Output:
        {
            "skills": [
                {
                    "skill_name": "Python",
                    "proficiency_level": "Expert",
                    "category": "Programming Languages"
                },
                {
                    "skill_name": "AWS",
                    "proficiency_level": "Advanced",
                    "category": "Cloud Platforms"
                }
            ]
        }
    """
    skills_list = []

    for skill in resume.skills:
        skill_entry = {
            "skill_name": skill.skill_name,
            "proficiency_level": skill.proficiency_level
            if skill.proficiency_level
            else "Not Specified",
        }

        # Include category if available for better organization
        if skill.category:
            skill_entry["category"] = skill.category

        skills_list.append(skill_entry)

    logger.debug(f"Extracted {len(skills_list)} skills from resume for optimization")

    return {"skills": skills_list}


# ==============================================================================
# JOB DESCRIPTION DATA EXTRACTION FOR SKILLS OPTIMIZATION
# ==============================================================================


def _extract_job_targets_for_alignment(job_description: JobDescription) -> dict[str, Any]:
    """
    Extract job requirements and ATS keywords for skills targeting.

    The Skills Optimizer Agent needs:
    - Requirements to identify which skills are most important
    - ATS keywords to ensure keyword coverage
    - Importance levels to prioritize skill selection

    Args:
        job_description: The JobDescription object with parsed requirements

    Returns:
        Dictionary with requirements and keywords for targeting

    Example Output:
        {
            "requirements": [
                {"requirement": "5+ years Python", "importance": "must_have"},
                {"requirement": "AWS experience", "importance": "should_have"}
            ],
            "ats_keywords": ["Python", "AWS", "Docker", "Kubernetes"]
        }
    """
    requirements_list = []

    for req in job_description.requirements:
        requirement_entry = {
            "requirement": req.requirement,
            "importance": req.importance,
        }

        # Include category if available for better grouping
        if hasattr(req, "category") and req.category:
            requirement_entry["category"] = req.category

        requirements_list.append(requirement_entry)

    logger.debug(f"Extracted {len(requirements_list)} requirements for skills targeting")

    # Extract ATS keywords (simple list of strings)
    ats_keywords = job_description.ats_keywords if job_description.ats_keywords else []
    logger.debug(f"Extracted {len(ats_keywords)} ATS keywords for coverage")

    return {
        "requirements": requirements_list,
        "ats_keywords": ats_keywords,
    }


# ==============================================================================
# STRATEGY DATA EXTRACTION FOR SKILLS OPTIMIZATION
# ==============================================================================


def _extract_strategy_for_context(strategy: AlignmentStrategy) -> dict[str, Any]:
    """
    Extract gap analysis results to guide skills optimization.

    The Skills Optimizer Agent needs:
    - Skill gaps to know which skills to add/emphasize
    - Matched skills to know which skills to keep/highlight
    - Keywords to integrate and skills guidance

    Args:
        strategy: The AlignmentStrategy object with gap analysis results

    Returns:
        Dictionary with strategy guidance for optimization

    Example Output:
        {
            "identified_gaps": [
                {"missing_skill": "Kubernetes", "importance": "must_have", "suggestion": "Add if candidate has experience"}
            ],
            "identified_matches": [
                {"resume_skill": "Python", "job_requirement": "5+ years Python", "match_score": 95}
            ],
            "keywords_to_integrate": ["Python", "AWS", "Docker"],
            "skills_guidance": "Prioritize cloud skills and organize by technical vs soft skills"
        }
    """
    strategy_data = {}

    # Extract identified gaps - critical for knowing what skills to add
    if strategy.identified_gaps:
        gaps_list = []
        for gap in strategy.identified_gaps:
            gap_entry = {
                "missing_skill": gap.missing_skill,
                "importance": gap.importance,
            }
            if gap.suggestion:
                gap_entry["suggestion"] = gap.suggestion
            gaps_list.append(gap_entry)

        strategy_data["identified_gaps"] = gaps_list
        logger.debug(f"Extracted {len(gaps_list)} skill gaps from strategy")

    # Extract identified matches - to know what to keep and emphasize
    if strategy.identified_matches:
        matches_list = []
        for match in strategy.identified_matches:
            match_entry = {
                "resume_skill": match.resume_skill,
                "job_requirement": match.job_requirement,
                "match_score": match.match_score,
            }
            matches_list.append(match_entry)

        strategy_data["identified_matches"] = matches_list
        logger.debug(f"Extracted {len(matches_list)} matched skills from strategy")

    # Extract keywords to integrate
    if strategy.keywords_to_integrate:
        strategy_data["keywords_to_integrate"] = strategy.keywords_to_integrate
        logger.debug(f"Extracted {len(strategy.keywords_to_integrate)} keywords to integrate")

    # Extract skills guidance - strategic guidance specific to skills optimization
    if strategy.skills_guidance:
        strategy_data["skills_guidance"] = strategy.skills_guidance
        logger.debug("Extracted skills guidance from strategy")

    return strategy_data


# ==============================================================================
# MAIN FORMATTING FUNCTION
# ==============================================================================


def format_skills_optimizer_context(
    resume: Resume,
    job_description: JobDescription,
    strategy: AlignmentStrategy,
    format_type: FormatType = "toon",
) -> str:
    """
    Format and filter data for the Skills Optimizer Agent.

    This function orchestrates the complete data preparation pipeline:
    1. Extract skills from Resume
    2. Extract requirements and ats_keywords from Job
    3. Extract gap analysis results from Strategy
    4. Combine into a structured dictionary
    5. Convert to TOON or Markdown format based on format_type
    6. Log token reduction metrics

    The Skills Optimizer Agent receives a streamlined context containing:
    - All data needed for comprehensive skills optimization
    - No redundant metadata or unnecessary fields
    - Optimized format for LLM processing

    Args:
        resume: The Resume object containing skills
        job_description: The JobDescription object containing job requirements
        strategy: The AlignmentStrategy object with gap analysis results
        format_type: Output format ("toon" or "markdown"), defaults to "toon"

    Returns:
        Formatted string ready for Skills Optimizer Agent consumption

    Token Optimization Strategy:
        - Remove all fields not directly used in skills optimization
        - Use TOON format to reduce JSON syntax overhead (~40-50% reduction)
        - Expected reduction: 55-70% compared to full JSON serialization

    Example Usage:
        >>> resume = Resume(...)
        >>> job = JobDescription(...)
        >>> strategy = AlignmentStrategy(...)
        >>> context = format_skills_optimizer_context(resume, job, strategy, format_type="toon")
        >>> # Context now contains optimized data for skills optimization
    """
    logger.info("Starting Skills Optimizer context formatting...")

    # ==============================================================================
    # STEP 1: Extract skills data
    # ==============================================================================
    skills_data = _extract_skills_for_optimization(resume)

    # ==============================================================================
    # STEP 2: Extract job requirements and ATS keywords
    # ==============================================================================
    job_targets_data = _extract_job_targets_for_alignment(job_description)

    # ==============================================================================
    # STEP 3: Extract strategy gaps and recommendations
    # ==============================================================================
    strategy_data = _extract_strategy_for_context(strategy)

    # ==============================================================================
    # STEP 4: Combine into structured input for Skills Optimizer Agent
    # ==============================================================================
    combined_context = {
        "current_skills": skills_data,
        "job_targets": job_targets_data,
        "strategy": strategy_data,
    }

    logger.debug(f"Combined context sections: {list(combined_context.keys())}")

    # ==============================================================================
    # STEP 5: Convert to specified format (TOON or Markdown)
    # ==============================================================================
    formatted_context = format_data(
        data=combined_context,
        format_type=format_type,
        description="Skills Optimizer Context: Select and organize skills to match job requirements and ensure ATS keyword coverage.",
    )

    # ==============================================================================
    # STEP 6: Calculate and log token reduction metrics
    # ==============================================================================
    # Calculate original size (if we had sent full JSON)
    original_context = (
        f"CURRENT SKILLS:\n{resume.model_dump_json(include={'skills'})}\n\n"
        f"JOB TARGETS:\n{job_description.model_dump_json(include={'requirements', 'ats_keywords'})}\n\n"
        f"STRATEGY:\n{strategy.model_dump_json()}"
    )
    original_tokens = estimate_tokens(original_context)

    # Calculate optimized size
    optimized_tokens = estimate_tokens(formatted_context)

    # Calculate reduction percentage
    if original_tokens > 0:
        reduction_percentage = ((original_tokens - optimized_tokens) / original_tokens) * 100
    else:
        reduction_percentage = 0

    logger.info(
        f"Skills Optimizer context formatting complete. "
        f"Original: ~{original_tokens} tokens, "
        f"Optimized: ~{optimized_tokens} tokens, "
        f"Reduction: {reduction_percentage:.1f}% ({format_type} format)"
    )

    return formatted_context
