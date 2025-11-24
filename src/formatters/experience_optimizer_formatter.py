"""
Experience Optimizer Agent Formatter
====================================

Formats and filters data specifically for the Experience Optimizer Agent to reduce
token usage while preserving all information needed for experience bullet optimization.

The Experience Optimizer Agent optimizes work experience bullets by:
- Aligning experience descriptions with job requirements
- Integrating ATS keywords naturally
- Highlighting relevant achievements and skills
- Ensuring STAR format (Situation, Task, Action, Result)

This formatter:
1. Extracts work_experience from Resume (excludes contact info, education, skills, summary)
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
- **No Data Loss**: All fields required for experience optimization are preserved
"""

from typing import Any

from src.core.logger import get_logger
from src.data_models.job import JobDescription
from src.data_models.resume import Resume
from src.data_models.strategy import AlignmentStrategy
from src.formatters.base_formatter import FormatType, estimate_tokens, format_data

logger = get_logger(__name__)


# ==============================================================================
# RESUME DATA EXTRACTION FOR EXPERIENCE OPTIMIZATION
# ==============================================================================


def _extract_work_experience_for_optimization(resume: Resume) -> dict[str, Any]:
    """
    Extract work experience data for bullet optimization.

    The Experience Optimizer Agent needs:
    - Job titles and companies for context
    - Start/end dates for chronology
    - Current descriptions and achievements to optimize
    - Skills used to align with job requirements

    Args:
        resume: The Resume object containing work experience

    Returns:
        Dictionary with work experience data ready for optimization

    Example Output:
        {
            "work_experience": [
                {
                    "job_title": "Senior Software Engineer",
                    "company_name": "Tech Corp",
                    "start_date": "2020-01-15",
                    "end_date": "2023-12-31",
                    "description": "Led backend development team",
                    "achievements": ["Built microservices platform", "Reduced latency by 40%"],
                    "skills_used": ["Python", "AWS", "Docker"]
                }
            ]
        }
    """
    experience_list = []

    for exp in resume.work_experience:
        experience_entry = {
            "job_title": exp.job_title,
            "company_name": exp.company_name,
            "start_date": exp.start_date,
            "end_date": exp.end_date if exp.end_date else "Present",
        }

        # Include description (role summary) - needed for context
        if exp.description:
            experience_entry["description"] = exp.description

        # Include achievements - these will be optimized
        if exp.achievements:
            experience_entry["achievements"] = exp.achievements

        # Include skills_used - helps align with job requirements
        if exp.skills_used:
            experience_entry["skills_used"] = exp.skills_used

        experience_list.append(experience_entry)

    logger.debug(f"Extracted {len(experience_list)} work experience entries for optimization")

    return {"work_experience": experience_list}


# ==============================================================================
# JOB DESCRIPTION DATA EXTRACTION FOR EXPERIENCE OPTIMIZATION
# ==============================================================================


def _extract_job_requirements_for_alignment(job_description: JobDescription) -> dict[str, Any]:
    """
    Extract job requirements and ATS keywords for experience alignment.

    The Experience Optimizer Agent needs:
    - Requirements to align experience bullets with job needs
    - ATS keywords to integrate naturally into descriptions
    - Importance levels to prioritize optimization efforts

    Args:
        job_description: The JobDescription object with parsed requirements

    Returns:
        Dictionary with requirements and keywords for alignment

    Example Output:
        {
            "requirements": [
                {"requirement": "5+ years Python", "importance": "must_have"},
                {"requirement": "Team leadership", "importance": "should_have"}
            ],
            "ats_keywords": ["Python", "AWS", "Agile", "Docker"]
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

    logger.debug(f"Extracted {len(requirements_list)} requirements for experience alignment")

    # Extract ATS keywords (simple list of strings)
    ats_keywords = job_description.ats_keywords if job_description.ats_keywords else []
    logger.debug(f"Extracted {len(ats_keywords)} ATS keywords for integration")

    return {
        "requirements": requirements_list,
        "ats_keywords": ats_keywords,
    }


# ==============================================================================
# STRATEGY DATA EXTRACTION FOR EXPERIENCE OPTIMIZATION
# ==============================================================================


def _extract_strategy_gaps_for_context(strategy: AlignmentStrategy) -> dict[str, Any]:
    """
    Extract gap analysis results to guide experience optimization.

    The Experience Optimizer Agent needs:
    - Skill gaps to address in experience bullets
    - Matched skills to emphasize
    - Recommendations for strategic positioning

    Args:
        strategy: The AlignmentStrategy object with gap analysis results

    Returns:
        Dictionary with strategy guidance for optimization

    Example Output:
        {
            "identified_gaps": [
                {"missing_skill": "Kubernetes", "importance": "must_have", "suggestion": "Highlight container orchestration experience"}
            ],
            "identified_matches": [
                {"resume_skill": "Python", "job_requirement": "5+ years Python", "match_score": 95}
            ],
            "keywords_to_integrate": ["Python", "AWS", "Docker"],
            "experience_guidance": "Emphasize leadership experience and quantify impact with metrics"
        }
    """
    strategy_data = {}

    # Extract identified gaps - critical for knowing what to emphasize
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

    # Extract identified matches - to know what to emphasize
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

    # Extract experience guidance - strategic guidance specific to experience optimization
    if strategy.experience_guidance:
        strategy_data["experience_guidance"] = strategy.experience_guidance
        logger.debug("Extracted experience guidance from strategy")

    return strategy_data


# ==============================================================================
# MAIN FORMATTING FUNCTION
# ==============================================================================


def format_experience_optimizer_context(
    resume: Resume,
    job_description: JobDescription,
    strategy: AlignmentStrategy,
    format_type: FormatType = "toon",
) -> str:
    """
    Format and filter data for the Experience Optimizer Agent.

    This function orchestrates the complete data preparation pipeline:
    1. Extract work_experience from Resume
    2. Extract requirements and ats_keywords from Job
    3. Extract gap analysis results from Strategy
    4. Combine into a structured dictionary
    5. Convert to TOON or Markdown format based on format_type
    6. Log token reduction metrics

    The Experience Optimizer Agent receives a streamlined context containing:
    - All data needed for comprehensive experience optimization
    - No redundant metadata or unnecessary fields
    - Optimized format for LLM processing

    Args:
        resume: The Resume object containing work experience
        job_description: The JobDescription object containing job requirements
        strategy: The AlignmentStrategy object with gap analysis results
        format_type: Output format ("toon" or "markdown"), defaults to "toon"

    Returns:
        Formatted string ready for Experience Optimizer Agent consumption

    Token Optimization Strategy:
        - Remove all fields not directly used in experience optimization
        - Use TOON format to reduce JSON syntax overhead (~40-50% reduction)
        - Expected reduction: 55-70% compared to full JSON serialization

    Example Usage:
        >>> resume = Resume(...)
        >>> job = JobDescription(...)
        >>> strategy = AlignmentStrategy(...)
        >>> context = format_experience_optimizer_context(resume, job, strategy, format_type="toon")
        >>> # Context now contains optimized data for experience optimization
    """
    logger.info("Starting Experience Optimizer context formatting...")

    # ==============================================================================
    # STEP 1: Extract work experience data
    # ==============================================================================
    work_experience_data = _extract_work_experience_for_optimization(resume)

    # ==============================================================================
    # STEP 2: Extract job requirements and ATS keywords
    # ==============================================================================
    job_requirements_data = _extract_job_requirements_for_alignment(job_description)

    # ==============================================================================
    # STEP 3: Extract strategy gaps and recommendations
    # ==============================================================================
    strategy_data = _extract_strategy_gaps_for_context(strategy)

    # ==============================================================================
    # STEP 4: Combine into structured input for Experience Optimizer Agent
    # ==============================================================================
    combined_context = {
        "resume_work_experience": work_experience_data,
        "job_requirements": job_requirements_data,
        "strategy_gaps_and_matches": strategy_data,
    }

    logger.debug(f"Combined context sections: {list(combined_context.keys())}")

    # ==============================================================================
    # STEP 5: Convert to specified format (TOON or Markdown)
    # ==============================================================================
    formatted_context = format_data(
        data=combined_context,
        format_type=format_type,
        description="Experience Optimizer Context: Optimize work experience bullets to align with job requirements and integrate ATS keywords.",
    )

    # ==============================================================================
    # STEP 6: Calculate and log token reduction metrics
    # ==============================================================================
    # Calculate original size (if we had sent full JSON)
    original_context = (
        f"RESUME WORK EXPERIENCE:\n{resume.model_dump_json(include={'work_experience'})}\n\n"
        f"JOB REQUIREMENTS:\n{job_description.model_dump_json(include={'requirements', 'ats_keywords'})}\n\n"
        f"STRATEGY GAPS & MATCHES:\n{strategy.model_dump_json()}"
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
        f"Experience Optimizer context formatting complete. "
        f"Original: ~{original_tokens} tokens, "
        f"Optimized: ~{optimized_tokens} tokens, "
        f"Reduction: {reduction_percentage:.1f}% ({format_type} format)"
    )

    return formatted_context
