"""
Professional Summary Agent Formatter
=====================================

PURPOSE:
--------
Formats and filters data specifically for the Summary Writer Agent to reduce
token usage while preserving all information needed for narrative summary generation.

The Summary Writer Agent's sole responsibility is to:
1. Craft compelling professional summaries (50-100 words)
2. Integrate ATS keywords naturally
3. Highlight candidate's strongest matches to the job
4. Create multiple narrative strategies and select the best one

CRITICAL REQUIREMENTS (What Summary Writer Agent NEEDS):
---------------------------------------------------------
Based on summary writer agent analysis and narrative principles:

1. From Resume:
   - professional_summary: Original summary text (to understand current state)
   - work_experience: CONDENSED version with job titles, companies, duration, top 2-3 achievements
   - skills: TOP 10-15 most relevant skills (prioritized by job matching)
   - education: Degree, field, institution (for context/credentials)
   - NOT: contact info, certifications, languages, full achievement lists

2. From JobDescription:
   - job_title: To tailor summary to specific role
   - summary: Brief role overview (context)
   - requirements: Only must-have and should-have (filter nice-to-have)
   - ats_keywords: All keywords for integration
   - job_level: For seniority matching
   - NOT: full_text (very long), company_name, location

3. From AlignmentStrategy:
   - professional_summary_guidance: CRITICAL - specific instructions
   - keywords_to_integrate: Must include these keywords
   - identified_matches: TOP 5-7 matches (highest match_score)
   - identified_gaps: Only must-have gaps (critical ones)
   - overall_fit_score: For context
   - summary_of_strategy: High-level direction
   - NOT: experience_guidance, skills_guidance, all detailed match/gap objects

TOKEN REDUCTION STRATEGY:
-------------------------
- Current: ~80-100K tokens (3 full model_dump_json() calls)
- Target: ~15-20K tokens (75-80% reduction)
- Method: Filter to only summary-writing fields + condense work_experience + TOON/Markdown formatting

DESIGN PRINCIPLES:
------------------
1. CLARITY: Variable names clearly indicate what data they contain
2. DEBUGGABILITY: Extensive logging at each filtering step
3. SAFETY: Validate that no critical fields are accidentally filtered out
4. DOCUMENTATION: Each function explains WHY certain fields are kept/removed
"""

from typing import Any

from src.core.logger import get_logger
from src.data_models.job import JobDescription
from src.data_models.resume import Resume
from src.data_models.strategy import AlignmentStrategy
from src.formatters.base_formatter import FormatType, estimate_tokens, format_data

logger = get_logger(__name__)


# ==============================================================================
# RESUME DATA EXTRACTION FOR SUMMARY WRITING
# ==============================================================================
# The Summary Writer needs to understand the candidate's background to craft
# a compelling narrative, but doesn't need all the details that other agents need.
# ==============================================================================


def _extract_resume_for_summary_writing(resume: Resume) -> dict[str, Any]:
    """
    Extract ONLY the resume fields needed for professional summary generation.

    The Summary Writer Agent needs to:
    - Understand the candidate's current professional summary
    - Know the candidate's career trajectory (job titles, companies, years)
    - See key achievements (but not exhaustive lists)
    - Know the candidate's top skills
    - Understand educational credentials

    We extract:
    - professional_summary: Original text (to build upon or replace)
    - work_experience: CONDENSED - job titles, companies, duration, top 2-3 achievements
    - skills: TOP 10-15 most relevant skills (by proficiency and job matching)
    - education: Degree, field, institution (for credentials)

    We EXCLUDE:
    - Contact info (full_name, email, phone, location, website) - not needed for summary
    - certifications, languages - not typically included in summary narrative
    - Full achievement lists - too verbose, only need highlights

    Args:
        resume: The Resume object containing candidate information

    Returns:
        Dictionary with condensed resume data for summary writing

    Example Output:
        {
            "professional_summary": "Current summary text...",
            "work_experience": [
                {
                    "title": "Senior Engineer",
                    "company": "Tech Corp",
                    "duration": "2020-01-15 - Present",
                    "years": 4.9,
                    "top_achievements": ["Led team of 5...", "Reduced costs by 40%"],
                    "skills_used": ["Python", "AWS"]
                }
            ],
            "top_skills": ["Python", "AWS", "Docker", ...],
            "education": [
                {"degree": "BS", "field": "Computer Science", "institution": "MIT"}
            ]
        }
    """
    resume_data_for_summary = {}

    # Extract original professional summary (to understand current state)
    if resume.professional_summary:
        resume_data_for_summary["professional_summary"] = resume.professional_summary
        logger.debug(f"Extracted professional summary ({len(resume.professional_summary)} chars)")

    # Extract CONDENSED work experience (job titles, companies, duration, top achievements only)
    if resume.work_experience:
        condensed_experience = []
        for exp in resume.work_experience:
            # Build duration string
            duration_str = f"{exp.start_date} - {exp.end_date if exp.end_date else 'Present'}"

            experience_entry = {
                "title": exp.job_title,
                "company": exp.company_name,
                "duration": duration_str,
                "years": exp.duration_in_years,  # Calculated property
            }

            # Include role description for context
            if exp.description:
                experience_entry["description"] = exp.description

            # Extract TOP 2-3 achievements only (not all bullets)
            # This gives enough context for summary writing without overwhelming token budget
            if exp.achievements:
                top_achievements = exp.achievements[:3]  # Top 3 only
                experience_entry["top_achievements"] = top_achievements

            # Include skills_used for skill matching
            if exp.skills_used:
                experience_entry["skills_used"] = exp.skills_used

            condensed_experience.append(experience_entry)

        resume_data_for_summary["work_experience"] = condensed_experience
        logger.debug(
            f"Extracted {len(condensed_experience)} work experience entries (condensed to top 3 achievements each)"
        )

    # Extract TOP 10-15 most relevant skills
    # Priority: skills matching job requirements, high proficiency, technical skills
    if resume.skills:
        # For summary writing, we don't need ALL skills - just the top ones
        # The full list will be in the skills section
        top_skills = []

        # Prioritize by proficiency level and take top skills
        # Sort by proficiency: Expert > Advanced > Intermediate > Beginner
        proficiency_order = {
            "Expert": 4,
            "Advanced": 3,
            "Intermediate": 2,
            "Beginner": 1,
            "Not Specified": 0,
        }

        sorted_skills = sorted(
            resume.skills,
            key=lambda s: proficiency_order.get(
                s.proficiency_level if s.proficiency_level else "Not Specified", 0
            ),
            reverse=True,
        )

        # Take top 15 skills for summary context
        for skill in sorted_skills[:15]:
            skill_entry = {
                "name": skill.skill_name,
                "proficiency": skill.proficiency_level
                if skill.proficiency_level
                else "Not Specified",
            }

            if skill.years_of_experience:
                skill_entry["years"] = skill.years_of_experience

            top_skills.append(skill_entry)

        resume_data_for_summary["top_skills"] = top_skills
        logger.debug(f"Extracted top {len(top_skills)} skills (from {len(resume.skills)} total)")

    # Extract education (degree, field, institution for credentials)
    if resume.education:
        education_list = []
        for edu in resume.education:
            education_entry = {
                "degree": edu.degree,
                "field": edu.field_of_study,
                "institution": edu.institution_name,
            }

            # Include graduation year for recency
            education_entry["graduation_year"] = edu.graduation_year

            education_list.append(education_entry)

        resume_data_for_summary["education"] = education_list
        logger.debug(f"Extracted {len(education_list)} education entries")

    # Calculate total years of experience (useful for summary narrative)
    if resume.work_experience:
        total_years = resume.total_years_of_experience
        resume_data_for_summary["total_years_experience"] = round(total_years, 1)
        logger.debug(f"Calculated total years of experience: {total_years}")

    return resume_data_for_summary


def _extract_job_for_summary_writing(job_description: JobDescription) -> dict[str, Any]:
    """
    Extract ONLY the job fields needed for professional summary generation.

    The Summary Writer Agent needs to:
    - Know the job title (to tailor summary to specific role)
    - Understand the role overview (brief context)
    - See must-have and should-have requirements (to emphasize relevant skills)
    - Know ATS keywords (to integrate naturally)
    - Understand seniority level (for tone/positioning)

    We extract:
    - job_title: To tailor summary ("Senior Software Engineer with...")
    - summary: Brief role overview (usually concise)
    - requirements: FILTERED to must-have and should-have only
    - ats_keywords: All keywords for natural integration
    - job_level: For seniority matching (Senior, Mid, Junior, etc.)

    We EXCLUDE:
    - full_text: Very long, already parsed into requirements
    - company_name: Not typically mentioned in professional summary
    - location: Not relevant to summary narrative

    Args:
        job_description: The JobDescription object

    Returns:
        Dictionary with filtered job data for summary writing

    Example Output:
        {
            "job_title": "Senior Software Engineer",
            "summary": "Looking for experienced backend engineer...",
            "requirements": [
                {"requirement": "5+ years Python", "importance": "must_have"},
                {"requirement": "AWS experience", "importance": "should_have"}
            ],
            "ats_keywords": ["Python", "AWS", "Docker"],
            "job_level": "Senior"
        }
    """
    job_data_for_summary = {}

    # Job title (critical for summary opening)
    job_data_for_summary["job_title"] = job_description.job_title

    # Summary (brief role overview, usually concise already)
    if job_description.summary:
        job_data_for_summary["summary"] = job_description.summary

    # Requirements - FILTER to must-have and should-have only (exclude nice-to-have)
    # The summary should focus on critical requirements, not exhaustive list
    if job_description.requirements:
        filtered_requirements = []

        for req in job_description.requirements:
            # Only include must-have and should-have (skip nice-to-have for summary)
            if req.importance in ["must_have", "should_have"]:
                req_entry = {
                    "requirement": req.requirement,
                    "importance": req.importance,
                }

                if hasattr(req, "category") and req.category:
                    req_entry["category"] = req.category

                filtered_requirements.append(req_entry)

        job_data_for_summary["requirements"] = filtered_requirements
        logger.debug(
            f"Filtered requirements: {len(filtered_requirements)} must-have/should-have "
            f"(from {len(job_description.requirements)} total)"
        )

    # ATS keywords (all of them - needed for natural integration)
    if job_description.ats_keywords:
        job_data_for_summary["ats_keywords"] = job_description.ats_keywords
        logger.debug(f"Extracted {len(job_description.ats_keywords)} ATS keywords")

    # Job level (for seniority matching and tone)
    if job_description.job_level:
        job_data_for_summary["job_level"] = job_description.job_level
        logger.debug(f"Extracted job level: {job_description.job_level}")

    return job_data_for_summary


def _extract_strategy_for_summary_writing(strategy: AlignmentStrategy) -> dict[str, Any]:
    """
    Extract ONLY the strategy fields needed for professional summary generation.

    The Summary Writer Agent needs to:
    - Follow specific guidance for summary writing
    - Integrate required keywords
    - Emphasize top skill matches
    - Address critical gaps (if any)
    - Understand overall alignment score

    We extract:
    - professional_summary_guidance: CRITICAL - specific instructions
    - keywords_to_integrate: Must include these keywords
    - identified_matches: TOP 5-7 matches (highest match_score) - not all
    - identified_gaps: Only must-have gaps - not all gaps
    - overall_fit_score: For context
    - summary_of_strategy: High-level direction

    We EXCLUDE:
    - experience_guidance: Not needed for summary (for experience optimizer)
    - skills_guidance: Not needed for summary (for skills optimizer)
    - All detailed matches/gaps beyond top ones (to reduce token usage)

    Args:
        strategy: The AlignmentStrategy object with strategic guidance

    Returns:
        Dictionary with filtered strategy data for summary writing

    Example Output:
        {
            "professional_summary_guidance": "Start with 'Senior Cloud Engineer with 8 years...'",
            "keywords_to_integrate": ["Python", "AWS", "Docker"],
            "top_matches": [
                {"resume_skill": "Python", "job_requirement": "5+ years Python", "match_score": 95},
                ...
            ],
            "critical_gaps": [
                {"missing_skill": "Kubernetes", "importance": "must_have", "suggestion": "..."}
            ],
            "overall_fit_score": 85.5,
            "summary_of_strategy": "Emphasize cloud and containerization experience"
        }
    """
    strategy_data_for_summary = {}

    # Professional summary guidance (CRITICAL - specific instructions)
    strategy_data_for_summary["professional_summary_guidance"] = (
        strategy.professional_summary_guidance
    )
    logger.debug(
        f"Extracted professional summary guidance ({len(strategy.professional_summary_guidance)} chars)"
    )

    # Keywords to integrate (all of them - critical for ATS)
    if strategy.keywords_to_integrate:
        strategy_data_for_summary["keywords_to_integrate"] = strategy.keywords_to_integrate
        logger.debug(f"Extracted {len(strategy.keywords_to_integrate)} keywords to integrate")

    # Identified matches - TOP 5-7 only (sorted by match_score)
    # The summary should emphasize the strongest matches, not list all of them
    if strategy.identified_matches:
        # Sort by match_score (highest first) and take top 7
        sorted_matches = sorted(
            strategy.identified_matches, key=lambda m: m.match_score, reverse=True
        )

        top_matches = []
        for match in sorted_matches[:7]:  # Top 7 matches only
            match_entry = {
                "resume_skill": match.resume_skill,
                "job_requirement": match.job_requirement,
                "match_score": match.match_score,
                "justification": match.justification,
            }
            top_matches.append(match_entry)

        strategy_data_for_summary["top_matches"] = top_matches
        logger.debug(
            f"Extracted top {len(top_matches)} matches (from {len(strategy.identified_matches)} total)"
        )

    # Identified gaps - ONLY must-have gaps (critical ones)
    # The summary should address critical gaps, not all gaps
    if strategy.identified_gaps:
        critical_gaps = []

        for gap in strategy.identified_gaps:
            # Only include must-have gaps (skip should-have and nice-to-have)
            if gap.importance == "must_have":
                gap_entry = {
                    "missing_skill": gap.missing_skill,
                    "importance": gap.importance,
                    "suggestion": gap.suggestion,
                }
                critical_gaps.append(gap_entry)

        if critical_gaps:
            strategy_data_for_summary["critical_gaps"] = critical_gaps
            logger.debug(f"Extracted {len(critical_gaps)} critical gaps (must-have only)")

    # Overall fit score (for context)
    strategy_data_for_summary["overall_fit_score"] = strategy.overall_fit_score

    # Summary of strategy (high-level direction)
    if strategy.summary_of_strategy:
        strategy_data_for_summary["summary_of_strategy"] = strategy.summary_of_strategy

    logger.debug(f"Extracted strategy data with fit score: {strategy.overall_fit_score}")

    return strategy_data_for_summary


# ==============================================================================
# MAIN FORMATTING FUNCTION
# ==============================================================================
# This is the primary entry point that orchestrates all extraction and
# formatting steps to produce the final context string for the Summary Writer Agent.
# ==============================================================================


def format_professional_summary_context(
    resume: Resume,
    job_description: JobDescription,
    strategy: AlignmentStrategy,
    format_type: FormatType = "toon",
) -> str:
    """
    Format and filter resume, job, and strategy data for the Summary Writer Agent.

    This function orchestrates the complete data preparation pipeline:
    1. Extract summary-relevant resume fields (condensed work_experience, top skills, education)
    2. Extract summary-relevant job fields (job_title, filtered requirements, ats_keywords)
    3. Extract summary-specific strategy guidance (professional_summary_guidance, top matches, critical gaps)
    4. Combine into a structured dictionary
    5. Convert to TOON or Markdown format based on format_type
    6. Log token reduction metrics

    The Summary Writer Agent receives a streamlined context containing:
    - All data needed for compelling narrative summary generation
    - No redundant metadata or verbose lists
    - Optimized format for LLM processing

    Args:
        resume: The Resume object containing candidate information
        job_description: The JobDescription object containing job requirements
        strategy: The AlignmentStrategy object with strategic guidance
        format_type: Output format ("toon" or "markdown"), defaults to "toon"

    Returns:
        Formatted string ready for Summary Writer Agent consumption

    Token Optimization Strategy:
        - Remove all fields not directly used in summary writing
        - Condense work_experience to top 2-3 achievements per role (~60% reduction)
        - Filter to top 10-15 skills instead of all skills
        - Filter requirements to must-have/should-have only
        - Filter matches to top 5-7 only
        - Filter gaps to must-have only
        - Use TOON format to reduce JSON syntax overhead (~40-50% reduction)
        - Expected reduction: 75-80% compared to full JSON serialization

    Example Usage:
        >>> resume = Resume(...)
        >>> job = JobDescription(...)
        >>> strategy = AlignmentStrategy(...)
        >>> context = format_professional_summary_context(resume, job, strategy, format_type="toon")
        >>> # Context now contains optimized data for summary writing
    """
    logger.info("Starting Professional Summary context formatting...")

    # ==============================================================================
    # STEP 1: Extract resume data for summary writing
    # ==============================================================================
    resume_data_for_summary = _extract_resume_for_summary_writing(resume)
    logger.debug(f"Resume data sections extracted: {list(resume_data_for_summary.keys())}")

    # ==============================================================================
    # STEP 2: Extract job description data for summary writing
    # ==============================================================================
    job_data_for_summary = _extract_job_for_summary_writing(job_description)
    logger.debug(f"Job data sections extracted: {list(job_data_for_summary.keys())}")

    # ==============================================================================
    # STEP 3: Extract strategy data for summary writing
    # ==============================================================================
    strategy_data_for_summary = _extract_strategy_for_summary_writing(strategy)
    logger.debug(f"Strategy data sections extracted: {list(strategy_data_for_summary.keys())}")

    # ==============================================================================
    # STEP 4: Combine into structured input for Summary Writer Agent
    # ==============================================================================
    combined_context = {
        "candidate_background": resume_data_for_summary,
        "target_role": job_data_for_summary,
        "strategic_guidance": strategy_data_for_summary,
    }

    # ==============================================================================
    # STEP 5: Convert to specified format (TOON or Markdown)
    # ==============================================================================
    formatted_context = format_data(
        data=combined_context,
        format_type=format_type,
        description="Professional Summary Writing Context: Craft compelling summary that highlights candidate's strongest matches, integrates ATS keywords naturally, and positions candidate for the target role.",
    )

    # ==============================================================================
    # STEP 6: Calculate and log token reduction metrics
    # ==============================================================================
    # Calculate original size (if we had sent full JSON)
    original_resume_json = resume.model_dump_json()
    original_job_json = job_description.model_dump_json()
    original_strategy_json = strategy.model_dump_json()
    original_combined = (
        f"RESUME DATA:\n{original_resume_json}\n\n"
        f"JOB DATA:\n{original_job_json}\n\n"
        f"STRATEGY:\n{original_strategy_json}"
    )
    original_tokens = estimate_tokens(original_combined)

    # Calculate optimized size
    optimized_tokens = estimate_tokens(formatted_context)

    # Calculate reduction percentage
    if original_tokens > 0:
        reduction_percentage = ((original_tokens - optimized_tokens) / original_tokens) * 100
    else:
        reduction_percentage = 0

    logger.info(
        f"Professional Summary context formatting complete. "
        f"Original: ~{original_tokens} tokens, "
        f"Optimized: ~{optimized_tokens} tokens, "
        f"Reduction: {reduction_percentage:.1f}% ({format_type} format)"
    )

    return formatted_context
