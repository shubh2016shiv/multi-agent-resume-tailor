"""
ATS Optimization Agent Formatter
=================================

PURPOSE:
--------
Formats and filters data specifically for the ATS Optimization Agent to reduce
token usage while preserving all information needed for resume assembly.

The ATS Optimization Agent's sole responsibility is to:
1. Assemble optimized components (summary, experience, skills) into a complete Resume
2. Generate markdown and JSON output formats
3. Validate ATS compatibility
4. Preserve contact info and education from original resume

CRITICAL REQUIREMENTS (What ATS Agent NEEDS):
----------------------------------------------
Based on assemble_resume_components() function analysis (src/agents/ats_optimization_agent.py:932-991):

1. From ProfessionalSummary:
   - ONLY the summary text string (not metadata like drafts, scores, critique)

2. From OptimizedExperienceSection:
   - ONLY the optimized_experiences list (Experience objects)
   - NOT optimization_notes, keywords_integrated, relevance_scores (metadata only)

3. From OptimizedSkillsSection:
   - ONLY the optimized_skills list (Skill objects)
   - NOT skill_categories, added_skills, removed_skills, optimization_notes (metadata only)

4. From Original Resume:
   - Contact info: full_name, email, phone_number, location, website_or_portfolio
   - education list
   - certifications list (optional)
   - languages list (optional)
   - NOT: work_experience (replaced by optimized), skills (replaced by optimized),
     professional_summary (replaced by optimized)

5. From JobDescription:
   - requirements list (for ATS keyword validation)
   - ats_keywords list (for keyword density calculation)
   - job_title (for context)
   - NOT: full_text, summary, company_name, location (not needed for assembly)

TOKEN REDUCTION STRATEGY:
-------------------------
- Current: ~40-60K tokens (5 full model_dump_json() calls with all metadata)
- Target: ~10-15K tokens (60-75% reduction)
- Method: Filter to only assembly-required fields + TOON/Markdown formatting

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
from src.data_models.resume import OptimizedSkillsSection, Resume
from src.formatters.base_formatter import FormatType, estimate_tokens, format_data

# Import agent-specific models
try:
    # from src.agents.ats_optimization_agent import OptimizedResume  # Unused import
    from src.agents.experience_optimizer_agent import OptimizedExperienceSection
    from src.agents.summary_writer_agent import ProfessionalSummary
except ImportError:
    # Fallback for direct script execution
    import sys
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    from src.agents.experience_optimizer_agent import OptimizedExperienceSection
    from src.agents.summary_writer_agent import ProfessionalSummary

logger = get_logger(__name__)


def _extract_summary_text_only(professional_summary_data: dict[str, Any]) -> str:
    """
    Extract ONLY the summary text from ProfessionalSummary model.

    CRITICAL: The ATS agent expects a string, not the full ProfessionalSummary object.

    The ProfessionalSummary model contains:
    - drafts: list[SummaryDraft] (multiple versions with scores/critique)
    - recommended_version: str (which draft name to use)
    - writing_notes: str (metadata about writing process)

    The ATS agent ONLY needs the actual summary text to assemble into the resume.

    Args:
        professional_summary_data: ProfessionalSummary dict from model_dump()

    Returns:
        The recommended summary text as a string

    Raises:
        ValueError: If cannot extract summary text
    """
    try:
        # Get the recommended version name
        recommended_version_name = professional_summary_data.get("recommended_version")

        if not recommended_version_name:
            logger.warning("No recommended_version found in ProfessionalSummary, using first draft")
            drafts = professional_summary_data.get("drafts", [])
            if drafts and len(drafts) > 0:
                summary_text = drafts[0].get("content", "")
                logger.info(f"Extracted summary text ({len(summary_text)} chars) from first draft")
                return summary_text
            else:
                raise ValueError("No drafts found in ProfessionalSummary")

        # Find the recommended draft
        drafts = professional_summary_data.get("drafts", [])
        recommended_draft = None

        for draft in drafts:
            if draft.get("version_name") == recommended_version_name:
                recommended_draft = draft
                break

        if not recommended_draft:
            logger.warning(
                f"Recommended version '{recommended_version_name}' not found, using first draft"
            )
            recommended_draft = drafts[0] if drafts else None

        if not recommended_draft:
            raise ValueError("No suitable draft found in ProfessionalSummary")

        summary_text = recommended_draft.get("content", "")

        logger.info(
            f"Extracted summary text ({len(summary_text)} chars) "
            f"from version: {recommended_draft.get('version_name')}"
        )

        return summary_text

    except Exception as e:
        logger.error(f"Failed to extract summary text: {e}", exc_info=True)
        raise ValueError(f"Cannot extract summary text from ProfessionalSummary: {e}") from e


def _extract_experience_list_only(
    optimized_experience_data: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Extract ONLY the optimized_experiences list from OptimizedExperienceSection model.

    CRITICAL: The ATS agent expects a list[Experience], not the full OptimizedExperienceSection.

    The OptimizedExperienceSection model contains:
    - optimized_experiences: list[Experience] (the actual work history entries)
    - optimization_notes: str (metadata about optimization decisions)
    - keywords_integrated: list[str] (tracking metadata)
    - relevance_scores: dict[str, float] (quality metrics)

    The ATS agent ONLY needs the optimized_experiences list for assembly.
    Metadata fields are for QA/logging only, not for resume assembly.

    Args:
        optimized_experience_data: OptimizedExperienceSection dict from model_dump()

    Returns:
        List of Experience dictionaries (work history entries)

    Raises:
        ValueError: If cannot extract experience list
    """
    try:
        optimized_experiences = optimized_experience_data.get("optimized_experiences", [])

        if not optimized_experiences:
            logger.warning("No optimized_experiences found in OptimizedExperienceSection")
            return []

        experience_count = len(optimized_experiences)
        logger.info(
            f"Extracted {experience_count} optimized experience entries (filtered out metadata)"
        )

        return optimized_experiences

    except Exception as e:
        logger.error(f"Failed to extract experience list: {e}", exc_info=True)
        raise ValueError(f"Cannot extract experiences from OptimizedExperienceSection: {e}") from e


def _extract_skills_list_only(optimized_skills_data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Extract ONLY the optimized_skills list from OptimizedSkillsSection model.

    CRITICAL: The ATS agent expects a list[Skill], not the full OptimizedSkillsSection.

    The OptimizedSkillsSection model contains:
    - optimized_skills: list[Skill] (the actual skill objects)
    - skill_categories: dict[str, list[str]] (grouping metadata)
    - added_skills: list[Skill] (tracking what was added)
    - removed_skills: list[str] (tracking what was removed)
    - optimization_notes: str (metadata about decisions)
    - ats_match_score: float (quality metric)

    The ATS agent ONLY needs the optimized_skills list for assembly.
    All other fields are metadata for tracking/QA, not for resume content.

    Args:
        optimized_skills_data: OptimizedSkillsSection dict from model_dump()

    Returns:
        List of Skill dictionaries

    Raises:
        ValueError: If cannot extract skills list
    """
    try:
        optimized_skills = optimized_skills_data.get("optimized_skills", [])

        if not optimized_skills:
            logger.warning("No optimized_skills found in OptimizedSkillsSection")
            return []

        skills_count = len(optimized_skills)
        logger.info(
            f"Extracted {skills_count} optimized skills (filtered out categories and metadata)"
        )

        return optimized_skills

    except Exception as e:
        logger.error(f"Failed to extract skills list: {e}", exc_info=True)
        raise ValueError(f"Cannot extract skills from OptimizedSkillsSection: {e}") from e


def _extract_contact_and_education_only(original_resume_data: dict[str, Any]) -> dict[str, Any]:
    """
    Extract ONLY contact info and education from original Resume.

    CRITICAL: The ATS agent already has optimized work_experience, skills, and summary
    from other agents. It ONLY needs contact info and education from the original resume.

    The Resume model contains:
    - full_name, email, phone_number, location, website_or_portfolio (NEEDED - contact info)
    - education: list[Education] (NEEDED - not modified by optimization agents)
    - certifications: list[str] (NEEDED - not modified by optimization agents)
    - languages: list[str] (NEEDED - not modified by optimization agents)
    - professional_summary: str (NOT NEEDED - replaced by ProfessionalSummary agent output)
    - work_experience: list[Experience] (NOT NEEDED - replaced by Experience Optimizer output)
    - skills: list[Skill] (NOT NEEDED - replaced by Skills Optimizer output)

    Args:
        original_resume_data: Resume dict from model_dump()

    Returns:
        Filtered dict with ONLY contact info, education, certifications, languages
    """
    try:
        # Extract ONLY the fields needed for assembly
        contact_and_education = {
            "full_name": original_resume_data.get("full_name", ""),
            "email": original_resume_data.get("email", ""),
            "phone_number": original_resume_data.get("phone_number"),
            "location": original_resume_data.get("location"),
            "website_or_portfolio": original_resume_data.get("website_or_portfolio"),
            "education": original_resume_data.get("education", []),
            "certifications": original_resume_data.get("certifications", []),
            "languages": original_resume_data.get("languages", []),
        }

        education_count = len(contact_and_education.get("education", []))
        cert_count = len(contact_and_education.get("certifications", []))

        logger.info(
            f"Extracted contact info + {education_count} education entries + "
            f"{cert_count} certifications (filtered out work_experience, skills, summary)"
        )

        return contact_and_education

    except Exception as e:
        logger.error(f"Failed to extract contact/education from resume: {e}", exc_info=True)
        raise ValueError(f"Cannot extract contact info and education: {e}") from e


def _extract_ats_validation_data_only(job_description_data: dict[str, Any]) -> dict[str, Any]:
    """
    Extract ONLY ATS validation data from JobDescription.

    CRITICAL: The ATS agent needs requirements and keywords for ATS validation,
    but doesn't need the full job description text or company details for assembly.

    The JobDescription model contains:
    - requirements: list[JobRequirement] (NEEDED - for keyword validation)
    - ats_keywords: list[str] (NEEDED - for keyword density calculation)
    - job_title: str (NEEDED - for context in validation)
    - full_text: str (NOT NEEDED - too large, not used in assembly)
    - summary: str (NOT NEEDED - not used in assembly)
    - company_name: str (NOT NEEDED - not used in assembly)
    - location: str (NOT NEEDED - not used in assembly)
    - job_level: str (NOT NEEDED - not used in assembly)

    Args:
        job_description_data: JobDescription dict from model_dump()

    Returns:
        Filtered dict with ONLY requirements, ats_keywords, job_title
    """
    try:
        ats_validation_data = {
            "job_title": job_description_data.get("job_title", ""),
            "requirements": job_description_data.get("requirements", []),
            "ats_keywords": job_description_data.get("ats_keywords", []),
        }

        requirements_count = len(ats_validation_data.get("requirements", []))
        keywords_count = len(ats_validation_data.get("ats_keywords", []))

        logger.info(
            f"Extracted ATS validation data: {requirements_count} requirements, "
            f"{keywords_count} keywords (filtered out full_text, summary, company details)"
        )

        return ats_validation_data

    except Exception as e:
        logger.error(f"Failed to extract ATS validation data: {e}", exc_info=True)
        raise ValueError(f"Cannot extract ATS data from JobDescription: {e}") from e


def format_ats_optimization_context(
    professional_summary: ProfessionalSummary,
    optimized_experience: OptimizedExperienceSection,
    optimized_skills: OptimizedSkillsSection,
    original_resume: Resume,
    job_description: JobDescription,
    format_type: FormatType = "toon",
) -> str:
    """
    Format context data for ATS Optimization Agent using specified format (TOON or Markdown).

    ASSEMBLY REQUIREMENTS:
    ----------------------
    The ATS agent assembles a complete Resume from:
    1. Summary text (string) - from ProfessionalSummary agent
    2. Experience list - from Experience Optimizer agent
    3. Skills list - from Skills Optimizer agent
    4. Contact info + Education - from original resume
    5. Job context - for ATS validation

    FILTERING STRATEGY:
    -------------------
    - Extract ONLY assembly-required data from each input
    - Remove all metadata fields (notes, scores, history, etc.)
    - Keep validation data (requirements, keywords) for ATS checks
    - Result: ~60-75% token reduction while preserving all necessary information

    Args:
        professional_summary: ProfessionalSummary object with multiple drafts
        optimized_experience: OptimizedExperienceSection object with optimized entries
        optimized_skills: OptimizedSkillsSection object with optimized skills
        original_resume: Original Resume object (for contact info and education)
        job_description: JobDescription object (for ATS validation)
        format_type: Format to use ("toon" for token reduction, "markdown" for readability)

    Returns:
        Formatted context string in the requested format ready for LLM input

    Raises:
        ValueError: If any critical data extraction fails
    """
    try:
        logger.info(
            f"Formatting ATS Optimization context: filtering and converting to {format_type.upper()}"
        )

        # STEP 1: Extract data from Pydantic models
        logger.debug("Step 1: Extracting data from Pydantic models...")
        professional_summary_data = professional_summary.model_dump()
        optimized_experience_data = optimized_experience.model_dump()
        optimized_skills_data = optimized_skills.model_dump()
        original_resume_data = original_resume.model_dump()
        job_description_data = job_description.model_dump()

        # STEP 2: Filter each input to only assembly-required fields
        logger.debug("Step 2: Filtering to assembly-required fields only...")

        summary_text = _extract_summary_text_only(professional_summary_data)
        experience_list = _extract_experience_list_only(optimized_experience_data)
        skills_list = _extract_skills_list_only(optimized_skills_data)
        contact_and_education = _extract_contact_and_education_only(original_resume_data)
        ats_validation_data = _extract_ats_validation_data_only(job_description_data)

        # STEP 3: Package filtered data for formatting
        logger.debug("Step 3: Packaging filtered data...")

        # Build structured data for ATS agent consumption
        filtered_data = {
            "professional_summary_text": summary_text,
            "optimized_experience_entries": experience_list,
            "optimized_skills_list": skills_list,
            "contact_information": contact_and_education,
            "ats_validation_context": ats_validation_data,
        }

        # STEP 4: Convert to requested format (TOON or Markdown)
        logger.debug(f"Step 4: Converting to {format_type.upper()} format...")

        formatted_content = format_data(
            filtered_data, format_type=format_type, description="ATS Optimization Assembly Data"
        )

        # STEP 5: Build final context with clear section labels
        if format_type == "markdown":
            context = f"""# ATS Optimization Agent - Resume Assembly Context

{formatted_content}

---

**Instructions for ATS Agent:**
1. Assemble complete Resume using professional_summary_text, optimized_experience_entries, optimized_skills_list
2. Use contact_information for contact details and education
3. Validate ATS compatibility using ats_validation_context
4. Generate markdown and JSON output formats
"""
        else:  # TOON format
            context = f"""ATS OPTIMIZATION ASSEMBLY DATA:

{formatted_content}

ASSEMBLY INSTRUCTIONS:
- Combine professional_summary_text + optimized_experience_entries + optimized_skills_list
- Use contact_information for resume header and education section
- Validate using ats_validation_context (requirements and keywords)
- Generate markdown and JSON outputs
"""

        # STEP 6: Log token reduction metrics
        logger.debug("Step 6: Calculating token reduction...")

        # Calculate original size (what we would have sent without filtering)
        original_context = (
            f"OPTIMIZED SUMMARY:\n{professional_summary.model_dump_json()}\n\n"
            f"OPTIMIZED EXPERIENCE:\n{optimized_experience.model_dump_json()}\n\n"
            f"OPTIMIZED SKILLS:\n{optimized_skills.model_dump_json()}\n\n"
            f"ORIGINAL RESUME:\n{original_resume.model_dump_json()}\n\n"
            f"TARGET JOB:\n{job_description.model_dump_json()}"
        )

        original_size = len(original_context)
        new_size = len(context)
        size_reduction_pct = (
            ((original_size - new_size) / original_size * 100) if original_size > 0 else 0
        )

        estimated_original_tokens = estimate_tokens(original_context)
        estimated_new_tokens = estimate_tokens(context)
        token_reduction_pct = (
            ((estimated_original_tokens - estimated_new_tokens) / estimated_original_tokens * 100)
            if estimated_original_tokens > 0
            else 0
        )

        logger.info(
            f"ATS Optimization context formatted ({format_type.upper()}): "
            f"{size_reduction_pct:.1f}% size reduction, "
            f"~{token_reduction_pct:.1f}% token reduction "
            f"({estimated_original_tokens} -> {estimated_new_tokens} tokens)"
        )

        return context

    except Exception as e:
        logger.error(f"Error formatting ATS Optimization context: {e}", exc_info=True)
        # Fallback to JSON format if conversion fails
        logger.warning("Falling back to JSON format for ATS Optimization context")
        return (
            f"OPTIMIZED SUMMARY:\n{professional_summary.model_dump_json()}\n\n"
            f"OPTIMIZED EXPERIENCE:\n{optimized_experience.model_dump_json()}\n\n"
            f"OPTIMIZED SKILLS:\n{optimized_skills.model_dump_json()}\n\n"
            f"ORIGINAL RESUME:\n{original_resume.model_dump_json()}\n\n"
            f"TARGET JOB:\n{job_description.model_dump_json()}"
        )
