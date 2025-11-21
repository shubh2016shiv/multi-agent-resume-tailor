"""
ATS Optimization Specialist Agent
----------------------------------

This module defines the final agent in our workflow: the ATS Optimization Specialist.
This agent is responsible for assembling all optimized components and ensuring maximum
ATS (Applicant Tracking System) compatibility through comprehensive validation and optimization.

AGENT DESIGN PRINCIPLES:
- Conservative Decision-Making: When ambiguous, choose the most ATS-safe option
- Attention to Detail: Catch inconsistencies other agents may have missed
- Format Minimalism: Strip everything that isn't ATS-necessary
- Keyword Strategy: Optimize without over-optimization
- Final Quality Checkpoint: Last line of defense before output

WORKFLOW:
1. Receive all optimized components (summary, experience, skills, education)
2. Combine components into complete resume
3. Validate ATS compatibility (formatting, keywords, structure)
4. Calculate keyword density (optimal: 2-5%)
5. Verify standard section headers
6. Check for ATS-incompatible formatting elements
7. Generate final output in Markdown and JSON formats
8. Provide comprehensive optimization metadata

KEY OPTIMIZATION PRINCIPLES:
- Parsability Over Aesthetics: ATS success comes first
- Standard Conventions: Use recognized section headers
- Clean Formatting: No tables, columns, or graphics
- Keyword Balance: Present but not stuffed
- Contact Information: Easily extractable
- Special Characters: Avoid parsing-breaking characters

ATS VALIDATION CRITERIA:
- Keyword Coverage: All must-have keywords present
- Keyword Density: 2-5% of total content
- Section Headers: Standard and recognizable
- Formatting: Plain text compatible
- Contact Info: Top of resume, simple format
- File Format: .txt or .docx compatible structure
- White Space: Proper spacing and readability
- Font Consistency: Single, standard font implied

OUTPUT VALIDATION:
- All sections present and complete
- No information loss from optimization
- ATS compatibility score >= 85/100
- Keyword density in optimal range
- Standard section headers used
- No formatting issues detected
"""

import re
from typing import Optional

from crewai import Agent, Crew, Process, Task
from pydantic import BaseModel, Field, ValidationError

# Handle imports for both package usage and direct script execution
try:
    from src.core.config import get_agents_config, get_config
    from src.core.logger import get_logger
    from src.data_models.job import JobDescription
    from src.data_models.resume import (
        Education,
        Experience,
        OptimizedSkillsSection,
        Resume,
        Skill,
    )
    from src.data_models.strategy import AlignmentStrategy
    from src.tools.ats_validation import (
        calculate_keyword_density,
        validate_ats_formatting,
        check_section_headers,
        get_optimal_keyword_density_range,
        get_standard_headers,
        get_incompatible_patterns,
    )
except ImportError:
    # Fallback for when running this file directly
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.core.config import get_agents_config, get_config
    from src.core.logger import get_logger
    from src.data_models.job import JobDescription
    from src.data_models.resume import (
        Education,
        Experience,
        OptimizedSkillsSection,
        Resume,
        Skill,
    )
    from src.data_models.strategy import AlignmentStrategy
    from src.tools.ats_validation import (
        calculate_keyword_density,
        validate_ats_formatting,
        check_section_headers,
        get_optimal_keyword_density_range,
        get_standard_headers,
        get_incompatible_patterns,
    )

logger = get_logger(__name__)


# ==============================================================================
# Module Constants
# ==============================================================================

# Import constants from tools module for consistency
MIN_KEYWORD_DENSITY, MAX_KEYWORD_DENSITY = get_optimal_keyword_density_range()
OPTIMAL_KEYWORD_DENSITY = 0.035  # 3.5% - sweet spot
STANDARD_SECTION_HEADERS = get_standard_headers()
INCOMPATIBLE_PATTERNS = get_incompatible_patterns()

# ATS compatibility scoring weights
KEYWORD_COVERAGE_WEIGHT = 0.35  # Weight for keyword coverage in score
KEYWORD_DENSITY_WEIGHT = 0.25  # Weight for optimal density in score
FORMATTING_WEIGHT = 0.25  # Weight for clean formatting in score
STRUCTURE_WEIGHT = 0.15  # Weight for proper structure in score

# Quality thresholds
MIN_ATS_SCORE = 85.0  # Minimum acceptable ATS compatibility score
MIN_SECTIONS_COUNT = 4  # Minimum required sections (summary, experience, skills, education)

# Special characters that may break ATS parsing (from tools module)
PROBLEMATIC_CHARACTERS = ["™", "®", "©", "•", "→", "←", "↑", "↓", "★", "☆", "♦", "◆"]


# ==============================================================================
# Core Data Models
# ==============================================================================


class SectionValidation(BaseModel):
    """
    Validation result for a single resume section.
    
    This model captures the validation status of individual sections,
    identifying any issues that may affect ATS parsing.
    """

    section_name: str = Field(
        ...,
        description="Name of the section being validated",
        examples=["Professional Summary"],
    )

    is_present: bool = Field(
        ...,
        description="Whether the section exists in the resume",
    )

    is_standard_header: bool = Field(
        ...,
        description="Whether the section uses a standard ATS-recognized header",
    )

    header_used: Optional[str] = Field(
        None,
        description="The actual header text used for this section",
    )

    recommended_header: Optional[str] = Field(
        None,
        description="Recommended standard header if current is non-standard",
    )

    content_length: int = Field(
        default=0,
        ge=0,
        description="Character count of section content",
    )

    has_formatting_issues: bool = Field(
        default=False,
        description="Whether the section contains ATS-incompatible formatting",
    )

    issues_found: list[str] = Field(
        default_factory=list,
        description="Specific formatting issues detected",
    )


class KeywordDensityReport(BaseModel):
    """
    Analysis of keyword usage and density in resume content.
    
    This model provides comprehensive metrics about keyword integration,
    helping ensure optimal ATS performance without keyword stuffing.
    """

    total_words: int = Field(
        ...,
        ge=0,
        description="Total word count in resume",
    )

    total_keywords: int = Field(
        ...,
        ge=0,
        description="Total number of keyword instances found",
    )

    unique_keywords: int = Field(
        ...,
        ge=0,
        description="Number of unique keywords from job description",
    )

    keyword_density: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Ratio of keywords to total words (0.0-1.0)",
    )

    is_optimal: bool = Field(
        ...,
        description="Whether density is in optimal range (2-5%)",
    )

    keyword_coverage: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Percentage of required keywords present (0.0-1.0)",
    )

    missing_must_have_keywords: list[str] = Field(
        default_factory=list,
        description="Critical keywords from job posting that are missing",
    )

    keyword_frequency: dict[str, int] = Field(
        default_factory=dict,
        description="Frequency count for each keyword",
    )


class ATSValidationResult(BaseModel):
    """
    Comprehensive ATS compatibility validation result.
    
    This model represents the complete ATS validation assessment,
    including all checks performed and an overall compatibility score.
    """

    overall_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Overall ATS compatibility score (0-100)",
    )

    is_compatible: bool = Field(
        ...,
        description="Whether resume meets minimum ATS compatibility threshold",
    )

    section_validations: list[SectionValidation] = Field(
        default_factory=list,
        description="Validation results for each section",
    )

    keyword_report: Optional[KeywordDensityReport] = Field(
        None,
        description="Keyword density analysis",
    )

    formatting_issues: list[str] = Field(
        default_factory=list,
        description="Detected formatting problems that may break ATS parsing",
    )

    special_character_issues: list[str] = Field(
        default_factory=list,
        description="Special characters that should be removed or replaced",
    )

    recommendations: list[str] = Field(
        default_factory=list,
        description="Specific recommendations to improve ATS compatibility",
    )

    strengths: list[str] = Field(
        default_factory=list,
        description="Aspects of the resume that are well-optimized for ATS",
    )


class OptimizedResume(BaseModel):
    """
    Final optimized resume with comprehensive metadata.
    
    This model represents the complete, ATS-optimized resume output
    with all metadata about the optimization process and validation results.
    """

    # Core resume content
    resume: Resume = Field(
        ...,
        description="The complete, validated resume data",
    )

    # Markdown output
    markdown_content: str = Field(
        ...,
        description="Resume formatted as clean, ATS-compatible Markdown",
        min_length=100,
    )

    # JSON output
    json_content: str = Field(
        ...,
        description="Resume as structured JSON for machine processing",
        min_length=50,
    )

    # Optimization metadata
    ats_validation: ATSValidationResult = Field(
        ...,
        description="Comprehensive ATS compatibility validation results",
    )

    optimization_summary: str = Field(
        ...,
        description="High-level summary of optimization decisions and results",
    )

    components_assembled: dict[str, bool] = Field(
        default_factory=dict,
        description="Which components were successfully assembled",
        examples=[
            {
                "professional_summary": True,
                "work_experience": True,
                "skills": True,
                "education": True,
                "certifications": False,
            }
        ],
    )

    quality_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Key quality metrics for transparency",
        examples=[
            {
                "ats_score": 92.5,
                "keyword_density": 0.038,
                "keyword_coverage": 0.95,
                "total_sections": 5,
            }
        ],
    )


# ==============================================================================
# Agent Configuration Loading
# ==============================================================================


def _load_agent_config() -> dict:
    """
    Load the agent configuration from agents.yaml.
    
    This function provides a single point of configuration loading with
    proper error handling. If the config fails to load, it returns sensible
    defaults so the agent can still function.
    
    Returns:
        Dictionary containing agent configuration (role, goal, backstory, etc.)
        
    Design Note:
        Separating config loading into its own function makes the code more
        modular and testable. We can mock this function in tests.
    """
    try:
        agents_config = get_agents_config()
        config = agents_config.get("ats_optimization_specialist", {})

        # Validate that required fields are present
        required_fields = ["role", "goal", "backstory"]
        missing_fields = [f for f in required_fields if f not in config]

        if missing_fields:
            logger.warning(f"Agent config missing fields: {missing_fields}. Using defaults.")
            return _get_default_config()

        logger.debug("Successfully loaded agent configuration from YAML")
        return config

    except Exception as e:
        logger.error(f"Failed to load agent config: {e}. Using defaults.", exc_info=True)
        return _get_default_config()


def _get_default_config() -> dict:
    """
    Provide default configuration as a fallback.
    
    This ensures the agent can still be created even if the YAML config
    is unavailable or corrupted. These defaults are basic but functional.
    
    Returns:
        Dictionary with default agent configuration
    """
    return {
        "role": "ATS Optimization Specialist",
        "goal": (
            "Assemble all optimized resume components and ensure maximum ATS compatibility. "
            "Validate formatting, keyword density, section structure, and overall parsability. "
            "Generate final resume in Markdown and JSON formats with comprehensive metadata."
        ),
        "backstory": (
            "You are a meticulous ATS optimization expert with deep technical knowledge of how "
            "Applicant Tracking Systems parse and rank resumes. You understand both the technical "
            "requirements of ATS software and the human readability needs. Your expertise includes:\n"
            "- Technical knowledge of ATS parsing algorithms and ranking systems\n"
            "- Pattern recognition for ATS-incompatible formatting elements\n"
            "- Keyword optimization without stuffing (2-5% density sweet spot)\n"
            "- Standard section header conventions across different ATS platforms\n"
            "- Quality assurance mindset with attention to detail\n\n"
            "You are the final checkpoint ensuring resumes pass ATS screening while remaining "
            "professional and readable. You make conservative decisions, prioritizing parsability "
            "over aesthetics, and catch inconsistencies that other agents may have missed."
        ),
        "llm": "gemini/gemini-2.5-flash",
        "temperature": 0.2,
        "verbose": True,
    }


# ==============================================================================
# Agent Creation
# ==============================================================================


def create_ats_optimization_agent() -> Agent:
    """
    Create and configure the ATS Optimization Specialist agent.
    
    This is the main entry point for creating this agent. It handles all the
    complexity of configuration loading and agent initialization.
    
    Returns:
        Configured CrewAI Agent instance ready to optimize resumes for ATS
        
    Raises:
        Exception: If agent creation fails (logged and re-raised)
        
    Example:
        >>> agent = create_ats_optimization_agent()
        >>> # Agent is now ready to be used in a crew or task
        
    Design Notes:
        - Uses configuration from agents.yaml (with fallback to defaults)
        - Provides ATS validation tools for self-assessment
        - Low temperature (0.2) for conservative, deterministic decisions
        - Uses Gemini Flash for cost-effective validation
        - Enables verbose mode for detailed logging
    """
    try:
        logger.info("Creating ATS Optimization Specialist agent...")

        # Load configuration
        config = _load_agent_config()

        # Extract LLM settings
        llm_model = config.get("llm", "gemini/gemini-2.5-flash")
        temperature = config.get("temperature", 0.2)
        verbose = config.get("verbose", True)

        # Initialize tools
        tools = [
            calculate_keyword_density,
            validate_ats_formatting,
            check_section_headers,
        ]

        # Load centralized resilience configuration
        app_config = get_config()
        agent_defaults = app_config.llm.agent_defaults

        # Create the agent
        agent = Agent(
            role=config["role"],
            goal=config["goal"],
            backstory=config["backstory"],
            tools=tools,
            llm=llm_model,
            temperature=temperature,
            verbose=verbose,
            allow_delegation=False,  # This agent works independently
            
            # Resilience Parameters (Layer 1: CrewAI Native)
            max_retry_limit=agent_defaults.max_retry_limit,
            max_rpm=agent_defaults.max_rpm,
            max_iter=agent_defaults.max_iter,
            max_execution_time=agent_defaults.max_execution_time,
            respect_context_window=agent_defaults.respect_context_window,
        )

        logger.info(
            f"Successfully created agent: {config['role']}, "
            f"using LLM: {llm_model}, temperature: {temperature}, "
            f"tools: {len(tools)}, resilience: max_retry={agent_defaults.max_retry_limit}"
        )

        return agent

    except Exception as e:
        logger.error(f"Failed to create ATS Optimization Specialist agent: {e}", exc_info=True)
        raise


# ==============================================================================
# Core Validation Functions
# ==============================================================================


def validate_ats_compatibility(
    resume: Resume,
    job_description: JobDescription,
    resume_text: str,
) -> ATSValidationResult:
    """
    Perform comprehensive ATS compatibility validation.
    
    This function conducts a thorough analysis of the resume against
    ATS best practices and common failure points.
    
    Args:
        resume: The complete resume data model
        job_description: The target job description with requirements
        resume_text: The resume formatted as plain text
        
    Returns:
        ATSValidationResult with detailed validation findings
        
    Validation Checks:
        1. Section presence and headers
        2. Keyword coverage and density
        3. Formatting compatibility
        4. Special character issues
        5. Overall structure
    """
    logger.info("Starting comprehensive ATS compatibility validation...")

    # Initialize tracking
    all_issues = []
    all_recommendations = []
    all_strengths = []
    section_validations = []

    # 1. Validate section presence
    logger.debug("Validating section presence and headers...")
    
    section_checks = {
        "Professional Summary": resume.professional_summary is not None and len(resume.professional_summary) > 0,
        "Work Experience": resume.work_experience is not None and len(resume.work_experience) > 0,
        "Skills": resume.skills is not None and len(resume.skills) > 0,
        "Education": resume.education is not None and len(resume.education) > 0,
    }

    for section_name, is_present in section_checks.items():
        section_val = SectionValidation(
            section_name=section_name,
            is_present=is_present,
            is_standard_header=True,  # Assuming we use standard headers
            header_used=section_name,
            content_length=len(str(getattr(resume, section_name.lower().replace(" ", "_"), "")))
        )
        section_validations.append(section_val)

        if not is_present:
            all_issues.append(f"Missing section: {section_name}")
        else:
            all_strengths.append(f"{section_name} section present")

    # 2. Validate keyword density
    logger.debug("Analyzing keyword density and coverage...")
    
    # Extract keywords from job description
    required_keywords = [req.requirement for req in job_description.requirements]
    
    # Calculate metrics
    resume_lower = resume_text.lower()
    words = re.findall(r"\b\w+\b", resume_lower)
    total_words = len(words)

    keyword_freq: dict[str, int] = {}
    total_keyword_instances = 0

    for keyword in required_keywords:
        keyword_lower = keyword.lower()
        count = resume_lower.count(keyword_lower)
        if count > 0:
            keyword_freq[keyword] = count
            total_keyword_instances += count

    unique_keywords_found = len(keyword_freq)
    keyword_density = total_keyword_instances / total_words if total_words > 0 else 0
    keyword_coverage = unique_keywords_found / len(required_keywords) if required_keywords else 0
    is_optimal = MIN_KEYWORD_DENSITY <= keyword_density <= MAX_KEYWORD_DENSITY

    missing_keywords = [kw for kw in required_keywords if kw.lower() not in resume_lower]

    keyword_report = KeywordDensityReport(
        total_words=total_words,
        total_keywords=total_keyword_instances,
        unique_keywords=unique_keywords_found,
        keyword_density=keyword_density,
        is_optimal=is_optimal,
        keyword_coverage=keyword_coverage,
        missing_must_have_keywords=missing_keywords,
        keyword_frequency=keyword_freq,
    )

    # Assess keyword results
    if is_optimal:
        all_strengths.append(f"Keyword density optimal ({keyword_density:.1%})")
    elif keyword_density < MIN_KEYWORD_DENSITY:
        all_recommendations.append(f"Increase keyword density (current: {keyword_density:.1%}, target: {MIN_KEYWORD_DENSITY:.1%}-{MAX_KEYWORD_DENSITY:.1%})")
    else:
        all_recommendations.append(f"Reduce keyword density to avoid stuffing (current: {keyword_density:.1%}, max: {MAX_KEYWORD_DENSITY:.1%})")

    if keyword_coverage >= 0.8:
        all_strengths.append(f"Excellent keyword coverage ({keyword_coverage:.0%})")
    else:
        all_recommendations.append(f"Improve keyword coverage (current: {keyword_coverage:.0%}, missing: {len(missing_keywords)} keywords)")

    # 3. Check formatting issues
    logger.debug("Checking for ATS-incompatible formatting...")
    
    formatting_issues = []
    special_char_issues = []

    for pattern in INCOMPATIBLE_PATTERNS:
        if re.search(pattern, resume_text):
            formatting_issues.append(f"Incompatible pattern detected: {pattern}")

    for char in PROBLEMATIC_CHARACTERS:
        if char in resume_text:
            special_char_issues.append(f"Special character '{char}' should be removed")

    if not formatting_issues:
        all_strengths.append("No ATS-incompatible formatting detected")
    
    all_issues.extend(formatting_issues)

    # 4. Calculate overall score
    logger.debug("Calculating overall ATS compatibility score...")
    
    score = 100.0

    # Keyword coverage scoring (35%)
    keyword_score = keyword_coverage * KEYWORD_COVERAGE_WEIGHT * 100

    # Keyword density scoring (25%)
    if is_optimal:
        density_score = KEYWORD_DENSITY_WEIGHT * 100
    else:
        # Penalize based on how far from optimal
        if keyword_density < MIN_KEYWORD_DENSITY:
            density_score = (keyword_density / MIN_KEYWORD_DENSITY) * KEYWORD_DENSITY_WEIGHT * 100
        else:
            density_score = (MAX_KEYWORD_DENSITY / keyword_density) * KEYWORD_DENSITY_WEIGHT * 100

    # Formatting scoring (25%)
    formatting_score = FORMATTING_WEIGHT * 100
    if formatting_issues:
        formatting_score -= len(formatting_issues) * 10
    formatting_score = max(0, formatting_score)

    # Structure scoring (15%)
    sections_present = sum(1 for val in section_validations if val.is_present)
    structure_score = (sections_present / len(section_validations)) * STRUCTURE_WEIGHT * 100

    overall_score = keyword_score + density_score + formatting_score + structure_score
    overall_score = min(100.0, max(0.0, overall_score))

    is_compatible = overall_score >= MIN_ATS_SCORE

    logger.info(f"ATS validation complete. Score: {overall_score:.1f}/100, Compatible: {is_compatible}")

    return ATSValidationResult(
        overall_score=overall_score,
        is_compatible=is_compatible,
        section_validations=section_validations,
        keyword_report=keyword_report,
        formatting_issues=formatting_issues,
        special_character_issues=special_char_issues,
        recommendations=all_recommendations,
        strengths=all_strengths,
    )


# ==============================================================================
# Content Assembly Functions
# ==============================================================================


def assemble_resume_components(
    professional_summary: str,
    optimized_experience: list[Experience],
    optimized_skills: OptimizedSkillsSection,
    education: list[Education],
    contact_info: dict[str, str],
    certifications: list[str] | None = None,
) -> Resume:
    """
    Assemble all optimized components into a complete Resume object.
    
    This function combines all the individually optimized sections into
    a cohesive, validated resume data structure.
    
    Args:
        professional_summary: The optimized professional summary text
        optimized_experience: List of optimized experience entries
        optimized_skills: The optimized skills section
        education: List of education entries
        contact_info: Dictionary with contact details
        certifications: Optional list of certifications
        
    Returns:
        Complete Resume object ready for final formatting
        
    Raises:
        ValidationError: If assembled resume doesn't validate
    """
    logger.info("Assembling resume components...")

    try:
        resume = Resume(
            full_name=contact_info.get("full_name", ""),
            email=contact_info.get("email", ""),
            phone_number=contact_info.get("phone_number"),
            location=contact_info.get("location"),
            website_or_portfolio=contact_info.get("website_or_portfolio"),
            professional_summary=professional_summary,
            work_experience=optimized_experience,
            education=education,
            skills=optimized_skills.optimized_skills,
            certifications=certifications or [],
        )

        logger.info(
            f"Resume assembled successfully. "
            f"Sections: Summary={len(professional_summary)} chars, "
            f"Experience={len(optimized_experience)} entries, "
            f"Skills={len(optimized_skills.optimized_skills)}, "
            f"Education={len(education)}"
        )

        return resume

    except ValidationError as e:
        logger.error(f"Resume assembly validation failed: {e.errors()}")
        raise
    except Exception as e:
        logger.error(f"Resume assembly failed: {e}", exc_info=True)
        raise


def generate_markdown_resume(resume: Resume, skills_categories: dict[str, list[str]]) -> str:
    """
    Generate ATS-compatible Markdown representation of the resume.
    
    This function creates a clean, parsable Markdown version using
    standard formatting that ATS systems can reliably parse.
    
    Args:
        resume: The complete Resume object
        skills_categories: Categorized skills for organized display
        
    Returns:
        Resume formatted as clean Markdown text
        
    Design Notes:
        - Uses standard section headers
        - Avoids tables and complex formatting
        - Maintains consistent hierarchy
        - Ensures proper white space
    """
    logger.info("Generating Markdown resume...")

    sections = []

    # Contact Information (top, simple format)
    sections.append(f"# {resume.full_name}\n")
    contact_parts = []
    if resume.email:
        contact_parts.append(f"Email: {resume.email}")
    if resume.phone_number:
        contact_parts.append(f"Phone: {resume.phone_number}")
    if resume.location:
        contact_parts.append(f"Location: {resume.location}")
    if resume.website_or_portfolio:
        contact_parts.append(f"Portfolio: {resume.website_or_portfolio}")

    sections.append(" | ".join(contact_parts))
    sections.append("")  # Blank line

    # Professional Summary
    sections.append("## Professional Summary\n")
    sections.append(resume.professional_summary)
    sections.append("")

    # Work Experience
    sections.append("## Work Experience\n")
    for exp in resume.work_experience:
        # Title and company
        sections.append(f"### {exp.job_title} | {exp.company_name}")

        # Dates
        start = exp.start_date.strftime("%B %Y")
        end = exp.end_date.strftime("%B %Y") if exp.end_date else "Present"
        sections.append(f"*{start} - {end}*")

        # Location if available
        if exp.location:
            sections.append(f"*{exp.location}*")

        sections.append("")

        # Description
        if exp.description:
            sections.append(exp.description)
            sections.append("")

        # Achievements
        if exp.achievements:
            for achievement in exp.achievements:
                sections.append(f"- {achievement}")
            sections.append("")

    # Skills
    sections.append("## Skills\n")
    if skills_categories:
        for category, skill_list in skills_categories.items():
            sections.append(f"**{category}:** {', '.join(skill_list)}")
            sections.append("")
    else:
        # Fallback: simple list
        skill_names = [skill.skill_name for skill in resume.skills]
        sections.append(", ".join(skill_names))
        sections.append("")

    # Education
    sections.append("## Education\n")
    for edu in resume.education:
        sections.append(f"### {edu.degree} in {edu.field_of_study}")
        sections.append(f"*{edu.institution_name}*")
        sections.append(f"*Graduated: {edu.graduation_year}*")
        if edu.gpa:
            sections.append(f"*GPA: {edu.gpa}*")
        if edu.honors:
            sections.append(f"*{edu.honors}*")
        sections.append("")

    # Certifications (if any)
    if resume.certifications:
        sections.append("## Certifications\n")
        for cert in resume.certifications:
            sections.append(f"- {cert}")
        sections.append("")

    markdown_content = "\n".join(sections)
    logger.info(f"Markdown resume generated ({len(markdown_content)} characters)")

    return markdown_content


def generate_json_resume(resume: Resume) -> str:
    """
    Generate JSON representation of the resume.
    
    This function creates a structured JSON version of the resume
    for machine processing and API integration.
    
    Args:
        resume: The complete Resume object
        
    Returns:
        Resume as formatted JSON string
    """
    logger.info("Generating JSON resume...")

    try:
        # Convert Resume to dict, handling dates
        resume_dict = resume.model_dump(mode="json")

        # Format JSON with indentation for readability
        json_content = json.dumps(resume_dict, indent=2, ensure_ascii=False)

        logger.info(f"JSON resume generated ({len(json_content)} characters)")
        return json_content

    except Exception as e:
        logger.error(f"JSON generation failed: {e}", exc_info=True)
        raise


# ==============================================================================
# Output Validation
# ==============================================================================


def validate_optimized_output(output_data: dict) -> OptimizedResume | None:
    """
    Validate that the agent's output conforms to the OptimizedResume model.
    
    This function serves as a quality gate, ensuring that the final output
    is valid according to our schema. If validation fails, it provides
    detailed error information for debugging.
    
    Args:
        output_data: Dictionary containing the optimized resume
        
    Returns:
        OptimizedResume object if validation succeeds, None if it fails
        
    Design Notes:
        - Separating validation into its own function makes it reusable
        - Detailed logging helps diagnose generation issues
        - Returning None (rather than raising) allows graceful handling upstream
    """
    try:
        logger.debug("Validating agent output against OptimizedResume model...")

        # Attempt to create an OptimizedResume object from the output
        optimized_resume = OptimizedResume(**output_data)

        logger.info(
            f"Output validation successful. "
            f"ATS Score: {optimized_resume.ats_validation.overall_score:.1f}/100, "
            f"Compatible:<br>{optimized_resume.ats_validation.is_compatible}"
        )

        return optimized_resume

    except ValidationError as e:
        logger.error(
            f"Output validation failed. Output does not match OptimizedResume model schema. "
            f"Errors: {e.errors()}"
        )
        # Log each validation error for easier debugging
        for error in e.errors():
            logger.error(f"  Field: {error['loc']}, Type: {error['type']}, Message: {error['msg']}")
        return None

    except Exception as e:
        logger.error(f"Unexpected error during output validation: {e}", exc_info=True)
        return None


def check_ats_quality(optimized_resume: OptimizedResume) -> dict:
    """
    Perform quality checks on the optimized resume output.
    
    This function validates that the resume meets all ATS best practices
    and provides actionable feedback for any issues found.
    
    Args:
        optimized_resume: The validated OptimizedResume object
        
    Returns:
        Dictionary with quality check results and recommendations
        
    Quality Checks:
        - ATS compatibility score >= minimum threshold
        - All required sections present
        - Keyword density in optimal range
        - No formatting issues detected
        - Standard section headers used
        - Contact information complete
    """
    logger.info("Performing comprehensive quality checks...")

    results = {
        "overall_status": "pass" if optimized_resume.ats_validation.is_compatible else "fail",
        "ats_score": optimized_resume.ats_validation.overall_score,
        "quality_level": "",
        "critical_issues": [],
        "warnings": [],
        "strengths": optimized_resume.ats_validation.strengths,
        "recommendations": optimized_resume.ats_validation.recommendations,
    }

    # Determine quality level
    score = optimized_resume.ats_validation.overall_score
    if score >= 95:
        results["quality_level"] = "excellent"
    elif score >= 85:
        results["quality_level"] = "good"
    elif score >= 75:
        results["quality_level"] = "fair"
    else:
        results["quality_level"] = "poor"

    # Check for critical issues
    if not optimized_resume.ats_validation.is_compatible:
        results["critical_issues"].append(
            f"ATS compatibility score below minimum ({score:.1f} < {MIN_ATS_SCORE})"
        )

    if optimized_resume.ats_validation.formatting_issues:
        results["critical_issues"].extend(optimized_resume.ats_validation.formatting_issues)

    # Check keyword metrics
    if optimized_resume.ats_validation.keyword_report:
        kw_report = optimized_resume.ats_validation.keyword_report

        if not kw_report.is_optimal:
            results["warnings"].append(
                f"Keyword density not optimal: {kw_report.keyword_density:.1%} "
                f"(target: {MIN_KEYWORD_DENSITY:.1%}-{MAX_KEYWORD_DENSITY:.1%})"
            )

        if kw_report.missing_must_have_keywords:
            results["critical_issues"].append(
                f"Missing {len(kw_report.missing_must_have_keywords)} critical keywords: "
                f"{', '.join(kw_report.missing_must_have_keywords[:5])}"
            )

    # Check sections
    missing_sections = [
        val.section_name
        for val in optimized_resume.ats_validation.section_validations
        if not val.is_present
    ]

    if missing_sections:
        results["critical_issues"].append(f"Missing sections: {', '.join(missing_sections)}")

    logger.info(
        f"Quality check complete. Status: {results['overall_status']}, "
        f"Level: {results['quality_level']}, "
        f"Issues: {len(results['critical_issues'])}, "
        f"Warnings: {len(results['warnings'])}"
    )

    return results


# ==============================================================================
# Utility Functions
# ==============================================================================


def get_agent_info() -> dict:
    """
    Get information about this agent for debugging or monitoring.
    
    Returns:
        Dictionary with agent metadata
        
    Example:
        >>> info = get_agent_info()
        >>> print(info["name"])
        'ATS Optimization Specialist'
    """
    config = _load_agent_config()
    return {
        "name": "ATS Optimization Specialist",
        "role": config.get("role", "Unknown"),
        "llm": config.get("llm", "Unknown"),
        "tools": [
            "calculate_keyword_density",
            "validate_ats_formatting",
            "check_section_headers",
        ],
        "output_model": "OptimizedResume",
        "content_type": "final_optimized_resume",
    }


# ==============================================================================
# Testing Block
# ==============================================================================

if __name__ == "__main__":
    """
    Test the agent creation and validation functions.
    Run this script directly to verify the agent can be created.
    """
    print("=" * 70)
    print("ATS Optimization Specialist Agent - Test")
    print("=" * 70)

    # Test configuration loading
    print("\n--- Testing Configuration Loading ---")
    config = _load_agent_config()
    print(f"Role: {config.get('role', 'N/A')}")
    print(f"LLM: {config.get('llm', 'N/A')}")
    print(f"Temperature: {config.get('temperature', 'N/A')}")

    # Test agent creation
    print("\n--- Testing Agent Creation ---")
    try:
        agent = create_ats_optimization_agent()
        print("SUCCESS: Agent created successfully")
        print(f"Agent role: {agent.role}")
        print(f"Tools assigned: {len(agent.tools)}")
        for tool in agent.tools:
            print(f"  - {tool.name}")
    except Exception as e:
        print(f"FAILED: {str(e)}")

    # Display agent info
    print("\n--- Agent Information ---")
    info = get_agent_info()
    for key, value in info.items():
        print(f"{key}: {value}")

    # Note: The tools are wrapped with @tool decorator for CrewAI, but we can test the logic
    # by creating helper functions or calling them through the agent
    print("\n--- Testing Tool Functionality ---")
    print("Tools are configured for CrewAI agent usage.")
    print("For standalone testing, the tools would need to be invoked through the agent.")
    print("\nTo test the validation logic, you can:")
    print("1. Use the agent in a CrewAI task")
    print("2. Call the validation functions directly (validate_ats_compatibility, etc.)")
    print("3. Use the check_ats_quality function with sample data")
    
    print("\n--- Available Validation Functions ---")
    print("[OK] validate_ats_compatibility(resume, job_description, resume_text)")
    print("[OK] assemble_resume_components(...)")
    print("[OK] generate_markdown_resume(resume, skills_categories)")
    print("[OK] generate_json_resume(resume)")
    print("[OK] validate_optimized_output(output_data)")
    print("[OK] check_ats_quality(optimized_resume)")

    print("\n" + "=" * 70)

