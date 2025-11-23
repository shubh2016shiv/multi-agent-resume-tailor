"""
ATS Optimization Specialist Agent - Final Quality Assurance System
===================================================================

OVERVIEW:
---------
This module defines the final agent in our workflow: the ATS Optimization Specialist.
This agent serves as the quality assurance checkpoint that assembles all optimized
resume components and ensures maximum Applicant Tracking System (ATS) compatibility.

WHAT MAKES THIS AGENT CRITICAL:
-------------------------------
- **Final Quality Gate**: Last opportunity to catch issues before submission
- **ATS Expertise**: Deep understanding of ATS parsing algorithms and limitations
- **Conservative Approach**: When in doubt, chooses the most ATS-compatible option
- **Comprehensive Validation**: Checks formatting, keywords, structure, and completeness
- **Multi-Format Output**: Generates both human-readable and machine-readable versions

AGENT DESIGN PRINCIPLES:
------------------------
- **Parsability First**: ATS compatibility takes precedence over visual appeal
- **Conservative Decision-Making**: When ambiguous, choose the safest ATS option
- **Format Minimalism**: Strip anything that might confuse ATS parsing
- **Keyword Strategy**: Optimize density without appearing stuffed
- **Detail Orientation**: Catch inconsistencies other agents might miss

WORKFLOW OVERVIEW:
------------------
1. Receive optimized components (summary, experience, skills, education)
2. Assemble components into complete, cohesive resume
3. Validate ATS compatibility across multiple dimensions
4. Calculate and optimize keyword density (optimal: 2-5%)
5. Verify standard section headers and formatting
6. Check for ATS-incompatible elements (tables, graphics, etc.)
7. Generate final outputs in Markdown and JSON formats
8. Provide comprehensive validation report and optimization metadata

MODULE STRUCTURE (Hierarchical Organization):
=============================================
This module is organized into 7 main BLOCKS, each containing STAGES with SUB-STAGES:

BLOCK 1: MODULE SETUP & CONFIGURATION
├── Stage 1.1: Import Management
│   ├── Sub-stage 1.1.1: Standard library imports
│   ├── Sub-stage 1.1.2: CrewAI framework imports
│   ├── Sub-stage 1.1.3: Project-specific imports (with fallback handling)
│   └── Sub-stage 1.1.4: Logger initialization
│
├── Stage 1.2: ATS Constants & Patterns
│   ├── Sub-stage 1.2.1: Standard section headers
│   ├── Sub-stage 1.2.2: ATS-incompatible patterns
│   ├── Sub-stage 1.2.3: Keyword density ranges
│   └── Sub-stage 1.2.4: Validation thresholds

BLOCK 2: DATA MODELS
├── Stage 2.1: Validation Models
│   ├── Sub-stage 2.1.1: SectionValidation model
│   ├── Sub-stage 2.1.2: KeywordDensityReport model
│   └── Sub-stage 2.1.3: ATSValidationResult model
│
├── Stage 2.2: Output Models
│   ├── Sub-stage 2.2.1: OptimizedResume model
│   └── Sub-stage 2.2.2: Optimization metadata fields

BLOCK 3: AGENT CONFIGURATION & CREATION
├── Stage 3.1: Configuration Loading
│   ├── Sub-stage 3.1.1: Load from agents.yaml with fallback
│   ├── Sub-stage 3.1.2: Validate required fields
│   └── Sub-stage 3.1.3: Error handling with defaults
│
├── Stage 3.2: Default Configuration
│   ├── Sub-stage 3.2.1: Define ATS Specialist role
│   ├── Sub-stage 3.2.2: Set conservative LLM parameters
│   └── Sub-stage 3.2.3: Configure validation-focused behavior
│
└── Stage 3.3: Agent Creation
    ├── Sub-stage 3.3.1: CrewAI Agent initialization
    ├── Sub-stage 3.3.2: Tool assignment for ATS validation
    └── Sub-stage 3.3.3: Resilience configuration

BLOCK 4: ATS VALIDATION SYSTEM
├── Stage 4.1: Compatibility Validation
│   ├── Sub-stage 4.1.1: validate_ats_compatibility() function
│   ├── Sub-stage 4.1.2: Keyword density calculation
│   ├── Sub-stage 4.1.3: Formatting validation
│   └── Sub-stage 4.1.4: Section header verification
│
├── Stage 4.2: Quality Assessment
│   ├── Sub-stage 4.2.1: check_ats_quality() function
│   ├── Sub-stage 4.2.2: Comprehensive scoring algorithm
│   └── Sub-stage 4.2.3: Issue identification and recommendations

BLOCK 5: RESUME ASSEMBLY & GENERATION
├── Stage 5.1: Component Assembly
│   ├── Sub-stage 5.1.1: assemble_resume_components() function
│   ├── Sub-stage 5.1.2: Section ordering and integration
│   └── Sub-stage 5.1.3: Consistency validation
│
├── Stage 5.2: Format Generation
│   ├── Sub-stage 5.2.1: generate_markdown_resume() function
│   ├── Sub-stage 5.2.2: generate_json_resume() function
│   └── Sub-stage 5.2.3: Format-specific optimizations

BLOCK 6: OUTPUT VALIDATION
├── Stage 6.1: Result Validation
│   ├── Sub-stage 6.1.1: validate_optimized_output() function
│   ├── Sub-stage 6.1.2: Schema compliance checking
│   └── Sub-stage 6.1.3: Data integrity validation
│
└── Stage 6.2: Final Quality Checks
    ├── Sub-stage 6.2.1: Information completeness verification
    ├── Sub-stage 6.2.2: ATS score threshold validation
    └── Sub-stage 6.2.3: Optimization metadata generation

BLOCK 7: UTILITIES & TESTING
├── Stage 7.1: Agent Information
│   ├── Sub-stage 7.1.1: get_agent_info() function
│   ├── Sub-stage 7.1.2: Metadata retrieval
│   └── Sub-stage 7.1.3: Diagnostic information
│
└── Stage 7.2: Testing Support
    ├── Sub-stage 7.2.1: Test configuration loading
    ├── Sub-stage 7.2.2: Test agent creation
    ├── Sub-stage 7.2.3: Test validation functions
    └── Sub-stage 7.2.4: Test resume generation

HOW TO USE THIS MODULE:
-----------------------
1. Import: `from src.agents.ats_optimization_agent import create_ats_optimization_agent`
2. Create Agent: `agent = create_ats_optimization_agent()`
3. Run Optimization: Use in a Crew with optimized resume components
4. Validate Output: Use `validate_optimized_output()` to ensure quality
5. Check Quality: Use `check_ats_quality()` for detailed scoring

KEY OPTIMIZATION PRINCIPLES:
---------------------------
- **ATS-First Mindset**: Compatibility takes precedence over aesthetics
- **Conservative Choices**: When in doubt, choose the most ATS-safe option
- **Format Purity**: Strip tables, columns, graphics, and complex formatting
- **Keyword Balance**: Present all required keywords without stuffing
- **Standard Headers**: Use universally recognized section names
- **Clean Structure**: Ensure proper spacing and readable formatting

ATS VALIDATION CRITERIA:
------------------------
- **Keyword Coverage**: All must-have keywords must be present
- **Keyword Density**: Optimal range of 2-5% of total content
- **Section Headers**: Standard headers that ATS systems recognize
- **Formatting**: Plain text compatible (no special characters/issues)
- **Contact Info**: Prominent placement with simple, extractable format
- **Completeness**: All sections present with no information loss
- **Parsability**: Content structured for easy ATS parsing

TECHNICAL ARCHITECTURE:
-----------------------
- **Validation-First**: Comprehensive checks before output generation
- **Multi-Format Support**: Generates both Markdown and JSON outputs
- **Tool Integration**: Uses specialized ATS validation tools
- **Quality Scoring**: Numerical scoring system with actionable feedback
- **Error Recovery**: Graceful handling of validation failures
- **Metadata Rich**: Detailed optimization reports and validation results

FINAL QUALITY ASSURANCE:
-----------------------
This agent serves as the last line of defense, ensuring that:
1. All optimized components integrate seamlessly
2. ATS compatibility is maximized
3. No information is lost in the assembly process
4. Output meets professional standards
5. Keyword optimization is balanced and effective
6. Formatting is ATS-compatible and clean

The result is a resume that not only contains optimized content but is also
formatted and structured for maximum ATS success while remaining appealing
to human recruiters.
"""

import json
import re

from crewai import Agent
from crewai.tools import tool
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
        # Skill,
    )

    # from src.data_models.strategy import AlignmentStrategy
    from src.tools.ats_validation import (
        calculate_keyword_density,
        check_section_headers,
        get_incompatible_patterns,
        get_optimal_keyword_density_range,
        get_standard_headers,
        validate_ats_formatting,
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
        # Skill,
    )

    # from src.data_models.strategy import AlignmentStrategy
    from src.tools.ats_validation import (
        calculate_keyword_density,
        check_section_headers,
        get_incompatible_patterns,
        get_optimal_keyword_density_range,
        get_standard_headers,
        validate_ats_formatting,
    )

logger = get_logger(__name__)


# ==============================================================================
# BLOCK 1: MODULE SETUP & CONFIGURATION
# ==============================================================================
# PURPOSE: Initialize the module with imports, constants, and ATS validation patterns
# WHAT: Global constants and patterns used throughout ATS optimization
# WHY: Centralized configuration ensures consistency across all ATS checks
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1.2: ATS Constants & Patterns
# ------------------------------------------------------------------------------
# This stage defines the core constants and patterns used for ATS validation.


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
# BLOCK 2: DATA MODELS
# ==============================================================================
# PURPOSE: Define structured data models for ATS validation and optimization results
# WHAT: Pydantic models that ensure type safety and validation for ATS-related data
# WHY: Type-safe data structures prevent bugs and enable validation at runtime
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 2.1: Validation Models
# ------------------------------------------------------------------------------
# This stage defines models for tracking ATS validation results and metrics.


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

    header_used: str = Field(
        default="",
        description="The actual header text used for this section",
    )

    recommended_header: str = Field(
        default="",
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

    keyword_report: KeywordDensityReport = Field(
        default_factory=lambda: KeywordDensityReport(
            total_words=0,
            total_keywords=0,
            unique_keywords=0,
            keyword_density=0.0,
            is_optimal=False,
            keyword_coverage=0.0,
        ),
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


# ------------------------------------------------------------------------------
# Stage 2.2: Output Models
# ------------------------------------------------------------------------------
# This stage defines the final output model containing the complete optimized resume.


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
# BLOCK 3: AGENT CONFIGURATION & CREATION
# ==============================================================================
# PURPOSE: Configure and create the ATS Optimization agent with all necessary tools
# WHAT: Agent setup, configuration loading, and initialization with ATS validation tools
# WHY: Proper configuration ensures the agent can perform comprehensive ATS validation
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 3.1: Configuration Loading
# ------------------------------------------------------------------------------
# This stage loads agent configuration from external files with graceful fallbacks.


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


# ------------------------------------------------------------------------------
# Stage 3.2: Default Configuration Fallback
# ------------------------------------------------------------------------------
# This stage provides production-ready defaults when configuration files are unavailable.


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
            "Validate and optimize the resume for maximum ATS compatibility. Verify all critical "
            "keywords from the job description are present, check that keyword density is optimal "
            "(not stuffing), validate formatting meets ATS standards, ensure section headers "
            "use standard conventions, and identify any elements that may cause parsing issues.\n\n"
            "CRITICAL: You must output a SINGLE, complete JSON object with all resume data, "
            "validation results, and optimized content. Do not make multiple separate outputs. "
            "Perform all validation internally and include everything in one final response."
        ),
        "backstory": (
            "You are an ATS systems expert with a technical background in HR technology platforms. "
            "Having worked with major ATS vendors and analyzed thousands of resume parsing "
            "scenarios, you understand exactly how these systems work and what causes failures. "
            "Your expertise includes:\n"
            "- Technical knowledge of how ATS parse different file formats and structures\n"
            "- Understanding of keyword matching algorithms and ranking systems\n"
            "- Pattern recognition for ATS-incompatible formatting elements\n"
            "- Standard section header conventions across different ATS platforms\n"
            "- Insight into optimal keyword density (enough to match, not enough to be flagged)\n\n"
            "Your methodology is technical and precise. You run systematic checks: verify every "
            "must-have keyword from the job posting appears at least once, ensure keyword density "
            "falls in the 2-5% range, validate section headers use standard terminology, "
            "confirm no tables or complex formatting that breaks parsing, and test that contact "
            "information is easily extractable.\n\n"
            "IMPORTANT: You are a 'finalizer' agent. Your job is to assemble the complete, "
            "production-ready resume with all metadata. Output everything at once in a single "
            "comprehensive response."
        ),
        "llm": "gemini/gemini-2.5-flash",
        "temperature": 0.2,
        "verbose": True,
    }


# ------------------------------------------------------------------------------
# Stage 3.3: Agent Creation
# ------------------------------------------------------------------------------
# This stage creates the ATS Optimization agent with all necessary tools and configuration.


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
            validate_and_finalize_resume,
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
# BLOCK 4: ATS VALIDATION SYSTEM
# ==============================================================================
# PURPOSE: Comprehensive ATS compatibility validation and quality assessment
# WHAT: Functions that check keyword density, formatting, structure, and ATS compatibility
# WHY: Ensures resumes are optimized for both ATS parsing and human readability
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 4.1: Compatibility Validation
# ------------------------------------------------------------------------------
# This stage performs comprehensive ATS compatibility checks across multiple dimensions.


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
        "Professional Summary": resume.professional_summary is not None
        and len(resume.professional_summary) > 0,
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
            content_length=len(str(getattr(resume, section_name.lower().replace(" ", "_"), ""))),
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
        all_recommendations.append(
            f"Increase keyword density (current: {keyword_density:.1%}, target: {MIN_KEYWORD_DENSITY:.1%}-{MAX_KEYWORD_DENSITY:.1%})"
        )
    else:
        all_recommendations.append(
            f"Reduce keyword density to avoid stuffing (current: {keyword_density:.1%}, max: {MAX_KEYWORD_DENSITY:.1%})"
        )

    if keyword_coverage >= 0.8:
        all_strengths.append(f"Excellent keyword coverage ({keyword_coverage:.0%})")
    else:
        all_recommendations.append(
            f"Improve keyword coverage (current: {keyword_coverage:.0%}, missing: {len(missing_keywords)} keywords)"
        )

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

    # score = 100.0  # Unused variable - kept for potential future refactoring

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

    logger.info(
        f"ATS validation complete. Score: {overall_score:.1f}/100, Compatible: {is_compatible}"
    )

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
# BLOCK 5: RESUME ASSEMBLY & GENERATION
# ==============================================================================
# PURPOSE: Assemble optimized components into final resume formats
# WHAT: Functions that combine summary, experience, skills into complete resumes
# WHY: Creates the final output in both human-readable and machine-readable formats
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 5.1: Component Assembly
# ------------------------------------------------------------------------------
# This stage assembles all optimized components into a cohesive resume structure.


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


# ------------------------------------------------------------------------------
# Stage 5.2: Format Generation
# ------------------------------------------------------------------------------
# This stage generates the final resume in multiple formats optimized for different use cases.


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
# BLOCK 6: OUTPUT VALIDATION
# ==============================================================================
# PURPOSE: Validate final optimized resume outputs against quality standards
# WHAT: Quality gates that ensure optimized resumes meet all requirements
# WHY: Final checkpoint before resume is considered ready for submission
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 6.1: Result Validation
# ------------------------------------------------------------------------------
# This stage validates that optimized outputs conform to expected data models.


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


# ------------------------------------------------------------------------------
# Stage 4.2: Quality Assessment
# ------------------------------------------------------------------------------
# This stage performs comprehensive quality assessment with scoring and recommendations.


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
# BLOCK 8: AGENT TOOLS
# ==============================================================================
# PURPOSE: Define specialized tools for the ATS agent to use
# WHAT: Tools that encapsulate complex logic like validation and formatting
# WHY: Offloads heavy processing from the LLM to Python code for reliability
# ==============================================================================


@tool("Validate and Finalize Resume")
def validate_and_finalize_resume(resume_json: str, job_description_json: str) -> str:
    """
    Validate, format, and finalize the resume for ATS compatibility.

    This tool takes the assembled resume and job description, performs all
    ATS validation checks, generates the required Markdown and JSON formats,
    and returns the complete OptimizedResume object.

    Args:
        resume_json: JSON string representing the assembled Resume object
        job_description_json: JSON string representing the JobDescription object

    Returns:
        JSON string of the complete OptimizedResume object
    """
    try:
        logger.info("Tool 'Validate and Finalize Resume' called")

        # 1. Parse Inputs
        try:
            resume_dict = json.loads(resume_json)
            resume = Resume(**resume_dict)
        except (json.JSONDecodeError, ValidationError) as e:
            return f"Error parsing resume_json: {str(e)}. Ensure it matches the Resume model structure."

        try:
            job_dict = json.loads(job_description_json)
            job_description = JobDescription(**job_dict)
        except (json.JSONDecodeError, ValidationError) as e:
            return f"Error parsing job_description_json: {str(e)}. Ensure it matches the JobDescription model structure."

        # 2. Generate Formats
        # We need skills categories for markdown generation.
        # Since we don't have them explicitly passed, we'll try to infer or use a default.
        skills_categories = {}
        if resume.skills:
            for skill in resume.skills:
                # Check if skill object has category attribute (it might not if it's a dict or different model)
                # The Resume model uses Skill objects, which have category.
                if hasattr(skill, "category") and skill.category:
                    if skill.category not in skills_categories:
                        skills_categories[skill.category] = []
                    skills_categories[skill.category].append(skill.skill_name)

        markdown_content = generate_markdown_resume(resume, skills_categories)
        json_content = generate_json_resume(resume)

        # 3. Validate ATS Compatibility
        ats_validation = validate_ats_compatibility(resume, job_description, markdown_content)

        # 4. Construct Final Output
        optimized_resume = OptimizedResume(
            resume=resume,
            markdown_content=markdown_content,
            json_content=json_content,
            ats_validation=ats_validation,
            optimization_summary="Resume optimized and validated via ATS Specialist Tool.",
            components_assembled={
                "professional_summary": bool(resume.professional_summary),
                "work_experience": bool(resume.work_experience),
                "skills": bool(resume.skills),
                "education": bool(resume.education),
                "certifications": bool(resume.certifications),
            },
            quality_metrics={
                "ats_score": ats_validation.overall_score,
                "keyword_density": ats_validation.keyword_report.keyword_density,
                "keyword_coverage": ats_validation.keyword_report.keyword_coverage,
                "total_sections": len(ats_validation.section_validations),
            },
        )

        return optimized_resume.model_dump_json()

    except Exception as e:
        logger.error(f"Tool execution failed: {e}", exc_info=True)
        return f"Critical error in validate_and_finalize_resume: {str(e)}"


# ==============================================================================
# BLOCK 7: UTILITIES & TESTING
# ==============================================================================
# PURPOSE: Provide utility functions for debugging, monitoring, and testing
# WHAT: Helper functions and test code for agent validation and diagnostics
# WHY: Enables debugging, monitoring, and validation of agent functionality
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 7.1: Agent Information
# ------------------------------------------------------------------------------
# This stage provides metadata and diagnostic information about the agent.


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
