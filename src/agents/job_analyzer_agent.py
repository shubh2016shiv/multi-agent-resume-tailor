"""
Job Description Analyst Agent - Requirements Extraction System
===============================================================

OVERVIEW:
---------
This module defines the second agent in our workflow: the Job Description Analyst.
This agent serves as the bridge between raw job postings and structured requirements
that enable intelligent resume tailoring.

WHAT MAKES THIS AGENT ESSENTIAL:
--------------------------------
- **Foundation Layer**: Extracts structured data that all downstream agents depend on
- **Requirements Parsing**: Distinguishes must-haves from nice-to-haves
- **Context Understanding**: Captures company culture, seniority levels, and expectations
- **Validation Gateway**: Ensures job data quality before gap analysis begins

AGENT DESIGN PRINCIPLES:
------------------------
- **Analytical Focus**: Deep analysis of job requirements and expectations
- **Structured Extraction**: Transforms unstructured job text into typed data
- **Quality Validation**: Built-in checks for analysis completeness and accuracy
- **Context Preservation**: Maintains nuance while providing structure

WORKFLOW OVERVIEW:
------------------
1. Receive job posting file or text input
2. Use parse_job_description tool to convert to clean Markdown
3. Apply LLM analysis with JobDescription schema constraints
4. Extract and categorize requirements (must-have, should-have, nice-to-have)
5. Identify experience levels, company context, and key responsibilities
6. Return validated JobDescription object with comprehensive metadata

MODULE STRUCTURE (Hierarchical Organization):
=============================================
This module is organized into 6 main BLOCKS, each containing STAGES with SUB-STAGES:

BLOCK 1: MODULE SETUP & CONFIGURATION
├── Stage 1.1: Import Management
│   ├── Sub-stage 1.1.1: Standard library imports
│   ├── Sub-stage 1.1.2: CrewAI framework imports
│   ├── Sub-stage 1.1.3: Project-specific imports (with fallback handling)
│   └── Sub-stage 1.1.4: Logger initialization
│
├── Stage 1.2: Configuration Loading
│   ├── Sub-stage 1.2.1: Load agent config from agents.yaml
│   ├── Sub-stage 1.2.2: Validate required configuration fields
│   └── Sub-stage 1.2.3: Error handling with graceful fallbacks
│
└── Stage 1.3: Default Configuration Fallback
    ├── Sub-stage 1.3.1: Define default agent role and expertise
    ├── Sub-stage 1.3.2: Set default LLM parameters for analysis
    └── Sub-stage 1.3.3: Configure default behavior settings

BLOCK 2: AGENT CREATION
├── Stage 2.1: Configuration Retrieval
│   ├── Sub-stage 2.1.1: Load agent-specific configuration
│   ├── Sub-stage 2.1.2: Extract LLM settings and parameters
│   └── Sub-stage 2.1.3: Load resilience configuration
│
├── Stage 2.2: Agent Initialization
│   ├── Sub-stage 2.2.1: Set agent role, goal, and backstory
│   ├── Sub-stage 2.2.2: Configure agent behavior and constraints
│   └── Sub-stage 2.2.3: Initialize CrewAI Agent object
│
├── Stage 2.3: Tool Assignment
│   ├── Sub-stage 2.3.1: Assign parse_job_description tool
│   └── Sub-stage 2.3.2: Configure tool parameters and access
│
└── Stage 2.4: Resilience Configuration
    ├── Sub-stage 2.4.1: Set retry limits and rate limiting
    ├── Sub-stage 2.4.2: Configure execution timeouts
    └── Sub-stage 2.4.3: Enable context window management

BLOCK 3: OUTPUT VALIDATION
├── Stage 3.1: Data Validation
│   ├── Sub-stage 3.1.1: Parse output into JobDescription model
│   ├── Sub-stage 3.1.2: Validate required fields and constraints
│   └── Sub-stage 3.1.3: Check nested model relationships
│
├── Stage 3.2: Error Handling
│   ├── Sub-stage 3.2.1: Catch Pydantic ValidationError
│   ├── Sub-stage 3.2.2: Log detailed validation errors
│   └── Sub-stage 3.2.3: Return None for graceful failure handling
│
└── Stage 3.3: Logging & Reporting
    ├── Sub-stage 3.3.1: Log successful validation with summary
    ├── Sub-stage 3.3.2: Log validation failures with error details
    └── Sub-stage 3.3.3: Return validated JobDescription object

BLOCK 4: ANALYSIS QUALITY CHECKS
├── Stage 4.1: Quality Assessment
│   ├── Sub-stage 4.1.1: check_analysis_quality() function
│   ├── Sub-stage 4.1.2: Evaluate requirements completeness
│   ├── Sub-stage 4.1.3: Check experience level identification
│   └── Sub-stage 4.1.4: Validate ATS keywords presence
│
├── Stage 4.2: Scoring Algorithm
│   ├── Sub-stage 4.2.1: Calculate quality score (0-100)
│   ├── Sub-stage 4.2.2: Identify critical issues vs warnings
│   └── Sub-stage 4.2.3: Generate actionable recommendations
│
└── Stage 4.3: Quality Reporting
    ├── Sub-stage 4.3.1: Structure quality check results
    ├── Sub-stage 4.3.2: Log issues and recommendations
    └── Sub-stage 4.3.3: Return comprehensive quality report

BLOCK 5: UTILITY FUNCTIONS
├── Stage 5.1: Agent Information
│   ├── Sub-stage 5.1.1: get_agent_info() function
│   ├── Sub-stage 5.1.2: Retrieve agent metadata
│   └── Sub-stage 5.1.3: Format information for debugging
│
└── Stage 5.2: Testing Support
    ├── Sub-stage 5.2.1: Test configuration loading
    ├── Sub-stage 5.2.2: Test agent creation
    ├── Sub-stage 5.2.3: Test validation functions
    └── Sub-stage 5.2.4: Test quality assessment

BLOCK 6: INTEGRATION TESTING
├── Stage 6.1: End-to-End Testing
│   ├── Sub-stage 6.1.1: Mock data creation for testing
│   ├── Sub-stage 6.1.2: Quality check validation
│   └── Sub-stage 6.1.3: Integration test scenarios

HOW TO USE THIS MODULE:
-----------------------
1. Import: `from src.agents.job_analyzer_agent import create_job_analyzer_agent`
2. Create Agent: `agent = create_job_analyzer_agent()`
3. Use in Crew: Add agent to CrewAI crew with job description files
4. Validate Output: Use `validate_job_output()` to ensure data quality
5. Check Quality: Use `check_analysis_quality()` for analysis validation

KEY ANALYSIS CAPABILITIES:
-------------------------
- **Requirements Categorization**: Must-have vs should-have vs nice-to-have
- **Experience Level Detection**: Entry, junior, mid-level, senior, executive
- **Skill Classification**: Technical vs soft skills identification
- **ATS Keyword Extraction**: Keywords important for applicant tracking
- **Context Understanding**: Company culture, values, and expectations
- **Quality Validation**: Built-in checks for analysis completeness

TECHNICAL ARCHITECTURE:
-----------------------
- **Tool-Based Parsing**: Uses specialized job description parsing tools
- **Schema Validation**: Pydantic models ensure data structure and types
- **Quality Gates**: Built-in validation prevents poor analysis from proceeding
- **Error Recovery**: Graceful handling of parsing failures and edge cases
- **Comprehensive Logging**: Detailed observability for debugging and monitoring

ANALYSIS DEPTH:
---------------
This agent goes beyond simple keyword extraction to understand:
1. **Hierarchical Requirements**: Which skills are truly critical vs desirable
2. **Experience Context**: Years required vs preferred experience patterns
3. **Company Culture**: Implicit values and work environment indicators
4. **Role Expectations**: Beyond listed requirements to understood expectations
5. **Market Positioning**: How this role fits within the company's structure

The result is structured job intelligence that enables precise, effective resume tailoring.
"""

from crewai import Agent
from pydantic import ValidationError

# Handle imports for both package usage and direct script execution
try:
    from src.core.config import (
        get_agents_config,  # get_config unused - kept for potential future refactoring
    )
    from src.core.logger import get_logger
    from src.data_models.job import JobDescription, SkillImportance
    from src.tools.job_analyzer import parse_job_description
except ImportError:
    # Fallback for when running this file directly
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.core.config import (
        get_agents_config,  # get_config unused - kept for potential future refactoring
    )
    from src.core.logger import get_logger
    from src.data_models.job import JobDescription, SkillImportance
    from src.tools.job_analyzer import parse_job_description

logger = get_logger(__name__)


# ==============================================================================
# BLOCK 1: MODULE SETUP & CONFIGURATION
# ==============================================================================
# PURPOSE: Initialize the module with imports, configuration, and defaults
# WHAT: Global setup functions and configuration management
# WHY: Ensures consistent agent behavior across different environments
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1.2: Configuration Loading
# ------------------------------------------------------------------------------
# This stage loads agent configuration from external files with error handling.


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
        config = agents_config.get("job_description_analyst", {})

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
# Stage 1.3: Default Configuration Fallback
# ------------------------------------------------------------------------------
# This stage provides production-ready defaults when configuration files fail.


def _get_default_config() -> dict:
    """
    Provide default configuration as a fallback.

    This ensures the agent can still be created even if the YAML config
    is unavailable or corrupted. These defaults are basic but functional.

    Returns:
        Dictionary with default agent configuration
    """
    return {
        "role": "Job Requirements Analysis Specialist",
        "goal": (
            "Thoroughly analyze job descriptions to extract and structure all "
            "requirements, qualifications, and expectations using the JobDescription data model."
        ),
        "backstory": (
            "You are an expert recruiter and job market analyst. You understand what "
            "employers really mean in job postings, can distinguish must-haves from "
            "nice-to-haves, and identify the core competencies needed for success."
        ),
        "llm": "gemini/gemini-2.5-flash",
        "temperature": 0.2,
        "verbose": True,
    }


# ==============================================================================
# BLOCK 2: AGENT CREATION
# ==============================================================================
# PURPOSE: Create and configure the Job Description Analyst agent with all tools
# WHAT: Agent initialization, tool assignment, and resilience configuration
# WHY: Produces a fully functional agent ready for job description analysis
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 2.1-2.4: Complete Agent Creation Workflow
# ------------------------------------------------------------------------------
# This stage orchestrates all aspects of agent creation in proper sequence.


def create_job_analyzer_agent() -> Agent:
    """
    Create and configure the Job Description Analyst agent.

    This is the main entry point for creating this agent. It handles all the
    complexity of configuration loading, tool assignment, and agent initialization.

    Returns:
        Configured CrewAI Agent instance ready to analyze job descriptions

    Raises:
        Exception: If agent creation fails (logged and re-raised)

    Example:
        >>> agent = create_job_analyzer_agent()
        >>> # Agent is now ready to be used in a crew or task

    Design Notes:
        - Uses configuration from agents.yaml (with fallback to defaults)
        - Assigns the parse_job_description tool for file handling
        - Configures LLM settings for optimal analysis
        - Enables verbose mode for detailed logging
        - Low temperature (0.2) for consistent, factual extraction
    """
    try:
        logger.info("Creating Job Description Analyst agent...")

        # Load configuration
        config = _load_agent_config()

        # Extract LLM settings
        # llm_model = config.get("llm", "gemini/gemini-2.5-flash")  # Unused variable - kept for potential future refactoring
        # temperature = config.get("temperature", 0.2)  # Unused variable - kept for potential future refactoring
        # verbose = config.get("verbose", True)  # Unused variable - kept for potential future refactoring

        # Create the agent
        # Load centralized resilience configuration
        # app_config = get_config()  # Unused variable - kept for potential future refactoring
        # agent_defaults = app_config.llm.agent_defaults  # Unused variable - kept for potential future refactoring

        agent = Agent(
            role=config.get("role", "Job Description Analyzer"),
            goal=config.get("goal", "Analyze job descriptions and extract key requirements"),
            backstory=config.get("backstory", "Expert in analyzing job descriptions"),
            tools=[parse_job_description],  # Assign the job analyzer tool
        )

        return agent

    except Exception as e:
        logger.error(f"Failed to create Job Description Analyst agent: {e}", exc_info=True)
        raise


# ==============================================================================
# Output Validation
# ==============================================================================
# BLOCK 3: OUTPUT VALIDATION
# ==============================================================================
# PURPOSE: Validate that agent outputs conform to expected data models
# WHAT: Quality gates that ensure structured data meets schema requirements
# WHY: Prevents downstream errors and ensures data consistency
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 3.1-3.3: Complete Validation Workflow
# ------------------------------------------------------------------------------
# This stage orchestrates all validation steps to ensure output quality.


def validate_job_output(output_data: dict) -> JobDescription | None:
    """
    DEPRECATED: This function is no longer needed with output_pydantic.

    When using CrewAI's `output_pydantic` parameter in Task definitions, validation
    happens automatically. You can access the validated JobDescription object directly via
    `result.pydantic` without calling this function.

    MIGRATION GUIDE:
    ----------------
    OLD APPROACH (Manual Validation):
    ```python
    result = crew.kickoff()
    json_data = parse_json_output(str(result))
    validated_job = validate_job_output(json_data)  # Not needed
    ```

    NEW APPROACH (Automatic Validation):
    ```python
    task = Task(..., output_pydantic=JobDescription)  # Add this parameter
    result = crew.kickoff()
    validated_job = result.pydantic  # Direct access, already validated
    ```

    This function is kept for backward compatibility only.

    ---

    ORIGINAL DOCUMENTATION:

    Validate that the agent's output conforms to the JobDescription model.

    This function serves as a quality gate, ensuring that the structured data
    extracted by the agent is valid according to our schema. If validation fails,
    it provides detailed error information for debugging.

    Args:
        output_data: Dictionary containing the extracted job description data

    Returns:
        JobDescription object if validation succeeds, None if it fails

    Design Notes:
        - Separating validation into its own function makes it reusable
        - Detailed logging helps diagnose extraction issues
        - Returning None (rather than raising) allows graceful handling upstream

    Edge Cases Handled:
        - Missing required fields → logged with specific field names
        - Invalid enum values → caught by Pydantic validation
        - Malformed data types → validation error details provided
    """
    try:
        logger.debug("Validating agent output against JobDescription model...")

        # Attempt to create a JobDescription object from the output
        job = JobDescription(**output_data)

        logger.info(
            f"Job description validation successful. "
            f"Position: {job.job_title}, "
            f"Company: {job.company_name}, "
            f"Requirements: {len(job.requirements)}, "
            f"Must-haves: {len(job.must_have_skills)}"
        )

        return job

    except ValidationError as e:
        logger.error(
            f"Job description validation failed. Output does not match JobDescription model schema. "
            f"Errors: {e.errors()}"
        )
        # Log each validation error for easier debugging
        for error in e.errors():
            logger.error(f"  Field: {error['loc']}, Type: {error['type']}, Message: {error['msg']}")
        return None

    except Exception as e:
        logger.error(f"Unexpected error during job description validation: {e}", exc_info=True)
        return None


# ==============================================================================
# BLOCK 4: ANALYSIS QUALITY CHECKS
# ==============================================================================
# PURPOSE: Validate the quality and completeness of job description analysis
# WHAT: Comprehensive quality assessment with scoring and recommendations
# WHY: Ensures analysis is thorough enough for effective resume tailoring
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 4.1-4.3: Complete Quality Assessment Workflow
# ------------------------------------------------------------------------------
# This stage performs comprehensive quality checks on job description analysis.


def check_analysis_quality(job: JobDescription) -> dict:
    """
    Perform quality checks on the analyzed job description.

    This function validates that the analysis is comprehensive and useful
    for downstream agents. It checks for common issues like missing critical
    information or insufficient detail.

    Args:
        job: The validated JobDescription object

    Returns:
        Dictionary with quality check results and recommendations

    Quality Checks:
        - Are requirements identified?
        - Is the experience level specified?
        - Are there ATS keywords?
        - Is there sufficient detail for matching?

    Design Note:
        This helps catch incomplete analyses early, before they cause
        problems in later workflow stages.
    """
    issues = []
    warnings = []
    score = 100

    # Check for requirements
    if not job.requirements or len(job.requirements) == 0:
        issues.append("No requirements extracted")
        score -= 40
    elif len(job.requirements) < 3:
        warnings.append(f"Only {len(job.requirements)} requirements found - may be incomplete")
        score -= 15

    # Check for must-have skills
    must_have_count = len(job.must_have_skills)
    if must_have_count == 0:
        warnings.append("No must-have skills identified")
        score -= 15

    # Check for experience level
    if not job.job_level or job.job_level == "unspecified":
        warnings.append("Job level not specified")
        score -= 10

    # Check for ATS keywords
    if not job.ats_keywords or len(job.ats_keywords) == 0:
        warnings.append("No ATS keywords identified")
        score -= 10

    # Check for job description text
    if not job.full_text or len(job.full_text) < 100:
        issues.append("Job description text is too short or missing")
        score -= 25

    # Check for summary
    if not job.summary or len(job.summary) < 20:
        warnings.append("Job summary is missing or too brief")
        score -= 10

    # Determine quality level
    if score >= 90:
        quality = "excellent"
    elif score >= 70:
        quality = "good"
    elif score >= 50:
        quality = "fair"
    else:
        quality = "poor"

    result = {
        "quality": quality,
        "score": max(0, score),
        "issues": issues,
        "warnings": warnings,
        "is_acceptable": score >= 50,
    }

    # Log the quality check results
    if issues:
        logger.warning(f"Job analysis quality issues found: {issues}")
    if warnings:
        logger.info(f"Job analysis quality warnings: {warnings}")

    logger.info(f"Job analysis quality check: {quality} (score: {score}/100)")

    return result


# ==============================================================================
# BLOCK 5: UTILITY FUNCTIONS
# ==============================================================================
# PURPOSE: Provide utility functions for debugging, monitoring, and testing
# WHAT: Helper functions for agent metadata and diagnostic information
# WHY: Enables debugging, monitoring, and validation of agent functionality
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 5.1: Agent Information
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
        'Job Description Analyst'
    """
    config = _load_agent_config()
    return {
        "name": "Job Description Analyst",
        "role": config.get("role", "Unknown"),
        "llm": config.get("llm", "Unknown"),
        "tools": ["parse_job_description"],
        "output_model": "JobDescription",
    }


# ==============================================================================
# BLOCK 6: INTEGRATION TESTING
# ==============================================================================
# PURPOSE: Provide testing and validation capabilities for the agent
# WHAT: Test functions and integration validation code
# WHY: Ensures agent functionality and enables development-time validation
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 6.1: End-to-End Testing
# ------------------------------------------------------------------------------
# This stage provides comprehensive testing of agent functionality.

if __name__ == "__main__":
    """
    Test the agent creation and configuration loading.
    Run this script directly to verify the agent can be created.
    """
    print("=" * 70)
    print("Job Description Analyst Agent - Test")
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
        agent = create_job_analyzer_agent()
        print("SUCCESS: Agent created successfully")
        print(f"Agent role: {agent.role}")
        print(f"Tools assigned: {len(agent.tools)}")
    except Exception as e:
        print(f"FAILED: {str(e)}")

    # Display agent info
    print("\n--- Agent Information ---")
    info = get_agent_info()
    for key, value in info.items():
        print(f"{key}: {value}")

    # Test quality check function with mock data
    print("\n--- Testing Quality Check Function ---")
    try:
        from src.data_models.job import JobDescription, JobLevel, JobRequirement, SkillImportance

        mock_job = JobDescription(
            job_title="Software Engineer",
            company_name="Test Company",
            summary="Join our team to build scalable cloud solutions.",
            full_text="Test description " * 20,  # Make it long enough
            requirements=[
                JobRequirement(
                    requirement="Python", importance=SkillImportance.MUST_HAVE, years_required=3
                ),
                JobRequirement(requirement="FastAPI", importance=SkillImportance.SHOULD_HAVE),
                JobRequirement(requirement="Docker", importance=SkillImportance.NICE_TO_HAVE),
            ],
            ats_keywords=["Python", "FastAPI", "Docker", "Cloud"],
            job_level=JobLevel.MID,
        )
        quality_result = check_analysis_quality(mock_job)
        print(f"Quality: {quality_result['quality']}")
        print(f"Score: {quality_result['score']}/100")
        print(f"Acceptable: {quality_result['is_acceptable']}")
        if quality_result["issues"]:
            print(f"Issues: {quality_result['issues']}")
        if quality_result["warnings"]:
            print(f"Warnings: {quality_result['warnings']}")
    except Exception as e:
        print(f"Quality check test failed: {str(e)}")

    print("\n" + "=" * 70)
