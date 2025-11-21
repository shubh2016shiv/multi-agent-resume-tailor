"""
Job Description Analyst Agent
-----------------------------

This module defines the second agent in our workflow: the Job Description Analyst.
This agent is responsible for analyzing job postings, extracting requirements,
and structuring them into our canonical JobDescription data model.

AGENT DESIGN PRINCIPLES:
- Single Responsibility: Analyze and structure job posting data
- Modularity: Clear separation between agent creation, configuration, and execution
- Robustness: Comprehensive error handling with graceful degradation
- Type Safety: Uses Pydantic models for validated, structured output
- Observability: Detailed logging at every step for debugging

WORKFLOW:
1. Receive job posting file path or text as input
2. Use the parse_job_description tool to convert file to Markdown
3. Analyze Markdown using LLM with JobDescription model schema
4. Extract key requirements: must-haves, nice-to-haves, responsibilities
5. Identify experience level, salary range, and other metadata
6. Return structured JobDescription object (JSON)

KEY ANALYSIS POINTS:
- Required vs preferred qualifications
- Technical skills vs soft skills
- Experience level and seniority
- Company culture and values
- Red flags or unclear requirements
"""

from crewai import Agent
from pydantic import ValidationError

# Handle imports for both package usage and direct script execution
try:
    from src.core.config import get_agents_config
    from src.core.logger import get_logger
    from src.data_models.job import JobDescription
    from src.tools.job_analyzer import parse_job_description
except ImportError:
    # Fallback for when running this file directly
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.core.config import get_agents_config
    from src.core.logger import get_logger
    from src.data_models.job import JobDescription
    from src.tools.job_analyzer import parse_job_description

logger = get_logger(__name__)


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
# Agent Creation
# ==============================================================================


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
        llm_model = config.get("llm", "gemini/gemini-2.5-flash")
        temperature = config.get("temperature", 0.2)
        verbose = config.get("verbose", True)

        # Create the agent
        agent = Agent(
            role=config["role"],
            goal=config["goal"],
            backstory=config["backstory"],
            tools=[parse_job_description],  # Assign the job analyzer tool
            llm=llm_model,
            temperature=temperature,
            verbose=verbose,
            allow_delegation=False,  # This agent works independently
            max_iter=5,  # Limit iterations to prevent infinite loops
        )

        logger.info(
            f"Successfully created agent: {config['role']}, "
            f"using LLM: {llm_model}, temperature: {temperature}"
        )

        return agent

    except Exception as e:
        logger.error(f"Failed to create Job Description Analyst agent: {e}", exc_info=True)
        raise


# ==============================================================================
# Output Validation
# ==============================================================================


def validate_job_output(output_data: dict) -> JobDescription | None:
    """
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
# Analysis Quality Checks
# ==============================================================================


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
# Testing Block
# ==============================================================================

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
