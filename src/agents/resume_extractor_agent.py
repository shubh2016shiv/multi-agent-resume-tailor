"""
Resume Content Extractor Agent
------------------------------

This module defines the first agent in our workflow: the Resume Content Extractor.
This agent is responsible for taking a resume file, extracting its content, and
structuring it into our canonical Resume data model.

AGENT DESIGN PRINCIPLES:
- Single Responsibility: This agent does ONE thing well - extract and structure resume data
- Modularity: Clear separation between agent creation, configuration, and execution
- Robustness: Comprehensive error handling with graceful degradation
- Type Safety: Uses Pydantic models for validated, structured output
- Observability: Detailed logging at every step for debugging

WORKFLOW:
1. Receive resume file path as input
2. Use the parse_resume tool to convert file to Markdown
3. Analyze Markdown using LLM with Resume model schema
4. Return structured Resume object (JSON)
"""

from crewai import Agent
from pydantic import ValidationError

# Handle imports for both package usage and direct script execution
try:
    from src.core.config import get_agents_config
    from src.core.logger import get_logger
    from src.data_models.resume import Resume
    from src.tools.resume_parser import parse_resume
except ImportError:
    # Fallback for when running this file directly
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.core.config import get_agents_config
    from src.core.logger import get_logger
    from src.data_models.resume import Resume
    from src.tools.resume_parser import parse_resume

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
        config = agents_config.get("resume_content_extractor", {})

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
        "role": "Resume Data Extraction Specialist",
        "goal": (
            "Parse and structure all information from resumes into comprehensive, "
            "machine-readable format using the Resume data model."
        ),
        "backstory": (
            "You are an expert in resume analysis and data structuring. You extract "
            "structured information from resumes, identify achievements, and understand "
            "career narratives to enable strategic comparison and optimization."
        ),
        "llm": "gemini/gemini-2.5-flash",
        "temperature": 0.2,
        "verbose": True,
    }


# ==============================================================================
# Agent Creation
# ==============================================================================


def create_resume_extractor_agent() -> Agent:
    """
    Create and configure the Resume Content Extractor agent.

    This is the main entry point for creating this agent. It handles all the
    complexity of configuration loading, tool assignment, and agent initialization.

    Returns:
        Configured CrewAI Agent instance ready to extract resume data

    Raises:
        Exception: If agent creation fails (logged and re-raised)

    Example:
        >>> agent = create_resume_extractor_agent()
        >>> # Agent is now ready to be used in a crew or task

    Design Notes:
        - Uses configuration from agents.yaml (with fallback to defaults)
        - Assigns the parse_resume tool for file handling
        - Configures LLM settings for optimal extraction
        - Enables verbose mode for detailed logging
    """
    try:
        logger.info("Creating Resume Content Extractor agent...")

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
            tools=[parse_resume],  # Assign the resume parser tool
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
        logger.error(f"Failed to create Resume Content Extractor agent: {e}", exc_info=True)
        raise


# ==============================================================================
# Output Validation
# ==============================================================================


def validate_resume_output(output_data: dict) -> Resume | None:
    """
    Validate that the agent's output conforms to the Resume model.

    This function serves as a quality gate, ensuring that the structured data
    extracted by the agent is valid according to our schema. If validation fails,
    it provides detailed error information for debugging.

    Args:
        output_data: Dictionary containing the extracted resume data

    Returns:
        Resume object if validation succeeds, None if it fails

    Design Notes:
        - Separating validation into its own function makes it reusable
        - Detailed logging helps diagnose extraction issues
        - Returning None (rather than raising) allows graceful handling upstream
    """
    try:
        logger.debug("Validating agent output against Resume model...")

        # Attempt to create a Resume object from the output
        resume = Resume(**output_data)

        logger.info(
            f"Resume validation successful. "
            f"Candidate: {resume.full_name}, "
            f"Experience: {len(resume.work_experience)} jobs, "
            f"Skills: {len(resume.skills)} skills"
        )

        return resume

    except ValidationError as e:
        logger.error(
            f"Resume validation failed. Output does not match Resume model schema. "
            f"Errors: {e.errors()}"
        )
        return None

    except Exception as e:
        logger.error(f"Unexpected error during resume validation: {e}", exc_info=True)
        return None


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
        'Resume Content Extractor'
    """
    config = _load_agent_config()
    return {
        "name": "Resume Content Extractor",
        "role": config.get("role", "Unknown"),
        "llm": config.get("llm", "Unknown"),
        "tools": ["parse_resume"],
        "output_model": "Resume",
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
    print("Resume Content Extractor Agent - Test")
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
        agent = create_resume_extractor_agent()
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

    print("\n" + "=" * 70)
