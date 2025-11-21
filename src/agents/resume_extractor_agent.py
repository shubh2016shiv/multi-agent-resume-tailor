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
