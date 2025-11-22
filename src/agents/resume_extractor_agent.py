"""
Resume Content Extractor Agent
==============================

OVERVIEW:
---------
This module defines the first agent in our workflow: the Resume Content Extractor.
This agent is responsible for taking a resume file, extracting its content, and
structuring it into our canonical Resume data model.

AGENT DESIGN PRINCIPLES:
------------------------
- Single Responsibility: This agent does ONE thing well - extract and structure resume data
- Modularity: Clear separation between agent creation, configuration, and execution
- Robustness: Comprehensive error handling with graceful degradation
- Type Safety: Uses Pydantic models for validated, structured output
- Observability: Detailed logging at every step for debugging

WORKFLOW OVERVIEW:
------------------
1. Receive resume file path as input
2. Use the parse_resume tool to convert file to Markdown
3. Analyze Markdown using LLM with Resume model schema
4. Return structured Resume object (JSON)

STRUCTURED OUTPUT ENFORCEMENT (Industry-Standard Approach):
--------------------------------------------------------------
This agent uses CrewAI's `output_pydantic` parameter to enforce structured outputs.
This is the RECOMMENDED approach for ensuring LLM outputs match Pydantic schemas.

HOW IT WORKS:
- When creating a Task, set `output_pydantic=Resume`
- CrewAI automatically provides the Resume schema to the LLM
- The LLM generates output that conforms to the schema
- CrewAI validates the output against the Pydantic model
- If validation fails, CrewAI retries automatically (up to max_retry_limit)
- The final result is a validated Resume object accessible via `result.pydantic`

BENEFITS:
- No manual JSON parsing required
- No manual validation logic needed
- Automatic retry on validation failures
- Type-safe access to structured data
- Follows CrewAI best practices

EXAMPLE USAGE:
--------------
```python
from crewai import Task
from src.agents.resume_extractor_agent import create_resume_extractor_agent
from src.data_models.resume import Resume

agent = create_resume_extractor_agent()

task = Task(
    description="Extract resume content from the provided file...",
    expected_output="A structured Resume object with all fields populated",
    agent=agent,
    output_pydantic=Resume,  # This enforces structured output
)

# Execute and access validated output
result = crew.kickoff()
validated_resume = result.pydantic  # Direct access to Resume object
print(validated_resume.full_name)
print(validated_resume.work_experience)
```

MODULE STRUCTURE (Hierarchical Organization):
----------------------------------------------
This module is organized into 4 main BLOCKS, each containing STAGES with SUB-STAGES:

BLOCK 1: MODULE SETUP & CONFIGURATION
├── Stage 1.1: Import Management
│   ├── Sub-stage 1.1.1: Standard library imports
│   ├── Sub-stage 1.1.2: CrewAI framework imports
│   ├── Sub-stage 1.1.3: Project-specific imports (with fallback handling)
│   └── Sub-stage 1.1.4: Logger initialization
│
├── Stage 1.2: Configuration Loading
│   ├── Sub-stage 1.2.1: Load agent config from YAML
│   ├── Sub-stage 1.2.2: Validate required configuration fields
│   └── Sub-stage 1.2.3: Handle configuration errors gracefully
│
└── Stage 1.3: Default Configuration Fallback
    ├── Sub-stage 1.3.1: Define default agent role and goal
    ├── Sub-stage 1.3.2: Set default LLM and temperature
    └── Sub-stage 1.3.3: Configure default behavior settings

BLOCK 2: AGENT CREATION
├── Stage 2.1: Configuration Retrieval
│   ├── Sub-stage 2.1.1: Load agent-specific configuration
│   ├── Sub-stage 2.1.2: Extract LLM settings (model, temperature)
│   └── Sub-stage 2.1.3: Load application-wide resilience settings
│
├── Stage 2.2: Agent Initialization
│   ├── Sub-stage 2.2.1: Set agent role, goal, and backstory
│   ├── Sub-stage 2.2.2: Configure agent behavior (verbose, delegation)
│   └── Sub-stage 2.2.3: Initialize CrewAI Agent object
│
├── Stage 2.3: Tool Assignment
│   ├── Sub-stage 2.3.1: Import parse_resume tool
│   └── Sub-stage 2.3.2: Assign tool to agent capabilities
│
└── Stage 2.4: Resilience Configuration
    ├── Sub-stage 2.4.1: Set retry limits and rate limits
    ├── Sub-stage 2.4.2: Configure execution timeouts
    └── Sub-stage 2.4.3: Enable context window management

BLOCK 3: UTILITY FUNCTIONS
├── Stage 3.1: Agent Information
│   ├── Sub-stage 3.1.1: Retrieve agent metadata
│   ├── Sub-stage 3.1.2: Format agent information dictionary
│   └── Sub-stage 3.1.3: Return structured agent info
│
└── Stage 3.2: Testing Support
    ├── Sub-stage 3.2.1: Test configuration loading
    ├── Sub-stage 3.2.2: Test agent creation
    └── Sub-stage 3.2.3: Display agent information

NOTE: The validate_resume_output() function has been REMOVED as it's no longer
needed with output_pydantic. CrewAI handles validation automatically.

HOW TO USE THIS MODULE:
-----------------------
1. Import: `from src.agents.resume_extractor_agent import create_resume_extractor_agent`
2. Create Agent: `agent = create_resume_extractor_agent()`
3. Create Task with output_pydantic: `Task(..., output_pydantic=Resume)`
4. Access Output: `validated_resume = result.pydantic`
"""

from crewai import Agent
from pydantic import ValidationError

# Handle imports for both package usage and direct script execution
try:
    from src.core.config import get_agents_config, get_config
    from src.core.logger import get_logger
    from src.data_models.resume import Resume
    from src.tools.resume_parser import parse_resume
except ImportError:
    # Fallback for when running this file directly
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.core.config import get_agents_config, get_config
    from src.core.logger import get_logger
    from src.data_models.resume import Resume
    from src.tools.resume_parser import parse_resume

logger = get_logger(__name__)


# ==============================================================================
# BLOCK 1: MODULE SETUP & CONFIGURATION
# ==============================================================================
# This block handles all setup tasks: imports, configuration loading, and defaults.
# It ensures the module can function even if configuration files are missing.

# ------------------------------------------------------------------------------
# Stage 1.2: Configuration Loading
# ------------------------------------------------------------------------------
# This stage loads agent configuration from YAML files with proper error handling.
# It validates that required fields are present and falls back to defaults if needed.


def _load_agent_config() -> dict:
    """
    Load the agent configuration from agents.yaml.

    STAGE: 1.2 - Configuration Loading
    PURPOSE: Retrieve agent-specific settings from configuration files

    SUB-STAGES:
    -----------
    1.2.1: Load Configuration from YAML
        - Calls get_agents_config() to retrieve all agent configurations
        - Extracts the "resume_extractor" section specifically
        - Returns empty dict if section doesn't exist

    1.2.2: Validate Required Fields
        - Checks for presence of: role, goal, backstory
        - Identifies any missing fields
        - Logs warning if fields are missing

    1.2.3: Error Handling & Fallback
        - Catches any exceptions during config loading
        - Logs error with full traceback for debugging
        - Returns default configuration as fallback

    Returns:
        dict: Agent configuration dictionary with role, goal, backstory, etc.

    Design Notes:
        - Graceful degradation: Always returns a valid config (defaults if needed)
        - Detailed logging helps diagnose configuration issues
        - Separated from agent creation for testability
    """
    try:
        # SUB-STAGE 1.2.1: Load Configuration from YAML
        agents_config = get_agents_config()
        config = agents_config.get("resume_extractor", {})

        # SUB-STAGE 1.2.2: Validate Required Fields
        required_fields = ["role", "goal", "backstory"]
        missing_fields = [f for f in required_fields if f not in config]

        if missing_fields:
            logger.warning(f"Agent config missing fields: {missing_fields}. Using defaults.")
            # SUB-STAGE 1.2.3: Fallback to defaults
            return _get_default_config()

        logger.debug("Successfully loaded agent configuration from YAML")
        return config

    except Exception as e:
        # SUB-STAGE 1.2.3: Error Handling
        logger.error(f"Failed to load agent config: {e}. Using defaults.", exc_info=True)
        return _get_default_config()


# ------------------------------------------------------------------------------
# Stage 1.3: Default Configuration Fallback
# ------------------------------------------------------------------------------
# This stage provides sensible defaults when configuration files are unavailable.
# These defaults ensure the agent can still function in any environment.


def _get_default_config() -> dict:
    """
    Provide default configuration as a fallback.

    STAGE: 1.3 - Default Configuration Fallback
    PURPOSE: Ensure agent can function even without configuration files

    SUB-STAGES:
    -----------
    1.3.1: Define Default Agent Identity
        - Sets role: "Resume Extraction Specialist"
        - Defines clear goal for the agent's purpose
        - Creates backstory that explains agent's expertise

    1.3.2: Set Default LLM Configuration
        - Model: "gemini/gemini-2.5-flash" (fast, cost-effective)
        - Temperature: 0.0 (deterministic, factual extraction)
        - Ensures consistent, accurate data extraction

    1.3.3: Configure Default Behavior
        - Verbose: True (detailed logging for debugging)
        - Provides transparency into agent operations

    Returns:
        dict: Complete default configuration dictionary

    Design Notes:
        - Defaults are production-ready, not just placeholders
        - Temperature 0.0 ensures consistent extraction (no creativity needed)
        - Verbose mode helps with debugging and monitoring
    """
    # SUB-STAGE 1.3.1: Define Default Agent Identity
    return {
        "role": "Resume Extraction Specialist",
        "goal": (
            "Extract all relevant information from the resume file and structure it "
            "into the canonical Resume data model."
        ),
        "backstory": (
            "You are an expert data extraction specialist with deep knowledge of resume formats. "
            "You can parse complex documents and identify key details like skills, experience, "
            "and education with high accuracy."
        ),
        # SUB-STAGE 1.3.2: Set Default LLM Configuration
        "llm": "gemini/gemini-2.5-flash",
        "temperature": 0.0,  # Deterministic extraction, no creativity needed
        # SUB-STAGE 1.3.3: Configure Default Behavior
        "verbose": True,  # Detailed logging for debugging
    }


# ==============================================================================
# BLOCK 2: AGENT CREATION
# ==============================================================================
# This block handles the creation and configuration of the CrewAI agent.
# It combines configuration, tools, and resilience settings into a working agent.

# ------------------------------------------------------------------------------
# Stage 2.1-2.4: Complete Agent Creation Workflow
# ------------------------------------------------------------------------------
# This function orchestrates all stages of agent creation in sequence.


def create_resume_extractor_agent() -> Agent:
    """
    Create and configure the Resume Content Extractor agent.

    STAGE: 2.1-2.4 - Complete Agent Creation Workflow
    PURPOSE: Orchestrate all stages to create a fully configured agent

    WORKFLOW STAGES:
    ---------------
    Stage 2.1: Configuration Retrieval
        Sub-stage 2.1.1: Load agent-specific configuration
        Sub-stage 2.1.2: Extract LLM settings (model, temperature, verbose)
        Sub-stage 2.1.3: Load application-wide resilience settings

    Stage 2.2: Agent Initialization
        Sub-stage 2.2.1: Set agent role, goal, and backstory from config
        Sub-stage 2.2.2: Configure agent behavior (verbose, no delegation)
        Sub-stage 2.2.3: Initialize CrewAI Agent object

    Stage 2.3: Tool Assignment
        Sub-stage 2.3.1: Import parse_resume tool (already imported at module level)
        Sub-stage 2.3.2: Assign tool to agent's capabilities list

    Stage 2.4: Resilience Configuration
        Sub-stage 2.4.1: Set retry limits (max_retry_limit)
        Sub-stage 2.4.2: Set rate limits (max_rpm - requests per minute)
        Sub-stage 2.4.3: Set iteration limits (max_iter)
        Sub-stage 2.4.4: Set execution timeouts (max_execution_time)
        Sub-stage 2.4.5: Enable context window management (respect_context_window)

    Returns:
        Agent: Fully configured CrewAI Agent ready for use in crews

    Raises:
        Exception: If agent creation fails (logged with full traceback)

    Design Notes:
        - All configuration comes from centralized sources (YAML + app config)
        - Resilience settings prevent agent from hanging or over-consuming resources
        - Tool assignment enables agent to parse resume files
        - No delegation allowed - this agent has a single, focused responsibility
    """
    try:
        logger.info("Creating Resume Content Extractor agent...")

        # ======================================================================
        # STAGE 2.1: Configuration Retrieval
        # ======================================================================

        # SUB-STAGE 2.1.1: Load agent-specific configuration
        config = _load_agent_config()

        # SUB-STAGE 2.1.2: Extract LLM settings
        # llm_model = config.get("llm", "gemini/gemini-2.5-flash")  # Unused variable - kept for potential future refactoring
        # temperature = config.get("temperature", 0.0)  # Unused variable - kept for potential future refactoring
        verbose = config.get("verbose", True)

        # SUB-STAGE 2.1.3: Load application-wide resilience settings
        app_config = get_config()
        agent_defaults = app_config.llm.agent_defaults

        # ======================================================================
        # STAGE 2.2: Agent Initialization
        # ======================================================================

        # SUB-STAGE 2.2.1: Set agent identity from configuration
        # SUB-STAGE 2.2.2: Configure behavior (verbose logging, no delegation)
        # SUB-STAGE 2.2.3: Initialize CrewAI Agent object
        agent = Agent(
            role=config.get("role", "Resume Extractor"),
            goal=config.get("goal", "Extract structured data from resumes"),
            backstory=config.get("backstory", "Expert in resume parsing"),
            # SUB-STAGE 2.3.2: Assign tool to agent
            tools=[parse_resume],  # parse_resume tool imported at module level
            verbose=verbose,
            allow_delegation=False,  # Single responsibility - no delegation needed
            # ==================================================================
            # STAGE 2.4: Resilience Configuration
            # ==================================================================
            # SUB-STAGE 2.4.1-2.4.5: Set all resilience parameters
            max_retry_limit=agent_defaults.max_retry_limit,  # Max retries on failure
            max_rpm=agent_defaults.max_rpm,  # Rate limiting (requests per minute)
            max_iter=agent_defaults.max_iter,  # Max iterations per task
            max_execution_time=agent_defaults.max_execution_time,  # Timeout protection
            respect_context_window=agent_defaults.respect_context_window,  # Token management
        )

        logger.info(
            f"Resume Extractor agent created successfully with resilience: "
            f"max_retry={agent_defaults.max_retry_limit}, max_rpm={agent_defaults.max_rpm}"
        )

        return agent

    except Exception as e:
        logger.error(f"Failed to create Resume Content Extractor agent: {e}", exc_info=True)
        raise


# ==============================================================================
# BLOCK 3: OUTPUT VALIDATION
# ==============================================================================
# This block validates that agent outputs conform to our data models.
# It serves as a quality gate before data flows to downstream agents.

# ------------------------------------------------------------------------------
# Stage 3.1-3.3: Complete Validation Workflow
# ------------------------------------------------------------------------------
# This function orchestrates all validation stages to ensure data quality.


def validate_resume_output(output_data: dict) -> Resume | None:
    """
    DEPRECATED: This function is no longer needed with output_pydantic.
    
    When using CrewAI's `output_pydantic` parameter in Task definitions, validation
    happens automatically. You can access the validated Resume object directly via
    `result.pydantic` without calling this function.
    
    MIGRATION GUIDE:
    ----------------
    OLD APPROACH (Manual Validation):
    ```python
    result = crew.kickoff()
    json_data = parse_json_output(str(result))
    validated_resume = validate_resume_output(json_data)  # Not needed
    ```
    
    NEW APPROACH (Automatic Validation):
    ```python
    task = Task(..., output_pydantic=Resume)  # Add this parameter
    result = crew.kickoff()
    validated_resume = result.pydantic  # Direct access, already validated
    ```
    
    This function is kept for backward compatibility only.
    
    ---
    
    ORIGINAL DOCUMENTATION:
    
    Validate that the agent's output conforms to the Resume model.

    STAGE: 3.1-3.3 - Complete Validation Workflow
    PURPOSE: Ensure extracted data matches our schema before use

    WORKFLOW STAGES:
    ---------------
    Stage 3.1: Data Validation
        Sub-stage 3.1.1: Attempt to create Resume object from output_data
        Sub-stage 3.1.2: Pydantic automatically validates:
            - Required fields presence (full_name, email, etc.)
            - Data types (dates as strings, years as integers)
            - Field constraints (email format, date format)
            - Enum values (if any)
        Sub-stage 3.1.3: Validate nested models (Experience, Education, etc.)

    Stage 3.2: Error Handling
        Sub-stage 3.2.1: Catch Pydantic ValidationError (schema violations)
        Sub-stage 3.2.2: Log detailed validation errors for debugging
        Sub-stage 3.2.3: Catch unexpected exceptions (programming errors)
        Sub-stage 3.2.4: Return None for graceful failure handling

    Stage 3.3: Logging & Reporting
        Sub-stage 3.3.1: Log successful validation with summary statistics
        Sub-stage 3.3.2: Log validation failures with error details
        Sub-stage 3.3.3: Return validated Resume object (or None)

    Args:
        output_data: Dictionary containing the extracted resume data from LLM

    Returns:
        Resume: Validated Resume object if validation succeeds
        None: If validation fails (allows graceful handling upstream)

    Design Notes:
        - Separating validation makes it reusable across different contexts
        - Detailed logging helps diagnose extraction issues quickly
        - Returning None (rather than raising) allows upstream code to handle failures
        - Pydantic does heavy lifting: type checking, constraint validation, etc.

    Edge Cases Handled:
        - Missing required fields → Pydantic catches, logged with field names
        - Invalid data types → Pydantic type coercion or error
        - Invalid enum values → Pydantic validation error
        - Malformed dates → Pydantic date parsing error
        - Nested model errors → Pydantic provides path to error location
    """
    try:
        logger.debug("Validating agent output against Resume model...")

        # ======================================================================
        # STAGE 3.1: Data Validation
        # ======================================================================

        # SUB-STAGE 3.1.1-3.1.3: Pydantic validates everything automatically
        # This single line triggers:
        #   - Type checking for all fields
        #   - Required field validation
        #   - Constraint validation (min/max, regex patterns, etc.)
        #   - Nested model validation (Experience, Education, Skill objects)
        resume = Resume(**output_data)

        # ======================================================================
        # STAGE 3.3: Logging & Reporting (Success Path)
        # ======================================================================

        # SUB-STAGE 3.3.1: Log successful validation with summary
        logger.info(
            f"Resume validation successful. "
            f"Candidate: {resume.full_name}, "
            f"Experience: {len(resume.work_experience)} jobs, "
            f"Skills: {len(resume.skills)} skills"
        )

        # SUB-STAGE 3.3.3: Return validated object
        return resume

    except ValidationError as e:
        # ======================================================================
        # STAGE 3.2: Error Handling (Validation Errors)
        # ======================================================================

        # SUB-STAGE 3.2.1: Catch Pydantic ValidationError
        # SUB-STAGE 3.2.2: Log detailed validation errors
        logger.error(
            f"Resume validation failed. Output does not match Resume model schema. "
            f"Errors: {e.errors()}"
        )
        # SUB-STAGE 3.2.4: Return None for graceful failure
        return None

    except Exception as e:
        # ======================================================================
        # STAGE 3.2: Error Handling (Unexpected Errors)
        # ======================================================================

        # SUB-STAGE 3.2.3: Catch unexpected exceptions
        logger.error(f"Unexpected error during resume validation: {e}", exc_info=True)
        # SUB-STAGE 3.2.4: Return None for graceful failure
        return None


# ==============================================================================
# BLOCK 4: UTILITY FUNCTIONS
# ==============================================================================
# This block provides helper functions for debugging, monitoring, and testing.

# ------------------------------------------------------------------------------
# Stage 4.1: Agent Information
# ------------------------------------------------------------------------------
# This stage provides metadata about the agent for debugging and monitoring.


def get_agent_info() -> dict:
    """
    Get information about this agent for debugging or monitoring.

    STAGE: 4.1 - Agent Information
    PURPOSE: Provide structured metadata about the agent

    SUB-STAGES:
    -----------
    4.1.1: Retrieve Agent Configuration
        - Loads current agent configuration (from YAML or defaults)
        - Extracts key metadata fields

    4.1.2: Format Agent Information Dictionary
        - Creates structured dictionary with:
            - Agent name (human-readable identifier)
            - Role (from configuration)
            - LLM model (which model the agent uses)
            - Tools (list of tools available to agent)
            - Output model (what data structure it produces)

    4.1.3: Return Structured Information
        - Returns dictionary ready for logging or display

    Returns:
        dict: Dictionary with agent metadata:
            - name: Human-readable agent name
            - role: Agent's role from configuration
            - llm: LLM model identifier
            - tools: List of tool names
            - output_model: Name of output Pydantic model

    Example:
        >>> info = get_agent_info()
        >>> print(info["name"])
        'Resume Content Extractor'
        >>> print(info["tools"])
        ['parse_resume']

    Use Cases:
        - Debugging: Understand agent configuration at runtime
        - Monitoring: Track which agents are being used
        - Documentation: Auto-generate agent capability lists
    """
    # SUB-STAGE 4.1.1: Retrieve Agent Configuration
    config = _load_agent_config()

    # SUB-STAGE 4.1.2: Format Agent Information Dictionary
    # SUB-STAGE 4.1.3: Return Structured Information
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
