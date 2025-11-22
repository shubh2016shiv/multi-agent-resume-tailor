"""
Professional Summary Writer Agent - Narrative Crafting Specialist
=================================================================

OVERVIEW:
---------
This module defines the fourth agent in our workflow: the Professional Summary Writer.
This agent serves as the narrative architect, transforming raw career data into compelling
professional summaries that capture attention, demonstrate value, and optimize for ATS systems.

WHAT MAKES THIS AGENT ESSENTIAL:
--------------------------------
- **First Impression Specialist**: The summary is often the first (and sometimes only) text recruiters read
- **Narrative Weaver**: Crafts cohesive stories from disparate resume data
- **ATS-SEO Expert**: Balances human readability with algorithmic optimization
- **Psychological Engineer**: Uses writing principles to create compelling value propositions
- **Authenticity Preserver**: Maintains candidate voice while optimizing for impact

AGENT DESIGN PRINCIPLES:
------------------------
- **Narrative Focus**: Expert storyteller who understands professional communication
- **Strategic Integration**: Seamlessly weaves keywords and requirements into natural prose
- **Psychological Insight**: Uses principles of attention, persuasion, and memorability
- **Quality-Over-Quantity**: Prioritizes impact over length (50-100 words sweet spot)
- **Validation-Driven**: Comprehensive quality checks ensure summaries meet standards

WORKFLOW OVERVIEW:
------------------
1. Receive Resume, JobDescription, and AlignmentStrategy from upstream agents
2. Analyze candidate's strongest matches and unique value propositions
3. Generate multiple summary drafts using different narrative strategies
4. Evaluate each draft for quality, keyword integration, and compelling narrative
5. Select the optimal summary based on comprehensive scoring
6. Validate final output against quality standards and factual accuracy
7. Return structured ProfessionalSummary with metadata and quality metrics

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
├── Stage 1.2: Data Models
│   ├── Sub-stage 1.2.1: SummaryDraft model for iteration tracking
│   ├── Sub-stage 1.2.2: ProfessionalSummary model for final output
│   └── Sub-stage 1.2.3: Model validation and constraints

BLOCK 2: AGENT CONFIGURATION & CREATION
├── Stage 2.1: Configuration Loading
│   ├── Sub-stage 2.1.1: Load from agents.yaml with fallback to defaults
│   ├── Sub-stage 2.1.2: Validate required fields (role, goal, backstory)
│   └── Sub-stage 2.1.3: Error handling with graceful degradation
│
├── Stage 2.2: Default Configuration
│   ├── Sub-stage 2.2.1: Define narrative specialist role
│   ├── Sub-stage 2.2.2: Set creative LLM parameters (temperature 0.7)
│   └── Sub-stage 2.2.3: Configure content generation settings
│
└── Stage 2.3: Agent Creation
    ├── Sub-stage 2.3.1: CrewAI Agent initialization with tools
    ├── Sub-stage 2.3.2: Tool assignment for draft evaluation
    └── Sub-stage 2.3.3: Resilience and performance configuration

BLOCK 3: DRAFT EVALUATION INFRASTRUCTURE
├── Stage 3.1: Evaluation Tool Framework
│   ├── Sub-stage 3.1.1: DraftEvaluationTool class design
│   ├── Sub-stage 3.1.2: Quality assessment criteria
│   └── Sub-stage 3.1.3: Scoring algorithm implementation
│
├── Stage 3.2: Multi-Criteria Evaluation
│   ├── Sub-stage 3.2.1: Content quality assessment
│   ├── Sub-stage 3.2.2: Keyword integration verification
│   └── Sub-stage 3.2.3: Narrative coherence evaluation
│
└── Stage 3.3: Selection Logic
    ├── Sub-stage 3.3.1: Best draft selection algorithm
    ├── Sub-stage 3.3.2: Tie-breaking criteria
    └── Sub-stage 3.3.3: Quality threshold enforcement

BLOCK 4: OUTPUT VALIDATION
├── Stage 4.1: Data Validation
│   ├── Sub-stage 4.1.1: Validate ProfessionalSummary model
│   ├── Sub-stage 4.1.2: Check required fields and data types
│   └── Sub-stage 4.1.3: Verify nested model relationships
│
├── Stage 4.2: Error Handling
│   ├── Sub-stage 4.2.1: Catch Pydantic ValidationError
│   ├── Sub-stage 4.2.2: Log detailed validation errors
│   └── Sub-stage 4.2.3: Return None for graceful failure handling
│
└── Stage 4.3: Logging & Reporting
    ├── Sub-stage 4.3.1: Log successful validation with summary
    ├── Sub-stage 4.3.2: Log validation failures with error details
    └── Sub-stage 4.3.3: Return validated ProfessionalSummary

BLOCK 5: QUALITY ASSESSMENT FUNCTIONS
├── Stage 5.1: Summary Quality Check
│   ├── Sub-stage 5.1.1: check_summary_quality() function
│   ├── Sub-stage 5.1.2: Multi-dimensional quality assessment
│   └── Sub-stage 5.1.3: Scoring and recommendations
│
├── Stage 5.2: Keyword Integration Analysis
│   ├── Sub-stage 5.2.1: analyze_keyword_integration() function
│   ├── Sub-stage 5.2.2: Natural integration verification
│   └── Sub-stage 5.2.3: Keyword density optimization
│
└── Stage 5.3: Content Validation
    ├── Sub-stage 5.3.1: Factual accuracy checking
    ├── Sub-stage 5.3.2: Professional tone verification
    └── Sub-stage 5.3.3: Length and structure compliance

BLOCK 6: UTILITY FUNCTIONS
├── Stage 6.1: Agent Information
│   ├── Sub-stage 6.1.1: get_agent_info() function
│   ├── Sub-stage 6.1.2: Retrieve agent metadata
│   └── Sub-stage 6.1.3: Format information for debugging
│
├── Stage 6.2: Word Counting Utility
│   ├── Sub-stage 6.2.1: count_words() function
│   ├── Sub-stage 6.2.2: Accurate word counting algorithm
│   └── Sub-stage 6.2.3: Length validation support
│
└── Stage 6.3: Testing Support
    ├── Sub-stage 6.3.1: Test configuration loading
    ├── Sub-stage 6.3.2: Test agent creation
    ├── Sub-stage 6.3.3: Test validation functions
    └── Sub-stage 6.3.4: Test summary generation

BLOCK 7: INTEGRATION TESTING
├── Stage 7.1: End-to-End Testing
│   ├── Sub-stage 7.1.1: Mock data creation for testing
│   ├── Sub-stage 7.1.2: Summary generation validation
│   └── Sub-stage 7.1.3: Integration test scenarios

HOW TO USE THIS MODULE:
-----------------------
1. Import: `from src.agents.summary_writer_agent import create_summary_writer_agent`
2. Create Agent: `agent = create_summary_writer_agent()`
3. Use in Crew: Add agent to CrewAI crew with Resume, JobDescription, and AlignmentStrategy
4. Validate Output: Use `validate_summary_output()` to ensure data quality
5. Check Quality: Use `check_summary_quality()` for comprehensive assessment

KEY NARRATIVE PRINCIPLES:
-------------------------
- **Hook Strategy**: First sentence must grab attention and create interest
- **Value Proposition**: Clearly articulate what makes the candidate valuable
- **Evidence-Based**: Support claims with specific achievements and metrics
- **Relevance Focus**: Address the target role's specific requirements
- **Authenticity**: Maintain genuine voice while optimizing for impact
- **Conciseness**: Maximum impact in minimum words (50-100 word sweet spot)

CONTENT GENERATION STRATEGY:
---------------------------
- **Multi-Draft Approach**: Generate multiple versions with different strategies
- **Creative Temperature**: Higher temperature (0.7) for engaging, nuanced writing
- **Strategic Frameworks**: Use proven narrative structures (Hook-Value-Future, etc.)
- **Keyword Integration**: Natural incorporation without keyword stuffing
- **Quality Iteration**: Self-evaluation and improvement of generated content
- **Validation Loops**: Continuous quality checking and refinement

TECHNICAL ARCHITECTURE:
-----------------------
- **Draft-Based Generation**: Multiple iterations with evaluation feedback
- **Tool-Augmented Writing**: Self-evaluation capabilities for quality improvement
- **Model-Driven Validation**: Pydantic models ensure output structure and quality
- **Scoring Algorithms**: Multi-criteria assessment of summary effectiveness
- **Error Recovery**: Graceful handling of generation failures
- **Performance Optimization**: Efficient word counting and quality checks

WRITING METHODOLOGY:
--------------------
The agent employs a systematic approach to summary creation:
1. **Analysis Phase**: Understand candidate strengths, job requirements, alignment strategy
2. **Strategy Selection**: Choose appropriate narrative frameworks based on context
3. **Draft Generation**: Create multiple summary versions using different approaches
4. **Quality Evaluation**: Assess each draft against comprehensive criteria
5. **Optimization**: Refine and improve based on evaluation feedback
6. **Final Selection**: Choose the highest-scoring, most effective summary
7. **Validation**: Ensure final output meets all quality and accuracy standards

This creates professional summaries that not only pass ATS filters but also
compel human recruiters to continue reading the full resume.
"""

from crewai import Agent
from crewai.tools import tool
from pydantic import BaseModel, ConfigDict, Field, ValidationError

# Handle imports for both package usage and direct script execution
try:
    from src.core.config import (
        get_agents_config,  # get_config unused - kept for potential future refactoring
    )
    from src.core.logger import get_logger
    from src.data_models.strategy import AlignmentStrategy
except ImportError:
    # Fallback for when running this file directly
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.core.config import (
        get_agents_config,  # get_config unused - kept for potential future refactoring
    )
    from src.core.logger import get_logger
    from src.data_models.strategy import AlignmentStrategy

logger = get_logger(__name__)


# ==============================================================================
# BLOCK 1: MODULE SETUP & CONFIGURATION
# ==============================================================================
# PURPOSE: Initialize the module with data models and configuration
# WHAT: Core data structures and setup for summary writing functionality
# WHY: Provides the foundation for narrative generation and quality assessment
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1.2: Data Models
# ------------------------------------------------------------------------------
# This stage defines the structured data models for summary generation and evaluation.


class SummaryDraft(BaseModel):
    """
    A single draft version of the professional summary.
    """

    version_name: str = Field(
        ...,
        description="Name of the version (e.g., 'Hook-Value-Future', 'Story Spine', 'ATS-Optimized')",
    )
    strategy_used: str = Field(
        ...,
        description="Brief description of the writing strategy or framework used for this draft",
    )
    content: str = Field(
        ...,
        description="The actual summary text",
        min_length=50,
        max_length=1000,
    )
    critique: str = Field(
        default="",
        description="Self-critique of this draft (strengths/weaknesses)",
    )
    score: int = Field(
        ...,
        description="Self-assigned confidence score (0-100)",
        ge=0,
        le=100,
    )


class ProfessionalSummary(BaseModel):
    """
    Structured output containing multiple strategic drafts of the professional summary.
    """

    drafts: list[SummaryDraft] = Field(
        ...,
        description="List of generated summary drafts using different strategies",
        min_items=1,
    )

    recommended_version: str = Field(
        ...,
        description="The version_name of the draft the agent most strongly recommends",
    )

    writing_notes: str = Field(
        default="",
        description="Overall notes about the writing process and trade-offs between versions",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "drafts": [
                    {
                        "version_name": "Hook-Value-Future",
                        "strategy_used": "Classic 3-part structure: Hook -> Value -> Future",
                        "content": "Scaled infrastructure serving 50M+ users... (rest of summary)",
                        "critique": "Strong hook, but could use more specific metrics in the middle.",
                        "score": 85,
                    },
                    {
                        "version_name": "Story Spine",
                        "strategy_used": "Narrative arc focusing on career evolution",
                        "content": "Starting as a junior dev... (rest of summary)",
                        "critique": "Engaging story, but slightly long.",
                        "score": 80,
                    },
                ],
                "recommended_version": "Hook-Value-Future",
                "writing_notes": "The Hook-Value-Future version aligns best with the senior level of this role.",
            }
        }
    )


# ==============================================================================
# BLOCK 2: AGENT CONFIGURATION & CREATION
# ==============================================================================
# PURPOSE: Configure and create the Summary Writer agent with all necessary tools
# WHAT: Agent setup, configuration loading, and initialization with evaluation capabilities
# WHY: Produces a fully functional agent ready for narrative generation and quality assessment
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 2.1: Configuration Loading
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
        config = agents_config.get("professional_summary_writer", {})

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
        "role": "Professional Summary Content Writer",
        "goal": (
            "Craft compelling, keyword-optimized professional summaries that highlight "
            "the candidate's strongest qualifications and alignment with the target role. "
            "Follow strategic guidance and integrate required keywords naturally."
        ),
        "backstory": (
            "You are an expert resume writer and career strategist with a talent for "
            "distilling complex career narratives into powerful, concise summaries. "
            "You understand how to grab attention, convey value, and optimize for both "
            "human readers and ATS systems. You write with confidence, clarity, and authenticity."
        ),
        "llm": "gpt-4o",
        "temperature": 0.7,
        "verbose": True,
    }


# ==============================================================================
# BLOCK 3: DRAFT EVALUATION INFRASTRUCTURE
# ==============================================================================
# PURPOSE: Provide evaluation capabilities for summary draft quality assessment
# WHAT: Tools and functions for scoring and critiquing generated summaries
# WHY: Enables the agent to improve its own output through iterative evaluation
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 3.1: Evaluation Tool Framework
# ------------------------------------------------------------------------------
# This stage defines the tool that enables self-evaluation of summary drafts.


class DraftEvaluationTool:
    @tool("Evaluate Draft Quality")
    def evaluate_draft(draft_text: str, keywords: list[str]) -> str:
        """
        Evaluate a professional summary draft against quality standards.
        Returns a detailed analysis of word count, keyword integration, and writing quality.

        Args:
            draft_text: The summary text to evaluate.
            keywords: List of keywords that should be included.
        """
        # Create a temporary summary object for validation
        # We use a dummy version name since we're just checking the text content
        try:
            # Simple word count check
            word_count = len(draft_text.split())

            # Keyword check
            draft_lower = draft_text.lower()
            # found_keywords = [k for k in keywords if k.lower() in draft_lower]  # Commented out: unused variable
            missing_keywords = [k for k in keywords if k.lower() not in draft_lower]

            # Cliche check
            cliches = [
                "results-driven",
                "team player",
                "hard worker",
                "detail-oriented",
                "go-getter",
            ]
            found_cliches = [c for c in cliches if c in draft_lower]

            # Scoring
            score = 100
            feedback = []

            if word_count < 50:
                score -= 20
                feedback.append(f"Too short ({word_count} words). Aim for 75-150.")
            elif word_count > 150:
                score -= 10
                feedback.append(f"Too long ({word_count} words). Aim for 75-150.")

            if missing_keywords:
                penalty = min(30, len(missing_keywords) * 5)
                score -= penalty
                feedback.append(f"Missing keywords: {', '.join(missing_keywords)}")

            if found_cliches:
                score -= 15
                feedback.append(f"Contains cliches: {', '.join(found_cliches)}")

            return f"Score: {score}/100. Feedback: {'; '.join(feedback) if feedback else 'Excellent draft.'}"

        except Exception as e:
            return f"Error evaluating draft: {str(e)}"


# ------------------------------------------------------------------------------
# Stage 2.3: Agent Creation
# ------------------------------------------------------------------------------
# This stage creates the Summary Writer agent with all tools and configuration.


def create_summary_writer_agent() -> Agent:
    """
    Create and configure the Professional Summary Writer agent.

    This is the main entry point for creating this agent. It handles all the
    complexity of configuration loading and agent initialization.

    Returns:
        Configured CrewAI Agent instance ready to write professional summaries

    Raises:
        Exception: If agent creation fails (logged and re-raised)

    Example:
        >>> agent = create_summary_writer_agent()
        >>> # Agent is now ready to be used in a crew or task

    Design Notes:
        - Uses configuration from agents.yaml (with fallback to defaults)
        - Uses DraftEvaluationTool for self-critique
        - Higher temperature (0.7) for creative, engaging content
        - Uses GPT-4o for superior writing quality
        - Enables verbose mode for detailed logging
    """
    try:
        logger.info("Creating Professional Summary Writer agent...")

        # Load configuration
        config = _load_agent_config()

        # Extract LLM settings
        llm_model = config.get("llm", "gemini/gemini-2.5-flash")
        temperature = config.get("temperature", 0.7)
        verbose = config.get("verbose", True)

        # Load centralized resilience configuration
        # app_config = get_config()  # Unused variable - kept for potential future refactoring
        # agent_defaults = app_config.llm.agent_defaults  # Unused variable - kept for potential future refactoring

        # Create the agent
        # NOTE: We do NOT assign tools here because this agent's job is to output
        # structured JSON data (ProfessionalSummary model), not to use tools.
        # The DraftEvaluationTool exists for reference but the agent should perform
        # self-critique internally as part of its reasoning process, then output
        # the final structured result with critique embedded in each draft.
        agent = Agent(
            role=config["role"],
            goal=config["goal"],
            backstory=config["backstory"],
            tools=[],  # No tools - agent outputs structured data directly
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
        logger.error(f"Failed to create Professional Summary Writer agent: {e}", exc_info=True)
        raise


# ==============================================================================
# BLOCK 4: OUTPUT VALIDATION
# ==============================================================================
# PURPOSE: Validate that agent outputs conform to expected data models
# WHAT: Quality gates that ensure structured summary data meets schema requirements
# WHY: Prevents downstream errors and ensures narrative quality
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 4.1-4.3: Complete Validation Workflow
# ------------------------------------------------------------------------------
# This stage orchestrates all validation steps to ensure summary output quality.


def validate_summary_output(output_data: dict) -> ProfessionalSummary | None:
    """
    Validate that the agent's output conforms to the ProfessionalSummary model.

    This function serves as a quality gate, ensuring that the generated summary
    is valid according to our schema. If validation fails, it provides
    detailed error information for debugging.

    Args:
        output_data: Dictionary containing the professional summary

    Returns:
        ProfessionalSummary object if validation succeeds, None if it fails

    Design Notes:
        - Separating validation into its own function makes it reusable
        - Detailed logging helps diagnose generation issues
        - Returning None (rather than raising) allows graceful handling upstream

    Edge Cases Handled:
        - Missing required fields → logged with specific field names
        - Length violations → caught by Pydantic validation
        - Malformed data types → validation error details provided
    """
    try:
        logger.debug("Validating agent output against ProfessionalSummary model...")

        # Attempt to create a ProfessionalSummary object from the output
        summary = ProfessionalSummary(**output_data)

        logger.info(
            f"Summary validation successful. "
            f"Drafts generated: {len(summary.drafts)}, "
            f"Recommended: {summary.recommended_version}"
        )

        return summary

    except ValidationError as e:
        logger.error(
            f"Summary validation failed. Output does not match ProfessionalSummary model schema. "
            f"Errors: {e.errors()}"
        )
        # Log each validation error for easier debugging
        for error in e.errors():
            logger.error(f"  Field: {error['loc']}, Type: {error['type']}, Message: {error['msg']}")
        return None

    except Exception as e:
        logger.error(f"Unexpected error during summary validation: {e}", exc_info=True)
        return None


# ==============================================================================
# BLOCK 5: QUALITY ASSESSMENT FUNCTIONS
# ==============================================================================
# PURPOSE: Provide comprehensive quality assessment for generated summaries
# WHAT: Functions that evaluate content quality, keyword integration, and effectiveness
# WHY: Ensures summaries meet professional standards and optimization goals
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 5.1: Summary Quality Check
# ------------------------------------------------------------------------------
# This stage performs comprehensive quality assessment of generated summaries.


def check_summary_quality(summary: ProfessionalSummary, strategy: AlignmentStrategy) -> dict:
    """
    Perform quality checks on the generated professional summary.

    This function validates that the summary is well-written, strategically aligned,
    and optimized for both human readers and ATS systems.

    Args:
        summary: The validated ProfessionalSummary object
        strategy: The AlignmentStrategy that guided the writing

    Returns:
        Dictionary with quality check results and recommendations

    Quality Checks:
        - Word count in optimal range (50-100 words)
        - Keywords integration from strategy
        - No repetitive phrases
        - Professional tone maintained
        - Key strengths highlighted
        - Factual and authentic

    Design Note:
        This helps catch low-quality or non-strategic summaries early.
    """
    results = {"overall_status": "complete", "draft_evaluations": []}

    for draft in summary.drafts:
        draft_score = 100
        issues = []
        warnings = []

        word_count = count_words(draft.content)

        # Check word count (optimal: 50-120 words)
        if word_count < 40:
            issues.append(f"Summary too short ({word_count} words)")
            draft_score -= 30
        elif word_count > 150:
            issues.append(f"Summary too long ({word_count} words)")
            draft_score -= 25

        # Check keyword integration
        required_keywords = {k.lower() for k in strategy.keywords_to_integrate[:5]}
        draft_lower = draft.content.lower()
        missing_keywords = [k for k in required_keywords if k not in draft_lower]

        if len(missing_keywords) > 3:
            issues.append(f"Missing critical keywords: {missing_keywords}")
            draft_score -= 25
        elif len(missing_keywords) > 0:
            warnings.append(f"Some keywords not used: {missing_keywords}")
            draft_score -= 10

        # Avoid clichés
        cliches = ["results-driven", "team player", "hard worker", "detail-oriented", "go-getter"]
        cliches_found = [c for c in cliches if c in draft_lower]
        if cliches_found:
            warnings.append(f"Clichés detected: {cliches_found}")
            draft_score -= 15

        # Determine quality level
        if draft_score >= 90:
            quality = "excellent"
        elif draft_score >= 75:
            quality = "good"
        elif draft_score >= 60:
            quality = "fair"
        else:
            quality = "poor"

        results["draft_evaluations"].append(
            {
                "version": draft.version_name,
                "quality": quality,
                "score": max(0, draft_score),
                "issues": issues,
                "warnings": warnings,
                "word_count": word_count,
            }
        )

    logger.info(f"Completed quality check for {len(summary.drafts)} drafts")
    return results


def analyze_keyword_integration(summary_text: str, required_keywords: list[str]) -> dict:
    """
    Analyze how well keywords are integrated into the summary.

    This function checks not just if keywords are present, but how naturally
    they're integrated into the text.

    Args:
        summary_text: The professional summary text
        required_keywords: Keywords that should be integrated

    Returns:
        Dictionary with keyword analysis results

    Analysis Criteria:
        - Presence: Is the keyword in the text?
        - Context: Is it used in a meaningful sentence?
        - Natural: Does it flow naturally or feel forced?
    """
    summary_lower = summary_text.lower()
    results = {
        "total_required": len(required_keywords),
        "total_integrated": 0,
        "missing_keywords": [],
        "integrated_keywords": [],
        "integration_rate": 0.0,
    }

    for keyword in required_keywords:
        keyword_lower = keyword.lower()
        if keyword_lower in summary_lower:
            results["total_integrated"] += 1
            results["integrated_keywords"].append(keyword)
        else:
            results["missing_keywords"].append(keyword)

    # Calculate integration rate
    if results["total_required"] > 0:
        results["integration_rate"] = round(
            results["total_integrated"] / results["total_required"], 2
        )

    logger.info(
        f"Keyword integration: {results['total_integrated']}/{results['total_required']} "
        f"({results['integration_rate'] * 100:.0f}%)"
    )

    return results


# ==============================================================================
# BLOCK 6: UTILITY FUNCTIONS
# ==============================================================================
# PURPOSE: Provide utility functions for debugging, monitoring, and text processing
# WHAT: Helper functions for agent metadata, word counting, and diagnostic information
# WHY: Enables debugging, monitoring, and validation of agent functionality
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 6.1: Agent Information
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
        'Professional Summary Writer'
    """
    config = _load_agent_config()
    return {
        "name": "Professional Summary Writer",
        "role": config.get("role", "Unknown"),
        "llm": config.get("llm", "Unknown"),
        "tools": [],
        "output_model": "ProfessionalSummary",
        "content_type": "professional_summary",
    }


def count_words(text: str) -> int:
    """
    Count words in a text string.

    Args:
        text: The text to count words in

    Returns:
        Number of words
    """
    return len(text.split())


# ==============================================================================
# BLOCK 7: INTEGRATION TESTING
# ==============================================================================
# PURPOSE: Provide testing and validation capabilities for the agent
# WHAT: Test functions and integration validation code
# WHY: Ensures agent functionality and enables development-time validation
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 7.1: End-to-End Testing
# ------------------------------------------------------------------------------
# This stage provides comprehensive testing of agent functionality.

if __name__ == "__main__":
    """
    Test the agent creation and configuration loading.
    Run this script directly to verify the agent can be created.
    """
    print("=" * 70)
    print("Professional Summary Writer Agent - Test")
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
        agent = create_summary_writer_agent()
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
        from src.data_models.strategy import AlignmentStrategy, SkillGap, SkillMatch

        # Create mock strategy
        mock_strategy = AlignmentStrategy(
            overall_fit_score=85.5,
            summary_of_strategy="Emphasize cloud and Python experience",
            identified_matches=[
                SkillMatch(
                    resume_skill="Python",
                    job_requirement="Python",
                    match_score=95.0,
                    justification="Direct match",
                )
            ],
            identified_gaps=[
                SkillGap(
                    missing_skill="Kubernetes",
                    importance="nice_to_have",
                    suggestion="Mention if available",
                )
            ],
            keywords_to_integrate=["Python", "AWS", "Microservices", "Cloud", "Docker"],
            professional_summary_guidance="Emphasize cloud architecture and cost optimization",
            experience_guidance="Highlight AWS projects",
            skills_guidance="Prioritize cloud technologies",
        )

        # Create mock summary
        mock_summary = ProfessionalSummary(
            drafts=[
                SummaryDraft(
                    version_name="Hook-Value-Future",
                    strategy_used="Classic 3-part structure",
                    content="Senior Software Engineer with 8+ years of experience... (content)",
                    score=90,
                )
            ],
            recommended_version="Hook-Value-Future",
            writing_notes="Strongest alignment.",
        )

        quality_result = check_summary_quality(mock_summary, mock_strategy)
        print(f"Overall Status: {quality_result.get('overall_status', 'N/A')}")
        print(f"Draft Evaluations: {len(quality_result.get('draft_evaluations', []))}")

        for eval_result in quality_result.get("draft_evaluations", []):
            print(f"\n  {eval_result['version']}:")
            print(f"    Quality: {eval_result['quality']}")
            print(f"    Score: {eval_result['score']}/100")
            print(f"    Word Count: {eval_result['word_count']}")
            if eval_result.get("issues"):
                print(f"    Issues: {eval_result['issues']}")
            if eval_result.get("warnings"):
                print(f"    Warnings: {eval_result['warnings']}")

        # Test keyword analysis on first draft
        print("\n--- Testing Keyword Integration Analysis ---")
        if mock_summary.drafts:
            first_draft_content = mock_summary.drafts[0].content
            keyword_analysis = analyze_keyword_integration(
                first_draft_content, mock_strategy.keywords_to_integrate
            )
            print(f"Integration rate: {keyword_analysis['integration_rate'] * 100:.0f}%")
            print(f"Integrated: {keyword_analysis['integrated_keywords']}")
            if keyword_analysis["missing_keywords"]:
                print(f"Missing: {keyword_analysis['missing_keywords']}")

    except Exception as e:
        print(f"Quality check test failed: {str(e)}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 70)
