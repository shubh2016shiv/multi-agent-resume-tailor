"""
Gap Analysis Specialist Agent - Strategic Alignment Engine
==========================================================

OVERVIEW:
---------
This module defines the third agent in our workflow: the Gap Analysis Specialist.
This agent serves as the strategic brain of the system, comparing candidate profiles
against job requirements to identify matches, gaps, and optimization opportunities.

WHAT MAKES THIS AGENT STRATEGIC:
--------------------------------
- **Decision Engine**: Determines what aspects of a resume to emphasize or modify
- **Gap Intelligence**: Identifies missing skills with priority rankings
- **Match Optimization**: Finds best alignment between candidate and role
- **Strategy Generation**: Creates actionable plans for resume tailoring
- **Confidence Scoring**: Quantifies alignment quality with statistical rigor

AGENT DESIGN PRINCIPLES:
------------------------
- **Comparative Analysis**: Expert at side-by-side resume vs job evaluation
- **Strategic Thinking**: Goes beyond simple matching to understand implications
- **Confidence-Based**: Provides statistical confidence in gap identifications
- **Actionable Insights**: Generates specific, implementable recommendations
- **Context Aware**: Considers experience levels, domains, and career trajectories

WORKFLOW OVERVIEW:
------------------
1. Receive structured Resume and JobDescription data from upstream agents
2. Perform multi-dimensional comparison across skills, experience, and requirements
3. Calculate match scores and confidence levels for each comparison
4. Identify critical gaps (must-have skills missing) with high priority
5. Identify enhancement opportunities (nice-to-have skills) with medium priority
6. Generate keyword optimization suggestions for ATS compatibility
7. Create comprehensive AlignmentStrategy with prioritized action items
8. Return structured strategy object for downstream content generation

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
├── Stage 1.2: Configuration Loading
│   ├── Sub-stage 1.2.1: Load agent config from agents.yaml
│   ├── Sub-stage 1.2.2: Extract LLM settings and parameters
│   └── Sub-stage 1.2.3: Load resilience configuration

BLOCK 2: AGENT CREATION
├── Stage 2.1: Configuration Retrieval
│   ├── Sub-stage 2.1.1: Load gap_analysis_specialist configuration
│   ├── Sub-stage 2.1.2: Extract specialized settings for comparative analysis
│   └── Sub-stage 2.1.3: Apply resilience parameters
│
├── Stage 2.2: Agent Initialization
│   ├── Sub-stage 2.2.1: Set agent role, goal, and backstory for gap analysis
│   ├── Sub-stage 2.2.2: Configure agent behavior (no delegation, verbose logging)
│   └── Sub-stage 2.2.3: Initialize CrewAI Agent object
│
└── Stage 2.3: Resilience Configuration
    ├── Sub-stage 2.3.1: Set retry limits and rate limiting
    ├── Sub-stage 2.3.2: Configure execution timeouts
    └── Sub-stage 2.3.3: Enable context window management

BLOCK 3: OUTPUT VALIDATION
├── Stage 3.1: Data Validation
│   ├── Sub-stage 3.1.1: Parse output into AlignmentStrategy model
│   ├── Sub-stage 3.1.2: Validate required fields and score ranges
│   └── Sub-stage 3.1.3: Check nested model relationships
│
├── Stage 3.2: Error Handling
│   ├── Sub-stage 3.2.1: Catch Pydantic ValidationError
│   ├── Sub-stage 3.2.2: Log detailed validation errors
│   └── Sub-stage 3.2.3: Return None for graceful failure handling
│
└── Stage 3.3: Logging & Reporting
    ├── Sub-stage 3.3.1: Log successful validation with summary statistics
    ├── Sub-stage 3.3.2: Log validation failures with error details
    └── Sub-stage 3.3.3: Return validated AlignmentStrategy object

BLOCK 4: ANALYSIS HELPER FUNCTIONS
├── Stage 4.1: Skill Normalization
│   ├── Sub-stage 4.1.1: normalize_skill() function
│   ├── Sub-stage 4.1.2: Standardize skill naming conventions
│   └── Sub-stage 4.1.3: Handle common synonyms and variations
│
├── Stage 4.2: Data Extraction
│   ├── Sub-stage 4.2.1: extract_resume_skills() function
│   ├── Sub-stage 4.2.2: extract_job_requirements() function
│   └── Sub-stage 4.2.3: Map requirements to importance levels
│
└── Stage 4.3: Analysis Utilities
    ├── Sub-stage 4.3.1: Helper functions for skill comparison
    ├── Sub-stage 4.3.2: Scoring algorithms for match confidence
    └── Sub-stage 4.3.3: Gap prioritization logic

BLOCK 5: ANALYSIS QUALITY CHECKS
├── Stage 5.1: Quality Assessment
│   ├── Sub-stage 5.1.1: check_analysis_quality() function
│   ├── Sub-stage 5.1.2: Validate strategy completeness
│   └── Sub-stage 5.1.3: Check logical consistency of results
│
├── Stage 5.2: Coverage Statistics
│   ├── Sub-stage 5.2.1: calculate_coverage_stats() function
│   ├── Sub-stage 5.2.2: Compute match percentages and ratios
│   └── Sub-stage 5.2.3: Generate coverage metrics
│
└── Stage 5.3: Quality Reporting
    ├── Sub-stage 5.3.1: Structure quality check results
    ├── Sub-stage 5.3.2: Log issues and recommendations
    └── Sub-stage 5.3.3: Return comprehensive quality report

BLOCK 6: UTILITY FUNCTIONS
├── Stage 6.1: Agent Information
│   ├── Sub-stage 6.1.1: get_agent_info() function
│   ├── Sub-stage 6.1.2: Retrieve agent metadata
│   └── Sub-stage 6.1.3: Format information for debugging
│
└── Stage 6.2: Testing Support
    ├── Sub-stage 6.2.1: Test configuration loading
    ├── Sub-stage 6.2.2: Test agent creation
    ├── Sub-stage 6.2.3: Test validation functions
    └── Sub-stage 6.2.4: Test helper functions

BLOCK 7: INTEGRATION TESTING
├── Stage 7.1: End-to-End Testing
│   ├── Sub-stage 7.1.1: Mock data creation for testing
│   ├── Sub-stage 7.1.2: Gap analysis validation
│   └── Sub-stage 7.1.3: Integration test scenarios

HOW TO USE THIS MODULE:
-----------------------
1. Import: `from src.agents.gap_analysis_agent import create_gap_analysis_agent`
2. Create Agent: `agent = create_gap_analysis_agent()`
3. Use in Crew: Add agent to CrewAI crew with Resume and JobDescription data
4. Validate Output: Use `validate_analysis_output()` to ensure data quality
5. Check Quality: Use `check_analysis_quality()` for analysis validation
6. Get Stats: Use `calculate_coverage_stats()` for coverage metrics

KEY ANALYSIS CAPABILITIES:
-------------------------
- **Multi-Dimensional Matching**: Skills, experience, domain, certifications
- **Confidence Scoring**: Statistical confidence in gap identifications
- **Priority Ranking**: Must-have vs should-have vs nice-to-have gaps
- **Strategic Recommendations**: Actionable plans for resume optimization
- **Keyword Intelligence**: ATS keyword optimization suggestions
- **Experience Alignment**: Years and level compatibility assessment

STRATEGIC INSIGHTS PROVIDED:
---------------------------
- **Gap Prioritization**: Which missing skills to address first
- **Match Optimization**: How to best position existing skills
- **Career Trajectory**: Alignment between candidate path and job requirements
- **Competitive Positioning**: How candidate stacks up against job needs
- **Optimization Roadmap**: Step-by-step plan for resume improvement

TECHNICAL ARCHITECTURE:
-----------------------
- **Comparative Algorithms**: Sophisticated matching and gap detection
- **Confidence Modeling**: Statistical approaches to gap certainty
- **Strategy Generation**: AI-driven recommendation engine
- **Validation Layers**: Multiple quality checks and error handling
- **Observable Operations**: Comprehensive logging and monitoring
- **Type Safety**: Full Pydantic model validation

ANALYSIS METHODOLOGY:
--------------------
The agent employs a systematic approach to gap analysis:
1. **Normalization**: Standardize skill names and requirement formats
2. **Multi-Level Comparison**: Skills, experience, domain, certifications
3. **Confidence Calculation**: Statistical measures of match quality
4. **Gap Classification**: Priority-based categorization of missing elements
5. **Strategy Synthesis**: Coherent action plan generation
6. **Quality Assurance**: Built-in validation of analysis results

The result is a comprehensive strategic assessment that guides all downstream
content optimization and ensures maximum alignment between candidate and role.
"""

from crewai import Agent
from pydantic import ValidationError

# Handle imports for both package usage and direct script execution
try:
    from src.core.config import get_agents_config, get_config
    from src.core.logger import get_logger
    from src.data_models.job import JobDescription
    from src.data_models.resume import Resume
    from src.data_models.strategy import AlignmentStrategy
except ImportError:
    # Fallback for when running this file directly
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.core.config import get_agents_config, get_config
    from src.core.logger import get_logger
    from src.data_models.job import JobDescription
    from src.data_models.resume import Resume
    from src.data_models.strategy import AlignmentStrategy

logger = get_logger(__name__)


# ==============================================================================
# BLOCK 1: MODULE SETUP & CONFIGURATION
# ==============================================================================
# PURPOSE: Initialize the module with imports and configuration
# WHAT: Global setup and agent configuration loading
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
        config = agents_config.get("gap_analysis_specialist", {})

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
        "role": "Gap Analysis Specialist",
        "goal": (
            "Compare candidate profiles against job requirements to identify "
            "matches, gaps, and optimization opportunities with strategic insights."
        ),
        "backstory": (
            "You are a strategic career advisor and recruitment expert. You excel at "
            "identifying alignment between candidates and roles, recognizing both "
            "obvious matches and subtle gaps that could impact success."
        ),
        "llm": "gemini/gemini-2.5-flash",
        "temperature": 0.3,
        "verbose": True,
    }


def create_gap_analysis_agent() -> Agent:
    """
    Create the Gap Analysis Specialist agent.

    Returns:
        Agent: A configured CrewAI agent ready for execution.
    """
    try:
        config = _load_agent_config()

        # Load centralized resilience configuration
        app_config = get_config()
        agent_defaults = app_config.llm.agent_defaults

        logger.info("Creating Gap Analysis Specialist agent...")

        agent = Agent(
            role=config["role"],
            goal=config["goal"],
            backstory=config["backstory"],
            allow_delegation=False,
            verbose=True,
            # Resilience Parameters (Layer 1: CrewAI Native)
            max_retry_limit=agent_defaults.max_retry_limit,
            max_rpm=agent_defaults.max_rpm,
            max_iter=agent_defaults.max_iter,
            max_execution_time=agent_defaults.max_execution_time,
            respect_context_window=agent_defaults.respect_context_window,
        )

        logger.info(
            f"Gap Analysis Specialist agent created successfully with resilience: "
            f"max_retry={agent_defaults.max_retry_limit}, max_rpm={agent_defaults.max_rpm}"
        )

        return agent

    except Exception as e:
        logger.error(f"Failed to create Gap Analysis Specialist agent: {e}", exc_info=True)
        raise


# ==============================================================================
# BLOCK 3: OUTPUT VALIDATION
# ==============================================================================
# PURPOSE: Validate that agent outputs conform to expected data models
# WHAT: Quality gates that ensure structured analysis data meets schema requirements
# WHY: Prevents downstream errors and ensures strategic data consistency
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 3.1-3.3: Complete Validation Workflow
# ------------------------------------------------------------------------------
# This stage orchestrates all validation steps to ensure analysis output quality.


def validate_analysis_output(output_data: dict) -> AlignmentStrategy | None:
    """
    Validate that the agent's output conforms to the AlignmentStrategy model.

    This function serves as a quality gate, ensuring that the analysis data
    is valid according to our schema. If validation fails, it provides
    detailed error information for debugging.

    Args:
        output_data: Dictionary containing the gap analysis results

    Returns:
        AlignmentStrategy object if validation succeeds, None if it fails

    Design Notes:
        - Separating validation into its own function makes it reusable
        - Detailed logging helps diagnose analysis issues
        - Returning None (rather than raising) allows graceful handling upstream

    Edge Cases Handled:
        - Missing required fields → logged with specific field names
        - Invalid score ranges → caught by Pydantic validation
        - Malformed data types → validation error details provided
    """
    try:
        logger.debug("Validating agent output against AlignmentStrategy model...")

        # Attempt to create an AlignmentStrategy object from the output
        strategy = AlignmentStrategy(**output_data)

        logger.info(
            f"Analysis validation successful. "
            f"Fit score: {strategy.overall_fit_score:.1f}%, "
            f"Skill matches: {len(strategy.identified_matches)}, "
            f"Skill gaps: {len(strategy.identified_gaps)}"
        )

        return strategy

    except ValidationError as e:
        logger.error(
            f"Analysis validation failed. Output does not match AlignmentStrategy model schema. "
            f"Errors: {e.errors()}"
        )
        # Log each validation error for easier debugging
        for error in e.errors():
            logger.error(f"  Field: {error['loc']}, Type: {error['type']}, Message: {error['msg']}")
        return None

    except Exception as e:
        logger.error(f"Unexpected error during analysis validation: {e}", exc_info=True)
        return None


# ==============================================================================
# BLOCK 4: ANALYSIS HELPER FUNCTIONS
# ==============================================================================
# PURPOSE: Provide utility functions for gap analysis and skill comparison
# WHAT: Helper functions for data normalization, extraction, and analysis
# WHY: Enables accurate skill matching and gap identification
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 4.1: Skill Normalization
# ------------------------------------------------------------------------------
# This stage provides functions for standardizing skill names and comparisons.


def normalize_skill(skill: str) -> str:
    """
    Normalize a skill string for consistent comparison.

    Normalization rules:
    - Convert to lowercase
    - Remove extra whitespace
    - Handle common synonyms (e.g., "JS" -> "javascript")

    Args:
        skill: Raw skill string

    Returns:
        Normalized skill string

    Design Note:
        Consistent skill normalization improves matching accuracy.
    """
    skill = skill.lower().strip()

    # Common synonyms mapping
    synonyms = {
        "js": "javascript",
        "ts": "typescript",
        "py": "python",
        "k8s": "kubernetes",
        "docker": "containerization",
        "ci/cd": "continuous integration",
        "ml": "machine learning",
        "ai": "artificial intelligence",
    }

    return synonyms.get(skill, skill)


def extract_resume_skills(resume: Resume) -> set[str]:
    """
    Extract all skills from a resume into a normalized set.

    This includes:
    - Explicitly listed skills
    - Skills mentioned in experience descriptions
    - Skills from project descriptions

    Args:
        resume: Resume object

    Returns:
        Set of normalized skill strings

    Design Note:
        Using a set ensures uniqueness and enables fast lookup.
    """
    skills = set()

    # Add explicit skills
    for skill in resume.skills:
        skills.add(normalize_skill(skill.name))

    # Extract skills from experience descriptions
    # This is a simple approach - the LLM agent will do more sophisticated extraction
    # The LLM will handle this more intelligently
    # This is just a fallback helper function

    logger.debug(f"Extracted {len(skills)} normalized skills from resume")
    return skills


def extract_job_requirements(job: JobDescription) -> dict[str, str]:
    """
    Extract job requirements into a normalized dictionary with importance levels.

    Args:
        job: JobDescription object

    Returns:
        Dictionary mapping normalized skill -> importance level

    Design Note:
        Preserving importance levels enables prioritized gap analysis.
    """
    requirements = {}

    for req in job.requirements:
        normalized = normalize_skill(req.requirement)
        requirements[normalized] = req.importance

    logger.debug(f"Extracted {len(requirements)} requirements from job description")
    return requirements


# ==============================================================================
# BLOCK 5: ANALYSIS QUALITY CHECKS
# ==============================================================================
# PURPOSE: Validate the quality and completeness of gap analysis results
# WHAT: Comprehensive quality assessment with scoring and recommendations
# WHY: Ensures analysis results are reliable and actionable for optimization
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 5.1-5.3: Complete Quality Assessment Workflow
# ------------------------------------------------------------------------------
# This stage performs comprehensive quality checks on gap analysis results.


def check_analysis_quality(strategy: AlignmentStrategy) -> dict:
    """
    Perform quality checks on the gap analysis report.

    This function validates that the analysis is comprehensive, logical,
    and actionable for downstream agents.

    Args:
        report: The validated AnalysisReport object

    Returns:
        Dictionary with quality check results and recommendations

    Quality Checks:
        - Is the match score reasonable (0-100)?
        - Are skill matches identified?
        - Are critical gaps identified if match is low?
        - Is there actionable feedback?
        - Are confidence scores present?

    Design Note:
        This helps catch incomplete or illogical analyses early.
    """
    issues = []
    warnings = []
    score = 100

    # Check fit score validity
    if strategy.overall_fit_score < 0 or strategy.overall_fit_score > 100:
        issues.append(f"Invalid fit score: {strategy.overall_fit_score}")
        score -= 40

    # Check for skill matches
    if not strategy.identified_matches or len(strategy.identified_matches) == 0:
        warnings.append("No skill matches identified")
        score -= 15

    # Check for skill gaps
    if not strategy.identified_gaps or len(strategy.identified_gaps) == 0:
        warnings.append("No skill gaps identified")
        score -= 10

    # Check for keywords
    if not strategy.keywords_to_integrate or len(strategy.keywords_to_integrate) == 0:
        issues.append("No keywords to integrate identified")
        score -= 25

    # Check for guidance fields
    if (
        not strategy.professional_summary_guidance
        or len(strategy.professional_summary_guidance) < 20
    ):
        issues.append("Professional summary guidance is missing or too brief")
        score -= 15

    if not strategy.experience_guidance or len(strategy.experience_guidance) < 20:
        issues.append("Experience guidance is missing or too brief")
        score -= 15

    if not strategy.skills_guidance or len(strategy.skills_guidance) < 20:
        issues.append("Skills guidance is missing or too brief")
        score -= 15

    # Logical consistency check
    if strategy.overall_fit_score > 90 and len(strategy.identified_gaps) > 5:
        warnings.append("High fit score but many gaps - may be inconsistent")
        score -= 10

    if strategy.overall_fit_score < 50 and len(strategy.identified_gaps) == 0:
        warnings.append("Low fit score but no gaps identified - inconsistent")
        score -= 15

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
        logger.warning(f"Analysis quality issues found: {issues}")
    if warnings:
        logger.info(f"Analysis quality warnings: {warnings}")

    logger.info(f"Gap analysis quality check: {quality} (score: {score}/100)")

    return result


def calculate_coverage_stats(strategy: AlignmentStrategy) -> dict:
    """
    Calculate additional statistics about the gap analysis.

    This provides a quick summary of coverage metrics that can be
    useful for monitoring and debugging.

    Args:
        report: The validated AnalysisReport object

    Returns:
        Dictionary with coverage statistics

    Example Output:
        {
            "total_matches": 12,
            "total_gaps": 5,
            "critical_gaps": 2,
            "coverage_ratio": 0.71,
            "match_score": 85.5
        }
    """
    total_matches = len(strategy.identified_matches)
    total_gaps = len(strategy.identified_gaps)
    total_keywords = len(strategy.keywords_to_integrate)

    # Calculate coverage ratio
    total_requirements = total_matches + total_gaps
    coverage_ratio = total_matches / total_requirements if total_requirements > 0 else 0

    stats = {
        "total_matches": total_matches,
        "total_gaps": total_gaps,
        "keywords_to_integrate": total_keywords,
        "coverage_ratio": round(coverage_ratio, 2),
        "fit_score": round(strategy.overall_fit_score, 1),
    }

    logger.info(
        f"Coverage stats: {stats['total_matches']} matches, "
        f"{stats['total_gaps']} gaps, "
        f"{stats['coverage_ratio'] * 100:.0f}% coverage"
    )

    return stats


# ==============================================================================
# BLOCK 6: UTILITY FUNCTIONS
# ==============================================================================
# PURPOSE: Provide utility functions for debugging, monitoring, and testing
# WHAT: Helper functions for agent metadata and diagnostic information
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
        'Gap Analysis Specialist'
    """
    config = _load_agent_config()
    return {
        "name": "Gap Analysis Specialist",
        "role": config.get("role", "Unknown"),
        "llm": config.get("llm", "Unknown"),
        "tools": [],
        "output_model": "AlignmentStrategy",
    }


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
    print("Gap Analysis Specialist Agent - Test")
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
        agent = create_gap_analysis_agent()
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

    # Test helper functions
    print("\n--- Testing Helper Functions ---")
    try:
        # Test skill normalization
        test_skills = ["Python", "JS", "k8s", "Docker", "  JavaScript  "]
        print("Skill Normalization:")
        for skill in test_skills:
            print(f"  {skill:20s} -> {normalize_skill(skill)}")
    except Exception as e:
        print(f"Helper function test failed: {str(e)}")

    # Test quality check function with mock data
    print("\n--- Testing Quality Check Function ---")
    try:
        from src.data_models.strategy import AlignmentStrategy, SkillGap, SkillMatch

        mock_strategy = AlignmentStrategy(
            overall_fit_score=85.5,
            summary_of_strategy="Focus on cloud and containerization experience",
            identified_matches=[
                SkillMatch(
                    resume_skill="Python Development",
                    job_requirement="5+ years of Python programming",
                    match_score=95.0,
                    justification="Candidate has 5 years of Python experience across multiple projects",
                ),
                SkillMatch(
                    resume_skill="Docker",
                    job_requirement="Container orchestration experience",
                    match_score=85.0,
                    justification="Extensive Docker usage in microservices architecture",
                ),
            ],
            identified_gaps=[
                SkillGap(
                    missing_skill="Kubernetes",
                    importance="must_have",
                    suggestion="Review Docker experience for any orchestration work that can be reframed as Kubernetes-adjacent skills",
                )
            ],
            keywords_to_integrate=["Python", "Docker", "Microservices", "AWS"],
            professional_summary_guidance="Emphasize cloud infrastructure experience and Python expertise",
            experience_guidance="Highlight Docker containerization projects at previous roles",
            skills_guidance="List cloud technologies first, prioritize AWS and Docker",
        )

        quality_result = check_analysis_quality(mock_strategy)
        print(f"Quality: {quality_result['quality']}")
        print(f"Score: {quality_result['score']}/100")
        print(f"Acceptable: {quality_result['is_acceptable']}")
        if quality_result["issues"]:
            print(f"Issues: {quality_result['issues']}")
        if quality_result["warnings"]:
            print(f"Warnings: {quality_result['warnings']}")

        # Test coverage stats
        print("\n--- Testing Coverage Stats ---")
        stats = calculate_coverage_stats(mock_strategy)
        for key, value in stats.items():
            print(f"{key}: {value}")

    except Exception as e:
        print(f"Quality check test failed: {str(e)}")

    print("\n" + "=" * 70)
