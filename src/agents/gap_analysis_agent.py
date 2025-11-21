"""
Gap Analysis Specialist Agent
-----------------------------

This module defines the third agent in our workflow: the Gap Analysis Specialist.
This agent is responsible for comparing a candidate's resume against a job description
to identify skill matches, gaps, and alignment opportunities.

AGENT DESIGN PRINCIPLES:
- Single Responsibility: Compare resume vs job requirements and identify gaps
- Modularity: Clear separation between agent creation, analysis logic, and validation
- Robustness: Comprehensive error handling with graceful degradation
- Type Safety: Uses Pydantic models for validated, structured output
- Observability: Detailed logging at every step for debugging

WORKFLOW:
1. Receive structured Resume and JobDescription objects as input
2. Analyze skill matches between resume and job requirements
3. Identify missing must-have skills (critical gaps)
4. Identify missing nice-to-have skills (opportunities)
5. Calculate match percentages and confidence scores
6. Provide prioritized gap recommendations
7. Return structured AlignmentStrategy object (JSON)

KEY ANALYSIS DIMENSIONS:
- Hard Skills: Technical competencies (e.g., Python, AWS, Docker)
- Soft Skills: Interpersonal abilities (e.g., leadership, communication)
- Experience Level: Years of experience alignment
- Domain Knowledge: Industry-specific expertise
- Certifications: Professional qualifications

CRITICAL INSIGHTS PROVIDED:
- Match Confidence Score (0-100%)
- Skill Coverage Percentage
- Critical Gaps (must-have skills missing)
- Enhancement Opportunities (nice-to-have skills)
- Experience Level Alignment
- Keyword Optimization Suggestions
"""
        )

        return agent

    except Exception as e:
        logger.error(f"Failed to create Gap Analysis Specialist agent: {e}", exc_info=True)
        raise


# ==============================================================================
# Output Validation
# ==============================================================================


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
# Analysis Helper Functions
# ==============================================================================


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
# Analysis Quality Checks
# ==============================================================================


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
# Testing Block
# ==============================================================================

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
