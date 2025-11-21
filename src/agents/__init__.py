"""
Agents Package
-------------

This package contains all the specialized agents for the Resume Tailor system.
Each agent is responsible for a specific part of the workflow and is built with:
- Modularity: Clear separation of concerns
- Robustness: Comprehensive error handling
- Documentation: Self-explanatory code with detailed comments
"""

from src.agents.resume_extractor_agent import (
    create_resume_extractor_agent,
    validate_resume_output,
)
from src.agents.job_analyzer_agent import (
    create_job_analyzer_agent,
    validate_job_output,
    check_analysis_quality as check_job_quality,
)
from src.agents.gap_analysis_agent import (
    create_gap_analysis_agent,
    validate_analysis_output,
    check_analysis_quality as check_strategy_quality,
    calculate_coverage_stats,
)
from src.agents.summary_writer_agent import (
    create_summary_writer_agent,
    validate_summary_output,
    check_summary_quality,
    analyze_keyword_integration,
)
from src.agents.experience_optimizer_agent import (
    create_experience_optimizer_agent,
    validate_experience_output,
    check_experience_quality,
    evaluate_experience_bullets,
    log_iteration_progress,
)
from src.agents.skills_optimizer_agent import (
    create_skills_optimizer_agent,
    validate_skills_output,
    check_skills_quality,
)

__all__ = [
    # Resume Extractor Agent
    "create_resume_extractor_agent",
    "validate_resume_output",
    # Job Analyzer Agent
    "create_job_analyzer_agent",
    "validate_job_output",
    "check_job_quality",
    # Gap Analysis Agent
    "create_gap_analysis_agent",
    "validate_analysis_output",
    "check_strategy_quality",
    "calculate_coverage_stats",
    # Professional Summary Writer Agent
    "create_summary_writer_agent",
    "validate_summary_output",
    "check_summary_quality",
    "analyze_keyword_integration",
    # Experience Section Optimizer Agent
    "create_experience_optimizer_agent",
    "validate_experience_output",
    "check_experience_quality",
    "evaluate_experience_bullets",
    "log_iteration_progress",
    # Skills Section Strategist Agent
    "create_skills_optimizer_agent",
    "validate_skills_output",
    "check_skills_quality",
]

