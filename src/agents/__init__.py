"""
Agents Package
-------------

This package contains all the specialized agents for the Resume Tailor system.
Each agent is responsible for a specific part of the workflow and is built with:
- Modularity: Clear separation of concerns
- Robustness: Comprehensive error handling
- Documentation: Self-explanatory code with detailed comments
"""

from src.agents.ats_optimization_agent import (
    assemble_resume_components,
    check_ats_quality,
    create_ats_optimization_agent,
    generate_json_resume,
    generate_markdown_resume,
    validate_ats_compatibility,
    validate_optimized_output,
)
from src.agents.experience_optimizer_agent import (
    check_experience_quality,
    create_experience_optimizer_agent,
    evaluate_experience_bullets,
    log_iteration_progress,
    validate_experience_output,
)
from src.agents.gap_analysis_agent import (
    calculate_coverage_stats,
    create_gap_analysis_agent,
    validate_analysis_output,
)
from src.agents.gap_analysis_agent import (
    check_analysis_quality as check_strategy_quality,
)
from src.agents.job_analyzer_agent import (
    check_analysis_quality as check_job_quality,
)
from src.agents.job_analyzer_agent import (
    create_job_analyzer_agent,
    validate_job_output,
)
from src.agents.resume_extractor_agent import (
    create_resume_extractor_agent,
    validate_resume_output,
)
from src.agents.skills_optimizer_agent import (
    check_skills_quality,
    create_skills_optimizer_agent,
    validate_skills_output,
)
from src.agents.summary_writer_agent import (
    analyze_keyword_integration,
    check_summary_quality,
    create_summary_writer_agent,
    validate_summary_output,
)
from src.agents.quality_assurance_agent import (
    calculate_weighted_score,
    check_formatting_standards,
    check_grammar_quality,
    check_qa_quality,
    check_red_flags,
    create_quality_assurance_agent,
    evaluate_accuracy,
    evaluate_ats_optimization,
    evaluate_relevance,
    validate_qa_output,
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
    # ATS Optimization Specialist Agent
    "create_ats_optimization_agent",
    "validate_optimized_output",
    "check_ats_quality",
    "validate_ats_compatibility",
    "assemble_resume_components",
    "generate_markdown_resume",
    "generate_json_resume",
    # Quality Assurance Reviewer Agent
    "create_quality_assurance_agent",
    "validate_qa_output",
    "check_qa_quality",
    "evaluate_accuracy",
    "evaluate_relevance",
    "evaluate_ats_optimization",
    "calculate_weighted_score",
    "check_grammar_quality",
    "check_formatting_standards",
    "check_red_flags",
]
