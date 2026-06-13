"""Public agent factory surface for the resume tailoring pipeline.

Import only active subpackage factories here. Legacy flat agent modules were
removed from the runtime path, so this facade intentionally avoids old helper
exports that would make package import fragile.
"""

from src.agents.ats_optimizer import create_ats_optimizer_agent
from src.agents.gap_analysis import create_gap_analysis_agent
from src.agents.job_description_analyser import create_job_analyzer_agent
from src.agents.professional_experience import create_professional_experience_agent
from src.agents.professional_summary import create_professional_summary_agent
from src.agents.quality_assessment import create_quality_assessment_agent
from src.agents.resume_parser import create_resume_extractor_agent
from src.agents.skill_optimizer import create_skill_optimizer_agent

__all__ = [
    "create_ats_optimizer_agent",
    "create_gap_analysis_agent",
    "create_job_analyzer_agent",
    "create_professional_experience_agent",
    "create_professional_summary_agent",
    "create_quality_assessment_agent",
    "create_resume_extractor_agent",
    "create_skill_optimizer_agent",
]
