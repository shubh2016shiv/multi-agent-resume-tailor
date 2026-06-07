"""
Job Description Analyser module.

Public surface:
  create_job_analyzer_agent  -- build the CrewAI Agent (use in orchestration nodes)
"""

from src.agents.job_description_analyser.agent import create_job_analyzer_agent

__all__ = [
    "create_job_analyzer_agent",
]
