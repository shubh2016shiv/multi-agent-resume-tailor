"""
Resume Extractor module.

Public surface:
  create_resume_extractor_agent  -- build the CrewAI Agent (use in orchestrator)
"""

from src.agents.resume_parser.agent import create_resume_extractor_agent

__all__ = [
    "create_resume_extractor_agent",
]
