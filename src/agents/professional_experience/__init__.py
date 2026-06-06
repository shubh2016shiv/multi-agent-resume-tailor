"""
Professional Experience module.

Public surface:
  create_professional_experience_agent  -- build the CrewAI Agent (use in orchestrator)
"""

from src.agents.professional_experience.agent import create_professional_experience_agent

__all__ = [
    "create_professional_experience_agent",
]
