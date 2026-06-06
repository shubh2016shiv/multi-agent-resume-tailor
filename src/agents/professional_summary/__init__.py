"""
Professional Summary module.

Public surface:
  create_professional_summary_agent  -- build the CrewAI Agent (use in orchestrator)
"""

from src.agents.professional_summary.agent import create_professional_summary_agent

__all__ = [
    "create_professional_summary_agent",
]
