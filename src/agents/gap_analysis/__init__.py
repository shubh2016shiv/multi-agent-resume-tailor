"""
Gap Analysis module.

Public surface:
  create_gap_analysis_agent  -- build the CrewAI Agent (use in orchestrator)
"""

from src.agents.gap_analysis.agent import create_gap_analysis_agent

__all__ = [
    "create_gap_analysis_agent",
]
