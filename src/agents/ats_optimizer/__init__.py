"""
ATS Optimizer module.

Public surface:
  create_ats_optimizer_agent  -- build the CrewAI Agent (use in orchestrator)
"""

from src.agents.ats_optimizer.agent import create_ats_optimizer_agent

__all__ = [
    "create_ats_optimizer_agent",
]
