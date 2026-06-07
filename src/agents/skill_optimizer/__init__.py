"""
Skill Optimizer module.

Public surface:
  create_skill_optimizer_agent  -- build the CrewAI Agent (use in orchestrator)
"""

from src.agents.skill_optimizer.agent import create_skill_optimizer_agent

__all__ = [
    "create_skill_optimizer_agent",
]
