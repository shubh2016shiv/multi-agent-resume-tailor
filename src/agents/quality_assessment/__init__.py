"""
Quality Assessment module.

Public surface:
  create_quality_assessment_agent  -- build the CrewAI Agent (use in orchestrator)

The code-owned render gate lives in engines.py:
  from src.agents.quality_assessment.engines import apply_quality_gate, should_render_resume
"""

from src.agents.quality_assessment.agent import create_quality_assessment_agent

__all__ = [
    "create_quality_assessment_agent",
]
