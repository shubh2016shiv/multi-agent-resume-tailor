"""
Quality Feedback module.

Public surface:
  create_quality_feedback_agent -- build the optional advisory reviewer

Deterministic scoring and release decisions live in src.resume_quality_evaluation.
"""

from src.agents.quality_feedback.agent import create_quality_feedback_agent

__all__ = ["create_quality_feedback_agent"]
