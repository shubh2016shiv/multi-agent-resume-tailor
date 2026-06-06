"""
Resume enhancement pipeline -- LangGraph-based orchestration.

Public surface:
  tailor_resume(resume_path, jd_path) -> OrchestrationResult

Internal modules (not for direct import by callers):
  state.py  -- ResumeEnhancementPipelineState TypedDict
  nodes.py  -- one function per agent call
  graph.py  -- DAG topology (nodes wired to edges)
  runner.py -- compiles the graph and invokes it
"""

from src.orchestration.runner import tailor_resume

__all__ = ["tailor_resume"]
