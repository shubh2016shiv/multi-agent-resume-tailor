"""
Resume enhancement pipeline -- LangGraph-based orchestration.

Public surface:
  tailor_resume(resume_path, jd_path) -> OrchestrationResult
  resume_paused_run(paused_run_path) -> OrchestrationResult

Internal modules (not for direct import by callers):
  state.py  -- ResumeEnhancementPipelineState TypedDict
  nodes/    -- one module per pipeline stage, one node function per module
  graph.py  -- DAG topology (nodes wired to edges)
  runner.py -- compiles the graph and invokes it
"""

from src.orchestration.runner import resume_paused_run, tailor_resume

__all__ = ["resume_paused_run", "tailor_resume"]
