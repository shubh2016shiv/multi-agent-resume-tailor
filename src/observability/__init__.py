"""Observability facade for the Resume Tailor multi-agent system.

This package is the ONE seam the rest of the app imports for tracing. The
backend is LangSmith; swapping vendors means changing only the backend modules,
not the ~15 call sites that import from here.

WHAT GETS CAPTURED
------------------
Agent behavior is traced on two layers (details in ``langsmith_backend.py``):
- Automatic LLM layer: every CrewAI/LiteLLM model call -> prompt, completion,
  tokens, cost, latency.
- Readable workflow layer: ``@trace_agent`` / ``@trace_tool`` add named spans
  so the dashboard shows a per-agent -> per-LLM-call tree.

Quick start
-----------
```python
# 1. Initialize once at application startup.
from src.observability import init_observability
init_observability("resume-tailor-agents")

# 2. Decorate agent / tool functions.
from src.observability import trace_agent, trace_tool

@trace_agent
def run_my_agent(input_data):
    return processed_data

# 3. Log iteration metrics during critique/regenerate loops.
from src.observability import log_iteration_metrics
log_iteration_metrics("my_agent", iteration=1, metrics={"score": 85, "tokens": 1200})
```

All functions degrade to safe no-ops when tracing is disabled or the
``LANGSMITH_API_KEY`` is unset — the pipeline always runs.
"""

from src.observability.iteration_metrics import log_iteration_metrics
from src.observability.langsmith_backend import init_observability, is_observability_enabled
from src.observability.tracing import trace_agent, trace_tool

__all__ = [
    "init_observability",
    "trace_agent",
    "trace_tool",
    "log_iteration_metrics",
    "is_observability_enabled",
]
