"""
Observability Module
====================

This module provides centralized observability for the Resume Tailor multi-agent
system using Weave from Weights & Biases.

Quick Start:
------------
```python
# 1. Initialize observability at application startup
from src.observability import init_observability
init_observability("resume-tailor-agents")

# 2. Decorate agent functions
from src.observability import trace_agent

@trace_agent
def run_my_agent(input_data):
    return processed_data

# 3. Decorate tool functions
from src.observability import trace_tool

@trace_tool
def evaluate_quality(content):
    return quality_score

# 4. Log iteration metrics
from src.observability import log_iteration_metrics

log_iteration_metrics(
    agent_name="my_agent",
    iteration=1,
    metrics={"score": 85, "tokens": 1200}
)
```

For detailed documentation, see: src/observability/weave_setup.py
"""

from .weave_setup import (
    get_weave_project_url,
    init_observability,
    is_weave_enabled,
    log_iteration_metrics,
    trace_agent,
    trace_tool,
)

__all__ = [
    "init_observability",
    "trace_agent",
    "trace_tool",
    "log_iteration_metrics",
    "is_weave_enabled",
    "get_weave_project_url",
]
