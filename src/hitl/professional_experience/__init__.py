"""Professional-experience HITL package: one feature, one home.

The whole candidate-clarification loop lives here:
  models.py          contracts (the fact gap, the question/answer, the paused run)
  clarifications.py  detect fact gaps + phrase questions (one LLM call) and
                     build the sheet entries (pure code)
  answers.py         route answered questions back to their exact bullets
  persistence.py     the paused-run directory: layout, clarification sheet,
                     manifest, and the append-only answer audit log

This package owns the human loop, not the machinery that pauses for it: the
LangGraph checkpointer and its serialization allowlist live in
src/orchestration/checkpointing.py, because every run checkpoints whether or not
it ever asks the candidate anything. Nothing here imports LangGraph.

Import concrete submodules directly; this initializer stays light so callers
that only need models never load LLM plumbing.
"""
