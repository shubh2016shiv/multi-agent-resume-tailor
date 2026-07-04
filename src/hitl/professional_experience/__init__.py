"""Professional-experience HITL package: one feature, one home.

The whole candidate-clarification loop lives here:
  models.py          contracts (findings, sheet entries, paused-run manifest)
  clarifications.py  detect fact gaps + phrase questions (one LLM call) and
                     build the sheet entries (pure code)
  answers.py         route answered questions back to their exact bullets
  persistence.py     clarification sheet + paused-run manifest + the SQLite
                     LangGraph checkpointer that makes pause/resume durable

Import concrete submodules directly; this initializer stays light so callers
that only need models never load LLM plumbing.
"""
