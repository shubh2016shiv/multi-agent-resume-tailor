"""LLM token tracking exceptions."""


# Subclasses ValueError (not a standalone Exception) because an exceeded budget
# is itself a value problem — the same category as the plain ValueError
# budget_guard.py raises for a negative max_tokens argument. A caller doing
# `except ValueError` catches both; catch this class specifically to
# distinguish "budget exceeded" from "bad argument."
class TokenBudgetExceeded(ValueError):
    """Raised when input text exceeds an explicit token budget."""
