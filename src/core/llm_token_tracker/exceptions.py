"""LLM token tracking exceptions."""


class TokenBudgetExceeded(ValueError):
    """Raised when input text exceeds an explicit token budget."""
