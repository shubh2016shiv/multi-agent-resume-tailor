"""Data records for LLM token tracking."""

from dataclasses import dataclass


@dataclass(frozen=True)  # immutable: this is a logging/audit record, never mutated after creation
class TokenUsage:
    """Structured token and cost details for one model interaction."""

    agent_name: str
    input_tokens: int
    output_tokens: int
    model: str
    cost_usd: float | None = None

    @property
    def total_tokens(self) -> int:
        """Return combined input and output tokens."""
        return self.input_tokens + self.output_tokens
