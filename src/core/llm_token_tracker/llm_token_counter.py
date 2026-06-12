"""LiteLLM-backed token and cost counter."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

try:
    from litellm import cost_per_token, token_counter
except ImportError:
    cost_per_token = None
    token_counter = None

from src.core.llm_token_tracker.llm_token_usage import TokenUsage
from src.core.logger import get_logger

logger = get_logger(__name__)


class TokenCounter:
    """Count tokens and estimate costs through LiteLLM when available."""

    def __init__(self) -> None:
        """Initialize availability once for this counter instance."""
        self._available = token_counter is not None and cost_per_token is not None
        if self._available:
            logger.debug("token_counter_available", provider="litellm")
        else:
            logger.warning("token_counter_unavailable", provider="litellm")

    @property
    def available(self) -> bool:
        """Return whether LiteLLM-backed counting is available."""
        return self._available

    def count_tokens(self, text: str, model: str) -> int:
        """Count tokens in a string.

        Args:
            text: Text to count.
            model: Model name to pass to LiteLLM.

        Returns:
            Token count, or 0 when counting is unavailable.
        """
        if not text:
            return 0
        return self.count_message_tokens([{"role": "user", "content": text}], model)

    def count_message_tokens(self, messages: list[dict[str, Any]], model: str) -> int:
        """Count tokens in chat-style messages.

        Args:
            messages: Message dictionaries with role/content keys.
            model: Model name to pass to LiteLLM.

        Returns:
            Token count, or 0 when counting is unavailable.
        """
        if not self._available or token_counter is None:
            logger.debug("token_count_skipped", reason="litellm_unavailable")
            return 0

        try:
            tokens = token_counter(model=model, messages=messages)
        except Exception as exc:
            logger.warning(
                "token_count_failed",
                model=model,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return 0
        return int(tokens or 0)

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int, model: str) -> float | None:
        """Estimate USD cost for a model interaction.

        Args:
            prompt_tokens: Input token count.
            completion_tokens: Output token count.
            model: Model name to pass to LiteLLM.

        Returns:
            Estimated USD cost, or None when unavailable.
        """
        if not self._available or cost_per_token is None:
            logger.debug("cost_estimation_skipped", reason="litellm_unavailable")
            return None

        try:
            prompt_cost, completion_cost = cost_per_token(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        except Exception as exc:
            logger.warning(
                "cost_estimation_failed",
                model=model,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return None
        return (prompt_tokens * prompt_cost) + (completion_tokens * completion_cost)

    def log_token_usage(
        self,
        agent_name: str,
        input_tokens: int,
        output_tokens: int,
        model: str,
        cost: float | None = None,
    ) -> None:
        """Log structured token usage for an agent interaction.

        Args:
            agent_name: Agent or component name.
            input_tokens: Prompt/input tokens.
            output_tokens: Completion/output tokens.
            model: Model used for the interaction.
            cost: Optional precomputed USD cost.
        """
        usage = self.build_usage(agent_name, input_tokens, output_tokens, model, cost)
        log_context: dict[str, Any] = {
            "agent": usage.agent_name,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
            "model": usage.model,
        }
        if usage.cost_usd is not None:
            log_context["cost_usd"] = usage.cost_usd
            log_context["cost_formatted"] = f"${usage.cost_usd:.6f}"
        logger.info("llm_call_complete", **log_context)

    def build_usage(
        self,
        agent_name: str,
        input_tokens: int,
        output_tokens: int,
        model: str,
        cost: float | None = None,
    ) -> TokenUsage:
        """Build a structured token usage record.

        Args:
            agent_name: Agent or component name.
            input_tokens: Prompt/input tokens.
            output_tokens: Completion/output tokens.
            model: Model used for the interaction.
            cost: Optional precomputed USD cost.

        Returns:
            Immutable token usage record.
        """
        resolved_cost = cost
        if resolved_cost is None:
            resolved_cost = self.estimate_cost(input_tokens, output_tokens, model)
        return TokenUsage(agent_name, input_tokens, output_tokens, model, resolved_cost)


@lru_cache
def get_token_counter() -> TokenCounter:
    """Return the process-wide token counter instance."""
    return TokenCounter()
