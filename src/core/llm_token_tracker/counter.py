"""LiteLLM-backed token and cost counter."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

try:
    from litellm.cost_calculator import cost_per_token
    from litellm.utils import token_counter
except ImportError:
    cost_per_token = None
    token_counter = None

from src.core.llm_token_tracker.usage import TokenUsage
from src.core.logger import get_logger

logger = get_logger(__name__)


class TokenCounter:
    """Count tokens and estimate costs through LiteLLM when available."""

    def __init__(self) -> None:
        """Initialize availability once for this counter instance."""
        ####################################################
        # STEP 1: DETECT WHETHER THE LITELLM TOKEN APIS ARE AVAILABLE#
        ####################################################
        # The whole package degrades gracefully when LiteLLM is missing,
        # so we decide that once here and reuse it everywhere else.
        self._available = token_counter is not None and cost_per_token is not None

        ####################################################
        # STEP 2: LOG THE PROVIDER AVAILABILITY DECISION#
        ####################################################
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
        ####################################################
        # STEP 1: SHORT-CIRCUIT EMPTY TEXT#
        ####################################################
        # An empty prompt has no tokens, and skipping the provider call keeps
        # the behavior obvious for callers and tests.
        if not text:
            return 0

        ####################################################
        # STEP 2: REUSE THE MESSAGE-COUNTING PATH#
        ####################################################
        # We normalize plain text into the same chat-message shape used by the
        # provider so all counting logic lives in one place.
        return self.count_message_tokens([{"role": "user", "content": text}], model)

    def count_message_tokens(self, messages: list[dict[str, Any]], model: str) -> int:
        """Count tokens in chat-style messages.

        Args:
            messages: Message dictionaries with role/content keys.
            model: Model name to pass to LiteLLM.

        Returns:
            Token count, or 0 when counting is unavailable.
        """
        ####################################################
        # STEP 1: EXIT EARLY WHEN TOKEN COUNTING IS NOT AVAILABLE#
        ####################################################
        # The contract of this package is graceful degradation, not hard
        # failure, when LiteLLM is unavailable in the environment.
        if not self._available or token_counter is None:
            logger.debug("token_count_skipped", reason="litellm_unavailable")
            return 0

        ####################################################
        # STEP 2: ASK LITELLM TO COUNT THE CHAT MESSAGE TOKENS#
        ####################################################
        try:
            tokens = token_counter(model=model, messages=messages)
        except Exception as exc:
            ####################################################
            # STEP 3: DEGRADE GRACEFULLY WHEN THE PROVIDER CALL FAILS#
            ####################################################
            logger.warning(
                "token_count_failed",
                model=model,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return 0

        ####################################################
        # STEP 4: NORMALIZE THE RESULT TO A PLAIN INTEGER#
        ####################################################
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
        ####################################################
        # STEP 1: EXIT EARLY WHEN COST ESTIMATION IS NOT AVAILABLE#
        ####################################################
        if not self._available or cost_per_token is None:
            logger.debug("cost_estimation_skipped", reason="litellm_unavailable")
            return None

        ####################################################
        # STEP 2: ASK LITELLM FOR THE INPUT AND OUTPUT TOKEN COSTS#
        ####################################################
        try:
            prompt_cost_usd, completion_cost_usd = cost_per_token(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        except Exception as exc:
            ####################################################
            # STEP 3: DEGRADE GRACEFULLY WHEN PRICING LOOKUP FAILS#
            ####################################################
            logger.warning(
                "cost_estimation_failed",
                model=model,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return None

        ####################################################
        # STEP 4: COLLAPSE THE TWO COST COMPONENTS INTO ONE USD TOTAL#
        ####################################################
        return prompt_cost_usd + completion_cost_usd

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
        ####################################################
        # STEP 1: BUILD THE NORMALIZED USAGE RECORD#
        ####################################################
        usage = self.build_usage(agent_name, input_tokens, output_tokens, model, cost)

        ####################################################
        # STEP 2: PREPARE THE STRUCTURED LOG PAYLOAD#
        ####################################################
        log_context: dict[str, Any] = {
            "agent": usage.agent_name,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
            "model": usage.model,
        }

        ####################################################
        # STEP 3: ADD COST FIELDS ONLY WHEN A COST EXISTS#
        ####################################################
        if usage.cost_usd is not None:
            log_context["cost_usd"] = usage.cost_usd
            log_context["cost_formatted"] = f"${usage.cost_usd:.6f}"

        ####################################################
        # STEP 4: EMIT THE FINAL STRUCTURED LOG EVENT#
        ####################################################
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
        ####################################################
        # STEP 1: RESPECT ANY CALLER-SUPPLIED COST#
        ####################################################
        resolved_cost = cost

        ####################################################
        # STEP 2: FALL BACK TO COST ESTIMATION WHEN NEEDED#
        ####################################################
        if resolved_cost is None:
            resolved_cost = self.estimate_cost(input_tokens, output_tokens, model)

        ####################################################
        # STEP 3: RETURN THE IMMUTABLE USAGE VALUE OBJECT#
        ####################################################
        return TokenUsage(agent_name, input_tokens, output_tokens, model, resolved_cost)


@lru_cache
def get_token_counter() -> TokenCounter:
    """Return the process-wide token counter instance."""
    ####################################################
    # STEP 1: RETURN THE PROCESS-WIDE SHARED COUNTER INSTANCE#
    ####################################################
    # The LRU cache turns this tiny factory into a singleton-style accessor
    # so callers across the process reuse the same initialized counter.
    return TokenCounter()
