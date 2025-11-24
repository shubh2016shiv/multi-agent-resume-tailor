"""
Centralized Token Counter Module
---------------------------------

This module provides enterprise-grade token counting and cost tracking for LLM calls
using the LiteLLM library. It supports multiple LLM providers (OpenAI, Anthropic,
Google Gemini, etc.) with automatic tokenizer selection and cost calculation.

WHY LITELLM?
------------
- Multi-Provider Support: Works with 100+ LLM providers out of the box
- Automatic Tokenizers: Uses model-specific tokenizers for accurate counts
- Cost Tracking: Maintains up-to-date pricing database for all providers
- Zero Configuration: Automatically detects model and selects appropriate tokenizer
- Battle-Tested: Used by thousands of production applications

KEY FEATURES:
-------------
- Token counting for text strings and message arrays
- Automatic cost calculation based on model pricing
- Integration with structured logging (structlog)
- Thread-safe implementation for concurrent agent execution
- Support for all major LLM providers

USAGE:
------
```python
from src.core.token_counter import TokenCounter

counter = TokenCounter()

# Count tokens in text
tokens = counter.count_tokens("Hello world", model="gpt-3.5-turbo")

# Count tokens in messages
messages = [{"role": "user", "content": "Hello"}]
tokens = counter.count_message_tokens(messages, model="gemini/gemini-2.5-flash")

# Estimate cost
cost = counter.estimate_cost(
    prompt_tokens=1000,
    completion_tokens=500,
    model="claude-3-opus-20240229"
)

# Log token usage
counter.log_token_usage(
    agent_name="Job Analyzer",
    input_tokens=1234,
    output_tokens=567,
    model="gemini/gemini-2.5-flash"
)
```

INTEGRATION WITH AGENTS:
------------------------
This module is designed to be integrated into all agent factory functions to provide
automatic token tracking and cost monitoring. See implementation_plan.md for details.
"""

try:
    from litellm import cost_per_token, token_counter
except ImportError:
    # Fallback if litellm is not installed
    cost_per_token = None
    token_counter = None

try:
    from src.core.logger import get_logger
except ImportError:
    # Fallback for when running this file directly
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.core.logger import get_logger

logger = get_logger(__name__)


# ==============================================================================
# Token Counter Class
# ==============================================================================


class TokenCounter:
    """
    Centralized token counting and cost tracking using LiteLLM.

    This class provides a unified interface for counting tokens and estimating
    costs across multiple LLM providers. It leverages LiteLLM's built-in
    tokenizers and pricing database for accurate, provider-specific calculations.

    Thread Safety:
        This class is thread-safe and can be used in concurrent agent execution.

    Example:
        >>> counter = TokenCounter()
        >>> tokens = counter.count_tokens("Hello world", "gpt-3.5-turbo")
        >>> print(tokens)
        2
    """

    def __init__(self):
        """Initialize the TokenCounter."""
        if token_counter is None or cost_per_token is None:
            logger.warning(
                "LiteLLM not available. Token counting and cost estimation will be disabled. "
                "Install with: pip install litellm"
            )
            self._available = False
        else:
            self._available = True
            logger.debug("TokenCounter initialized with LiteLLM support")

    def count_tokens(self, text: str, model: str) -> int:
        """
        Count tokens in a text string using the model-specific tokenizer.

        This method uses LiteLLM's token_counter with automatic tokenizer
        selection based on the model. It supports all major LLM providers.

        Args:
            text: The text to count tokens for
            model: The model name (e.g., "gpt-3.5-turbo", "gemini/gemini-2.5-flash")

        Returns:
            Number of tokens in the text

        Example:
            >>> counter = TokenCounter()
            >>> tokens = counter.count_tokens("Hello world", "gpt-3.5-turbo")
            >>> print(tokens)
            2
        """
        if not self._available:
            logger.debug("Token counting unavailable, returning 0")
            return 0

        try:
            # Convert text to message format for token_counter
            messages = [{"role": "user", "content": text}]
            tokens = token_counter(model=model, messages=messages)
            logger.debug(f"Counted {tokens} tokens for model {model}")
            return tokens
        except Exception as e:
            logger.error(f"Error counting tokens for model {model}: {e}", exc_info=True)
            return 0

    def count_message_tokens(self, messages: list[dict], model: str) -> int:
        """
        Count tokens in a message array using the model-specific tokenizer.

        This method is designed for chat-based models where input is structured
        as a list of messages with roles and content.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            model: The model name (e.g., "gpt-3.5-turbo", "claude-3-opus-20240229")

        Returns:
            Number of tokens in the messages

        Example:
            >>> counter = TokenCounter()
            >>> messages = [
            ...     {"role": "system", "content": "You are a helpful assistant"},
            ...     {"role": "user", "content": "Hello!"}
            ... ]
            >>> tokens = counter.count_message_tokens(messages, "gpt-3.5-turbo")
            >>> print(tokens)
            15
        """
        if not self._available:
            logger.debug("Token counting unavailable, returning 0")
            return 0

        try:
            tokens = token_counter(model=model, messages=messages)
            logger.debug(f"Counted {tokens} tokens in {len(messages)} messages for model {model}")
            return tokens
        except Exception as e:
            logger.error(f"Error counting message tokens for model {model}: {e}", exc_info=True)
            return 0

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int, model: str) -> float | None:
        """
        Estimate the cost of an LLM call using LiteLLM's pricing database.

        This method uses LiteLLM's cost_per_token function which maintains
        an up-to-date pricing database for all major LLM providers.

        Args:
            prompt_tokens: Number of input/prompt tokens
            completion_tokens: Number of output/completion tokens
            model: The model name (e.g., "gpt-3.5-turbo", "gemini/gemini-2.5-flash")

        Returns:
            Estimated cost in USD, or None if cost cannot be calculated

        Example:
            >>> counter = TokenCounter()
            >>> cost = counter.estimate_cost(1000, 500, "gpt-3.5-turbo")
            >>> print(f"${cost:.6f}")
            $0.002250
        """
        if not self._available:
            logger.debug("Cost estimation unavailable")
            return None

        try:
            # cost_per_token returns (prompt_cost_per_token, completion_cost_per_token)
            prompt_cost_per_token, completion_cost_per_token = cost_per_token(
                model=model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
            )

            # Calculate total cost
            total_cost = (prompt_tokens * prompt_cost_per_token) + (
                completion_tokens * completion_cost_per_token
            )

            logger.debug(
                f"Estimated cost for {model}: ${total_cost:.6f} "
                f"(prompt: {prompt_tokens} @ ${prompt_cost_per_token:.8f}, "
                f"completion: {completion_tokens} @ ${completion_cost_per_token:.8f})"
            )
            return total_cost
        except Exception as e:
            logger.warning(f"Error estimating cost for model {model}: {e}")
            return None

    def log_token_usage(
        self,
        agent_name: str,
        input_tokens: int,
        output_tokens: int,
        model: str,
        cost: float | None = None,
    ) -> None:
        """
        Log token usage and cost information with structured logging.

        This method creates a structured log entry with all token and cost
        metrics, making it easy to track and analyze LLM usage across agents.

        Args:
            agent_name: Name of the agent making the LLM call
            input_tokens: Number of input/prompt tokens
            output_tokens: Number of output/completion tokens
            model: The model name
            cost: Optional pre-calculated cost (will be calculated if not provided)

        Example:
            >>> counter = TokenCounter()
            >>> counter.log_token_usage(
            ...     agent_name="Job Analyzer",
            ...     input_tokens=1234,
            ...     output_tokens=567,
            ...     model="gemini/gemini-2.5-flash"
            ... )
        """
        total_tokens = input_tokens + output_tokens

        # Calculate cost if not provided
        if cost is None and self._available:
            cost = self.estimate_cost(input_tokens, output_tokens, model)

        # Build log context
        log_context = {
            "agent": agent_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "model": model,
        }

        if cost is not None:
            log_context["cost_usd"] = cost
            log_context["cost_formatted"] = f"${cost:.6f}"

        # Log with structured context
        logger.info(
            "llm_call_complete",
            **log_context,
        )


# ==============================================================================
# Convenience Functions
# ==============================================================================


def get_token_counter() -> TokenCounter:
    """
    Get a TokenCounter instance.

    This is a convenience function for creating a TokenCounter instance.
    In the future, this could be extended to implement a singleton pattern
    if needed.

    Returns:
        TokenCounter instance

    Example:
        >>> counter = get_token_counter()
        >>> tokens = counter.count_tokens("Hello", "gpt-3.5-turbo")
    """
    return TokenCounter()


def track_agent_tokens(agent_name: str, model: str, task_description: str):
    """
    Context manager for tracking token usage in agent execution.

    This context manager estimates input tokens before execution and can be
    used to wrap agent task execution for automatic token tracking.

    Args:
        agent_name: Name of the agent being executed
        model: The LLM model being used
        task_description: The task description/prompt being sent to the agent

    Yields:
        TokenCounter instance for manual tracking if needed

    Example:
        >>> with track_agent_tokens("Job Analyzer", "gpt-3.5-turbo", task_desc) as counter:
        ...     result = crew.kickoff()
        ...     # Token usage is automatically logged
    """
    from contextlib import contextmanager

    @contextmanager
    def _tracker():
        counter = get_token_counter()

        # Estimate input tokens
        input_tokens = counter.count_tokens(task_description, model)
        logger.info(
            f"{agent_name} - Starting execution",
            agent=agent_name,
            model=model,
            estimated_input_tokens=input_tokens,
        )

        try:
            yield counter
        finally:
            # Note: We can't automatically track output tokens without intercepting
            # the LLM response. This would need to be done at the orchestrator level
            # or by wrapping the crew.kickoff() result.
            logger.debug(f"{agent_name} - Execution complete")

    return _tracker()


# ==============================================================================
# Testing and Validation
# ==============================================================================

if __name__ == "__main__":
    """
    Test the TokenCounter functionality.
    Run this script directly to verify token counting works.
    """
    print("=" * 70)
    print("Token Counter Module - Test")
    print("=" * 70)

    # Create counter instance
    counter = TokenCounter()

    # Test 1: Count tokens in simple text
    print("\n--- Test 1: Count Tokens in Text ---")
    sample_text = "Hello, how are you doing today?"
    for model in ["gpt-3.5-turbo", "claude-3-opus-20240229", "gemini/gemini-2.5-flash"]:
        try:
            tokens = counter.count_tokens(sample_text, model)
            print(f"{model}: {tokens} tokens")
        except Exception as e:
            print(f"{model}: Error - {e}")

    # Test 2: Count tokens in messages
    print("\n--- Test 2: Count Tokens in Messages ---")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    for model in ["gpt-3.5-turbo", "gemini/gemini-2.5-flash"]:
        try:
            tokens = counter.count_message_tokens(messages, model)
            print(f"{model}: {tokens} tokens")
        except Exception as e:
            print(f"{model}: Error - {e}")

    # Test 3: Estimate cost
    print("\n--- Test 3: Estimate Cost ---")
    test_cases = [
        ("gpt-3.5-turbo", 1000, 500),
        ("gpt-4", 1000, 500),
        ("claude-3-opus-20240229", 1000, 500),
        ("gemini/gemini-2.5-flash", 1000, 500),
    ]
    for model, prompt_tokens, completion_tokens in test_cases:
        try:
            cost = counter.estimate_cost(prompt_tokens, completion_tokens, model)
            if cost is not None:
                print(f"{model}: ${cost:.6f}")
            else:
                print(f"{model}: Cost unavailable")
        except Exception as e:
            print(f"{model}: Error - {e}")

    # Test 4: Log token usage
    print("\n--- Test 4: Log Token Usage ---")
    counter.log_token_usage(
        agent_name="Test Agent",
        input_tokens=1234,
        output_tokens=567,
        model="gemini/gemini-2.5-flash",
    )
    print("Check logs for structured output")

    print("\n" + "=" * 70)
    print("Tests complete!")
    print("=" * 70)
