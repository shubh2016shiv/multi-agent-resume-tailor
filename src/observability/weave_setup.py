"""
Weave Observability Integration
================================

This module provides centralized observability for the Resume Tailor multi-agent system
using Weave from Weights & Biases. It enables comprehensive tracking of:

- Agent execution flows (inputs, outputs, intermediate steps)
- Iterative improvement loops (critique cycles, score progression)
- Token usage and cost tracking
- Quality metrics and evaluation scores
- Performance bottlenecks and timing

WHY WEAVE?
----------
- LLM-native tracing: Built specifically for AI agent workflows
- Automatic call hierarchy: Visualizes agent-to-agent and agent-to-tool flows
- Token tracking: Native support for LLM usage metrics
- Rich dashboard: Beautiful UI for debugging complex agent interactions
- Free tier: Generous 100GB storage for individual projects

ARCHITECTURE:
-------------
This module provides three main abstractions:

1. **init_observability()**: One-time initialization of Weave tracking
2. **@trace_agent**: Decorator for high-level agent execution functions
3. **@trace_tool**: Decorator for tools and helper functions called by agents
4. **log_iteration_metrics()**: Helper to log structured metrics during iterations

USAGE PATTERN:
--------------
```python
# 1. Initialize once at application startup
from src.observability import init_observability
init_observability("resume-tailor-agents")

# 2. Decorate agent execution functions
from src.observability import trace_agent

@trace_agent
def run_experience_optimizer(resume, job_desc, strategy):
    # Agent logic here
    return optimized_section

# 3. Decorate tools used by agents
from src.observability import trace_tool

@trace_tool
def evaluate_experience_bullets(bullets):
    # Evaluation logic
    return scores

# 4. Log metrics during iterations
from src.observability import log_iteration_metrics

log_iteration_metrics(
    agent_name="experience_optimizer",
    iteration=2,
    metrics={
        "quality_score": 87.5,
        "tokens_used": 1500,
        "improvement": "+12.3"
    }
)
```

INTEGRATION NOTES:
------------------
- Weave automatically creates a trace tree of all decorated function calls
- Parent-child relationships are preserved (agent -> tool -> sub-tool)
- All inputs/outputs are automatically serialized and logged
- Dashboard available at: https://wandb.ai/your-username/resume-tailor-agents
"""

import functools
import inspect
import os
from collections.abc import Callable
from typing import Any, TypeVar

from src.core.config import get_config
from src.core.logger import get_logger

# Initialize structured logger for this module
logger = get_logger(__name__)

# Type variable for generic decorator typing
F = TypeVar("F", bound=Callable[..., Any])

# Global flag to track if Weave is initialized
_weave_initialized = False

# Global reference to weave module (lazy import)
_weave = None


def _lazy_import_weave():
    """
    Lazy import of weave to avoid initialization issues.

    This function imports weave only when needed, preventing import-time
    side effects and allowing the application to start even if weave is
    not configured or available.

    Returns:
        The weave module if available, None otherwise
    """
    global _weave

    if _weave is not None:
        return _weave

    try:
        import weave

        _weave = weave
        return weave
    except ImportError:
        logger.warning(
            "weave_import_failed",
            message="Weave is not installed. Install with: pip install weave",
            impact="Observability features will be disabled",
        )
        return None


def init_observability(
    project_name: str = "resume-tailor-agents", entity: str | None = None, enabled: bool = True
) -> bool:
    """
    Initialize Weave observability for the project.

    This function should be called once at application startup, typically in
    the main orchestrator or entry point. It configures Weave to track all
    decorated functions and log metrics to the W&B dashboard.

    Args:
        project_name: Name of the W&B project (default: "resume-tailor-agents")
        entity: W&B entity/team name (optional, uses default if not provided)
        enabled: Whether to enable Weave tracking (default: True)

    Returns:
        bool: True if initialization succeeded, False otherwise

    Example:
        >>> init_observability("resume-tailor-prod", entity="my-team")
        True

    Notes:
        - Requires WANDB_API_KEY environment variable to be set
        - If initialization fails, the application continues without observability
        - Safe to call multiple times (subsequent calls are no-ops)
    """
    global _weave_initialized

    if _weave_initialized:
        logger.info(
            "weave_already_initialized",
            project=project_name,
            message="Weave is already initialized, skipping",
        )
        return True

    if not enabled:
        logger.info("weave_disabled", message="Weave observability is disabled by configuration")
        return False

    weave = _lazy_import_weave()
    if weave is None:
        return False

    try:
        # Check for API key from config (loaded from .env file)
        config = get_config()
        api_key = config.wandb_api_key

        # Fallback to environment variable for backward compatibility
        if not api_key:
            api_key = os.getenv("WANDB_API_KEY")

        if not api_key:
            logger.warning(
                "wandb_api_key_missing",
                message="WANDB_API_KEY not found in config or environment. Set it in .env file to enable observability.",
                hint="Get your API key from: https://wandb.ai/authorize and add WANDB_API_KEY=your_key to .env file",
            )
            return False

        # Initialize Weave
        # Note: weave.init() takes project name as positional argument
        # Entity/team is typically set via WANDB_ENTITY environment variable
        if entity:
            # Set entity via environment if provided
            os.environ["WANDB_ENTITY"] = entity

        weave.init(project_name)

        _weave_initialized = True

        logger.info(
            "weave_initialized",
            project=project_name,
            entity=entity or "default",
            message="Weave observability successfully initialized",
            dashboard_url=f"https://wandb.ai/{entity or 'your-username'}/{project_name}",
        )

        return True

    except Exception as e:
        logger.error(
            "weave_init_failed",
            error=str(e),
            error_type=type(e).__name__,
            message="Failed to initialize Weave observability",
        )
        return False


def trace_agent(func: F) -> F:
    """
    Decorator to trace agent execution with Weave.

    This decorator wraps high-level agent execution functions, capturing:
    - Function inputs (resume, job description, strategy, etc.)
    - Function outputs (optimized sections, reports, etc.)
    - Execution time and metadata
    - Parent-child relationships with tool calls

    Use this decorator for:
    - Agent task execution functions
    - Orchestrator workflow methods
    - High-level coordination functions

    Args:
        func: The function to decorate

    Returns:
        Decorated function with Weave tracing

    Example:
        >>> @trace_agent
        >>> def optimize_experience_section(resume, job_desc, strategy):
        >>>     # Agent logic
        >>>     return optimized_section

    Notes:
        - Automatically logs to Weave if initialized
        - Falls back gracefully if Weave is not available
        - Preserves function signature and docstring
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        weave = _lazy_import_weave()

        # If Weave is not initialized or not available, execute without tracing
        if not _weave_initialized or weave is None:
            return func(*args, **kwargs)

        try:
            # Create a simple wrapper function without type annotations to avoid
            # Weave's introspection issues with Union types (Python 3.10+ | syntax)
            # This wrapper will be traced by Weave instead of the original function
            def traced_wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            # Remove all annotations to prevent Weave from introspecting UnionType
            # This avoids the "'types.UnionType' object has no attribute '__name__'" error
            traced_wrapper.__annotations__ = {}

            # Create a signature without annotations to prevent introspection issues
            try:
                sig = inspect.signature(func)
                # Create a new signature with empty annotations
                params = []
                for param in sig.parameters.values():
                    # Create parameter without annotation
                    new_param = inspect.Parameter(
                        param.name,
                        param.kind,
                        default=param.default,
                        annotation=inspect.Parameter.empty,
                    )
                    params.append(new_param)
                new_sig = inspect.Signature(params, return_annotation=inspect.Signature.empty)
                traced_wrapper.__signature__ = new_sig
            except (ValueError, TypeError):
                # If signature introspection fails, just proceed without it
                pass

            # Use Weave's op decorator with explicit name to avoid signature introspection
            traced_func = weave.op(name=func.__name__)(traced_wrapper)
            result = traced_func(*args, **kwargs)

            logger.debug(
                "agent_traced",
                function=func.__name__,
                message="Agent execution traced successfully",
            )

            return result

        except Exception as e:
            logger.error(
                "agent_trace_failed",
                function=func.__name__,
                error=str(e),
                message="Failed to trace agent execution, executing without trace",
            )
            # Fallback: execute function without tracing
            return func(*args, **kwargs)

    return wrapper


def trace_tool(func: F) -> F:
    """
    Decorator to trace tool/helper function calls with Weave.

    This decorator wraps lower-level tool functions that are called by agents,
    capturing the detailed execution flow and creating a hierarchical trace tree.

    Use this decorator for:
    - Agent tools (evaluation functions, parsers, formatters)
    - Helper functions called during agent execution
    - Iterative improvement loops (critique/regenerate cycles)

    Args:
        func: The function to decorate

    Returns:
        Decorated function with Weave tracing

    Example:
        >>> @trace_tool
        >>> def evaluate_experience_bullets(bullets):
        >>>     # Evaluation logic
        >>>     return scores

    Notes:
        - Creates child traces under parent agent traces
        - Automatically captures inputs/outputs
        - Falls back gracefully if Weave is not available
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        weave = _lazy_import_weave()

        # If Weave is not initialized or not available, execute without tracing
        if not _weave_initialized or weave is None:
            return func(*args, **kwargs)

        try:
            # Create a simple wrapper function without type annotations to avoid
            # Weave's introspection issues with Union types (Python 3.10+ | syntax)
            # This wrapper will be traced by Weave instead of the original function
            def traced_wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            # Remove all annotations to prevent Weave from introspecting UnionType
            # This avoids the "'types.UnionType' object has no attribute '__name__'" error
            traced_wrapper.__annotations__ = {}

            # Create a signature without annotations to prevent introspection issues
            try:
                sig = inspect.signature(func)
                # Create a new signature with empty annotations
                params = []
                for param in sig.parameters.values():
                    # Create parameter without annotation
                    new_param = inspect.Parameter(
                        param.name,
                        param.kind,
                        default=param.default,
                        annotation=inspect.Parameter.empty,
                    )
                    params.append(new_param)
                new_sig = inspect.Signature(params, return_annotation=inspect.Signature.empty)
                traced_wrapper.__signature__ = new_sig
            except (ValueError, TypeError):
                # If signature introspection fails, just proceed without it
                pass

            # Use Weave's op decorator with explicit name to avoid signature introspection
            traced_func = weave.op(name=func.__name__)(traced_wrapper)
            result = traced_func(*args, **kwargs)

            logger.debug(
                "tool_traced", function=func.__name__, message="Tool execution traced successfully"
            )

            return result

        except Exception as e:
            logger.error(
                "tool_trace_failed",
                function=func.__name__,
                error=str(e),
                message="Failed to trace tool execution, executing without trace",
            )
            # Fallback: execute function without tracing
            return func(*args, **kwargs)

    return wrapper


def log_iteration_metrics(agent_name: str, iteration: int, metrics: dict[str, Any]) -> None:
    """
    Log structured metrics for iterative improvement loops.

    This function logs metrics during agent iteration cycles (e.g., critique
    and regenerate loops) to track quality progression, token usage, and
    improvement over iterations.

    Args:
        agent_name: Name of the agent performing iterations
        iteration: Current iteration number (0-indexed or 1-indexed)
        metrics: Dictionary of metrics to log (scores, tokens, improvements, etc.)

    Example:
        >>> log_iteration_metrics(
        >>>     agent_name="experience_optimizer",
        >>>     iteration=2,
        >>>     metrics={
        >>>         "quality_score": 87.5,
        >>>         "tokens_used": 1500,
        >>>         "improvement_delta": 12.3,
        >>>         "issues_remaining": 1
        >>>     }
        >>> )

    Notes:
        - Metrics are logged to both Weave and structlog
        - Supports any JSON-serializable values in metrics dict
        - Safe to call even if Weave is not initialized
    """
    weave = _lazy_import_weave()

    # Always log to structlog for local observability
    logger.info("iteration_metrics", agent=agent_name, iteration=iteration, **metrics)

    # If Weave is initialized, also log to Weave
    if _weave_initialized and weave is not None:
        try:
            # Prepare metrics with context
            weave_metrics = {"agent_name": agent_name, "iteration": iteration, **metrics}

            # Log to Weave
            weave.log(weave_metrics)

            logger.debug(
                "metrics_logged_to_weave",
                agent=agent_name,
                iteration=iteration,
                metrics_count=len(metrics),
            )

        except Exception as e:
            logger.error(
                "weave_log_failed",
                agent=agent_name,
                iteration=iteration,
                error=str(e),
                message="Failed to log metrics to Weave",
            )


def is_weave_enabled() -> bool:
    """
    Check if Weave observability is currently enabled and initialized.

    Returns:
        bool: True if Weave is initialized and ready to use, False otherwise

    Example:
        >>> if is_weave_enabled():
        >>>     print("Observability is active")
    """
    return _weave_initialized


def get_weave_project_url() -> str | None:
    """
    Get the URL to the Weave dashboard for this project.

    Returns:
        str: URL to the Weave dashboard, or None if not initialized

    Example:
        >>> url = get_weave_project_url()
        >>> if url:
        >>>     print(f"View traces at: {url}")
    """
    if not _weave_initialized:
        return None

    weave = _lazy_import_weave()
    if weave is None:
        return None

    try:
        # Get project info from Weave context
        # Note: This is a placeholder - actual implementation depends on Weave API
        entity = os.getenv("WANDB_ENTITY", "your-username")
        project = os.getenv("WANDB_PROJECT", "resume-tailor-agents")
        return f"https://wandb.ai/{entity}/{project}"
    except Exception:
        return None
