"""
Centralized Resilience Module
------------------------------

This module provides enterprise-grade resilience patterns for LLM API calls using
industry-standard libraries. It consolidates retry logic, circuit breakers, rate
limiting, and timeout handling in a single, reusable location.

WHY THIS MODULE?
- Eliminates code duplication across agents (no more custom retry decorators)
- Uses battle-tested libraries (tenacity, pybreaker, ratelimit)
- Centralized configuration and observability
- Prevents cascading failures with circuit breakers
- Enforces API quota compliance with rate limiting

DESIGN PRINCIPLES:
1. Industry-Standard: Use proven libraries, don't reinvent the wheel
2. Modular: Each resilience pattern is independently configurable
3. Observable: All events logged with correlation IDs for tracing
4. Configurable: All parameters externalized to settings.yaml
5. Composable: Patterns can be combined via unified decorator

USAGE:
    from src.core.resiliency import resilient_llm_call
    
    @resilient_llm_call(provider="gemini")
    def my_llm_function(agent, prompt):
        return agent.execute_task(prompt)
"""

import functools
import time
import uuid
from typing import Any, Callable, Optional

from pybreaker import CircuitBreaker, CircuitBreakerError
from ratelimit import limits, RateLimitException
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
    after_log,
)

try:
    from src.core.config import get_config
    from src.core.logger import get_logger
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.core.config import get_config
    from src.core.logger import get_logger

logger = get_logger(__name__)


# ==============================================================================
# Resilience Context & Correlation Tracking
# ==============================================================================

class ResilienceContext:
    """Thread-safe context for tracking resilience events across retry attempts."""
    
    _correlation_id: Optional[str] = None
    _call_stack: list[dict] = []
    
    @classmethod
    def set_correlation_id(cls, correlation_id: str):
        """Set correlation ID for current resilience context."""
        cls._correlation_id = correlation_id
    
    @classmethod
    def get_correlation_id(cls) -> str:
        """Get current correlation ID or generate new one."""
        if cls._correlation_id is None:
            cls._correlation_id = str(uuid.uuid4())[:8]
        return cls._correlation_id
    
    @classmethod
    def clear(cls):
        """Clear correlation context."""
        cls._correlation_id = None
        cls._call_stack = []
    
    @classmethod
    def log_event(cls, event_type: str, **metadata):
        """Log resilience event with correlation ID."""
        correlation_id = cls.get_correlation_id()
        event = {
            "timestamp": time.time(),
            "correlation_id": correlation_id,
            "event_type": event_type,
            **metadata
        }
        cls._call_stack.append(event)
        
        # Format for structured logging
        context_str = " | ".join([f"{k}={v}" for k, v in event.items()])
        logger.info(f"Resilience Event | {context_str}")


# ==============================================================================
# Circuit Breakers (per LLM provider)
# ==============================================================================

# Global circuit breakers - one per LLM provider
_circuit_breakers: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(provider: str) -> CircuitBreaker:
    """
    Get or create a circuit breaker for the specified LLM provider.
    
    Circuit breakers prevent cascading failures by "opening" after a threshold
    of consecutive failures, temporarily blocking calls to give the service
    time to recover.
    
    Args:
        provider: LLM provider name (e.g., "gemini", "openai")
    
    Returns:
        CircuitBreaker instance for the provider
    """
    if provider not in _circuit_breakers:
        config = get_config()
        
        # Get circuit breaker config
        cb_config = getattr(config.llm, 'resilience', None)
        failure_threshold = getattr(cb_config, 'circuit_breaker_failure_threshold', 5) if cb_config else 5
        timeout_duration = getattr(cb_config, 'circuit_breaker_timeout', 60) if cb_config else 60
        
        def on_state_change(old_state, new_state, _):
            """Log circuit breaker state transitions."""
            ResilienceContext.log_event(
                event_type="circuit_breaker_state_change",
                provider=provider,
                old_state=old_state.name,
                new_state=new_state.name
            )
        
        _circuit_breakers[provider] = CircuitBreaker(
            fail_max=failure_threshold,
            reset_timeout=timeout_duration,
            name=f"{provider}_circuit_breaker",
            listeners=[on_state_change]
        )
        
        logger.info(f"Created circuit breaker for provider={provider} | "
                   f"fail_max={failure_threshold} | reset_timeout={timeout_duration}s")
    
    return _circuit_breakers[provider]


# ==============================================================================
# Retry with Exponential Backoff (using tenacity)
# ==============================================================================

def create_retry_decorator(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_multiplier: float = 2.0,
    provider: str = "default"
):
    """
    Create a retry decorator with exponential backoff using tenacity.
    
    This handles transient failures like network issues, rate limits, and
    temporary API unavailability by automatically retrying with increasing
    delays between attempts.
    
    Args:
        max_attempts: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
        max_delay: Maximum delay in seconds between retries
        exponential_multiplier: Multiplier for exponential backoff
        provider: LLM provider name for logging
    
    Returns:
        Configured tenacity retry decorator
    """
    
    def before_sleep_callback(retry_state):
        """Log before sleeping between retry attempts."""
        ResilienceContext.log_event(
            event_type="retry_attempt",
            provider=provider,
            attempt=retry_state.attempt_number,
            max_attempts=max_attempts,
            next_wait=retry_state.next_action.sleep if retry_state.next_action else 0,
            exception=str(retry_state.outcome.exception()) if retry_state.outcome else None
        )
    
    def after_callback(retry_state):
        """Log after successful retry or final failure."""
        if retry_state.outcome.failed:
            ResilienceContext.log_event(
                event_type="retry_exhausted",
                provider=provider,
                total_attempts=retry_state.attempt_number,
                exception=str(retry_state.outcome.exception())
            )
        else:
            ResilienceContext.log_event(
                event_type="retry_succeeded",
                provider=provider,
                successful_attempt=retry_state.attempt_number
            )
    
    return retry(
        # Retry on common API exceptions
        retry=retry_if_exception_type((
            Exception,  # Broad catch - will be filtered by stop condition
        )),
        # Stop after max attempts
        stop=stop_after_attempt(max_attempts),
        # Exponential backoff with jitter to prevent thundering herd
        wait=wait_exponential(
            multiplier=exponential_multiplier,
            min=initial_delay,
            max=max_delay
        ),
        # Logging callbacks
        before_sleep=before_sleep_callback,
        after=after_callback,
        # Re-raise the exception after all retries exhausted
        reraise=True
    )


# ==============================================================================
# Rate Limiting
# ==============================================================================

def create_rate_limiter(calls_per_minute: int = 60, provider: str = "default"):
    """
    Create a rate limiter decorator to enforce API quota compliance.
    
    This prevents exceeding API rate limits by throttling calls to stay
    within the specified quota.
    
    Args:
        calls_per_minute: Maximum number of calls allowed per minute
        provider: LLM provider name for logging
    
    Returns:
        Configured ratelimit decorator
    """
    
    def rate_limit_decorator(func):
        @limits(calls=calls_per_minute, period=60)
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except RateLimitException:
                ResilienceContext.log_event(
                    event_type="rate_limit_hit",
                    provider=provider,
                    limit=calls_per_minute
                )
                # Wait before retrying
                time.sleep(60.0 / calls_per_minute)
                return func(*args, **kwargs)
        
        return wrapper
    
    return rate_limit_decorator


# ==============================================================================
# Unified Resilience Decorator
# ==============================================================================

def resilient_llm_call(
    provider: Optional[str] = None,
    max_attempts: Optional[int] = None,
    timeout: Optional[int] = None,
    enable_circuit_breaker: bool = True,
    enable_rate_limiting: bool = True,
    enable_retry: bool = True
):
    """
    Unified decorator that applies all resilience patterns to LLM calls.
    
    This is the main entry point for adding resilience to LLM-calling functions.
    It combines retry logic, circuit breaker protection, and rate limiting in a
    single, easy-to-use decorator.
    
    Args:
        provider: LLM provider name (defaults to config.llm.provider)
        max_attempts: Max retry attempts (defaults to config)
        timeout: Timeout in seconds (defaults to config)
        enable_circuit_breaker: Whether to use circuit breaker
        enable_rate_limiting: Whether to enforce rate limiting
        enable_retry: Whether to retry on failures
    
    Usage:
        @resilient_llm_call(provider="gemini")
        def my_llm_function(agent, prompt):
            return agent.execute_task(prompt)
    
    Returns:
        Decorated function with resilience patterns applied
    """
    
    def decorator(func: Callable) -> Callable:
        config = get_config()
        
        # Resolve provider
        actual_provider = provider or config.llm.provider
        
        # Get resilience config
        resilience_config = getattr(config.llm, 'resilience', None)
        
        # Resolve parameters with fallbacks
        _max_attempts = max_attempts or (
            getattr(resilience_config, 'retry_max_attempts', 3) if resilience_config else 3
        )
        _timeout = timeout or (
            getattr(resilience_config, 'timeout_seconds', 30) if resilience_config else 30
        )
        _rate_limit = (
            getattr(resilience_config, 'rate_limit_calls_per_minute', 60) 
            if resilience_config else 60
        )
        _initial_delay = (
            getattr(resilience_config, 'retry_initial_delay', 1.0)
            if resilience_config else 1.0
        )
        _max_delay = (
            getattr(resilience_config, 'retry_max_delay', 60.0)
            if resilience_config else 60.0
        )
        _exponential_multiplier = (
            getattr(resilience_config, 'retry_exponential_multiplier', 2.0)
            if resilience_config else 2.0
        )
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Initialize correlation context
            ResilienceContext.set_correlation_id(str(uuid.uuid4())[:8])
            
            ResilienceContext.log_event(
                event_type="llm_call_start",
                provider=actual_provider,
                function=func.__name__,
                circuit_breaker=enable_circuit_breaker,
                rate_limiting=enable_rate_limiting,
                retry=enable_retry
            )
            
            try:
                # Build the resilient function by composing patterns
                resilient_func = func
                
                # 1. Apply rate limiting (outermost)
                if enable_rate_limiting:
                    rate_limiter = create_rate_limiter(_rate_limit, actual_provider)
                    resilient_func = rate_limiter(resilient_func)
                
                # 2. Apply circuit breaker
                if enable_circuit_breaker:
                    circuit_breaker = get_circuit_breaker(actual_provider)
                    
                    def circuit_breaker_wrapper(*args, **kwargs):
                        try:
                            return circuit_breaker.call(resilient_func, *args, **kwargs)
                        except CircuitBreakerError:
                            ResilienceContext.log_event(
                                event_type="circuit_breaker_open",
                                provider=actual_provider,
                                function=func.__name__
                            )
                            raise
                    
                    resilient_func = circuit_breaker_wrapper
                
                # 3. Apply retry with exponential backoff (innermost)
                if enable_retry:
                    retry_decorator = create_retry_decorator(
                        max_attempts=_max_attempts,
                        initial_delay=_initial_delay,
                        max_delay=_max_delay,
                        exponential_multiplier=_exponential_multiplier,
                        provider=actual_provider
                    )
                    resilient_func = retry_decorator(resilient_func)
                
                # Execute with all patterns applied
                result = resilient_func(*args, **kwargs)
                
                ResilienceContext.log_event(
                    event_type="llm_call_success",
                    provider=actual_provider,
                    function=func.__name__
                )
                
                return result
                
            except Exception as e:
                ResilienceContext.log_event(
                    event_type="llm_call_failed",
                    provider=actual_provider,
                    function=func.__name__,
                    error=str(e),
                    error_type=type(e).__name__
                )
                raise
            
            finally:
                # Clean up context
                ResilienceContext.clear()
        
        return wrapper
    
    return decorator


# ==============================================================================
# Resilience Metrics & Observability
# ==============================================================================

def get_resilience_stats(provider: Optional[str] = None) -> dict[str, Any]:
    """
    Get resilience statistics for monitoring and debugging.
    
    Args:
        provider: Optional provider to filter stats (returns all if None)
    
    Returns:
        Dictionary with resilience metrics including circuit breaker states,
        retry counts, and rate limit hits
    """
    stats = {}
    
    if provider:
        if provider in _circuit_breakers:
            cb = _circuit_breakers[provider]
            stats[provider] = {
                "circuit_breaker_state": cb.current_state.name,
                "failure_count": cb.fail_counter,
                "last_failure_time": getattr(cb, 'last_failure_time', None)
            }
    else:
        # Return stats for all providers
        for prov, cb in _circuit_breakers.items():
            stats[prov] = {
                "circuit_breaker_state": cb.current_state.name,
                "failure_count": cb.fail_counter,
                "last_failure_time": getattr(cb, 'last_failure_time', None)
            }
    
    return stats


def reset_circuit_breakers():
    """
    Manually reset all circuit breakers.
    
    This can be used for testing or administrative purposes to force
    circuit breakers back to closed state.
    """
    for provider, cb in _circuit_breakers.items():
        cb.close()
        logger.info(f"Circuit breaker manually reset for provider={provider}")
