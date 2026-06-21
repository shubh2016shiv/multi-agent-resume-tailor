"""Shared LLM entry points for tool-layer judgment calls."""

from .review_requests import request_review
from .structured_output import request_structured_output

__all__ = [
    "request_review",
    "request_structured_output",
]
