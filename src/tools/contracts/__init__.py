"""Shared contracts used across tool-layer capabilities."""

from .review import Confidence, Location, ReviewComment, ReviewResult, Section, Severity

__all__ = [
    "Confidence",
    "Location",
    "ReviewComment",
    "ReviewResult",
    "Section",
    "Severity",
]
