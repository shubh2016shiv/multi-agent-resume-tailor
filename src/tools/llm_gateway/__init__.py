"""Shared LLM entry points for tool-layer judgment calls."""

from .embedding_similarity import cosine_similarity, embed_texts, max_similarity
from .review_requests import request_review
from .structured_output import request_structured_output

__all__ = [
    "cosine_similarity",
    "embed_texts",
    "max_similarity",
    "request_review",
    "request_structured_output",
]
