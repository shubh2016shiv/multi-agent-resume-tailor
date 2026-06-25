"""Embedding-based text similarity via the configured embedding model.

A cross-cutting provider call (like structured_output.py and review_requests.py):
it turns text into vectors and compares them. It returns scores only -- the
decision of which score counts as a match (the threshold) belongs to the caller.
"""

import math

import litellm

# One value today; promote to LLMSettings if a second embedding model is ever needed.
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def embed_texts(texts: list[str], model: str = DEFAULT_EMBEDDING_MODEL) -> list[list[float]]:
    """Return one embedding vector per input text, in order.

    Expects a non-empty list of non-empty strings. Returns vectors of equal
    dimension, aligned by index to `texts`. Raises on provider failure.
    """
    response = litellm.embedding(model=model, input=texts)
    return [item["embedding"] for item in response["data"]]


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """Return cosine similarity in [-1.0, 1.0] for two equal-length vectors.

    Expects non-zero vectors of equal length (real embeddings always are).
    Pure function, no I/O.
    """
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b, strict=True))
    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))
    return dot_product / (magnitude_a * magnitude_b)


def max_similarity(query_text: str, candidate_texts: list[str]) -> float:
    """Return the highest cosine similarity between query and any candidate.

    Expects a non-empty query. Embeds query+candidates in one batch and returns
    the best score, or 0.0 when there are no candidates. No threshold decision
    here -- the caller owns the cutoff.
    """
    if not candidate_texts:
        return 0.0
    query_vector, *candidate_vectors = embed_texts([query_text, *candidate_texts])
    return max(
        cosine_similarity(query_vector, candidate_vector)
        for candidate_vector in candidate_vectors
    )
