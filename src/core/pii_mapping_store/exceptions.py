"""Exceptions for the PII mapping store."""


class PiiMappingStoreError(RuntimeError):
    """Base error for the PII mapping store."""


class RedisUnavailableError(PiiMappingStoreError):
    """Raised when services.redis_url is unset or the Redis client cannot connect.

    The PII mapping path fails closed: it never falls back to an in-memory or
    skipped store, because losing the mapping would leak placeholders into the
    final resume or, worse, drop the masking guarantee.
    """


class RedactionNotCompletedError(PiiMappingStoreError):
    """Raised when a run's mapping is missing or expired when it was required.

    Signals that LLM extraction was attempted before redaction stored a mapping,
    or that rehydration ran after the key had already expired or been deleted.
    """


class UnredactedExtractionInputError(PiiMappingStoreError):
    """Raised when text bound for the LLM still contains a run's real PII.

    The redaction record exists, but the supplied extraction input was not the
    redacted text: a real PII value survived in it. The path fails closed rather
    than let raw PII reach the model.
    """
