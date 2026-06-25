"""Public facade for the PII mapping store package.

    from src.core.pii_mapping_store import (
        save_pii_mapping, assert_redaction_completed, load_pii_mapping, delete_pii_mapping,
    )

Owns ephemeral, run-scoped storage of the PII placeholder->value mapping in Redis,
between redaction (which produces it) and final rehydration (which consumes it).
"""

from src.core.pii_mapping_store.exceptions import (
    PiiMappingStoreError,
    RedactionNotCompletedError,
    RedisUnavailableError,
    UnredactedExtractionInputError,
)
from src.core.pii_mapping_store.redis_pii_mapping_store import (
    assert_extraction_input_redacted,
    delete_pii_mapping,
    load_pii_mapping,
    save_pii_mapping,
)

__all__ = [
    "PiiMappingStoreError",
    "RedactionNotCompletedError",
    "RedisUnavailableError",
    "UnredactedExtractionInputError",
    "assert_extraction_input_redacted",
    "delete_pii_mapping",
    "load_pii_mapping",
    "save_pii_mapping",
]
