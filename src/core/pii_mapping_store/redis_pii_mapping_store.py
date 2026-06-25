"""Ephemeral Redis storage of one run's PII placeholder mapping.

This is run-state, not a cache: the mapping (placeholder -> real value) is the
sensitive bridge between redaction and final rehydration. It is stored with a TTL,
keyed by run_id, deleted on every terminal path, and never logged by value. The
path fails closed: if Redis is unreachable, callers raise rather than proceed.
"""

import json

import redis
from redis.exceptions import RedisError

from src.core.logger import get_logger
from src.core.pii_mapping_store.exceptions import (
    RedactionNotCompletedError,
    RedisUnavailableError,
    UnredactedExtractionInputError,
)
from src.core.settings import get_config

logger = get_logger(__name__)

_KEY_PREFIX = "resume_tailor:pii_mapping:"


def save_pii_mapping(run_id: str, mapping: dict[str, str]) -> None:
    """Store a run's placeholder->value mapping in Redis with a TTL.

    Records {"mapping": ..., "redaction_completed": true} so a later extraction
    can confirm redaction ran even when the mapping itself is empty (no PII found).

    Raises:
        RedisUnavailableError: if Redis is unset or unreachable (fail closed).
    """
    client = _get_redis_client()
    payload = json.dumps({"mapping": mapping, "redaction_completed": True})
    ttl_seconds = get_config().services.pii_mapping_ttl_seconds
    try:
        client.set(_mapping_key(run_id), payload, ex=ttl_seconds)
    except RedisError as error:
        raise RedisUnavailableError(f"Failed to store PII mapping for run {run_id}") from error
    logger.info("pii_mapping_stored", run_id=run_id, mapping_count=len(mapping))


def assert_extraction_input_redacted(run_id: str, candidate_text: str) -> None:
    """Verify text bound for the LLM contains no raw PII for this run.

    Closes the extraction safety contract: the agent supplies the extraction
    argument, so checking a "redaction ran" flag is not enough. This loads the
    run's mapping and rejects the call if any real PII value still appears in
    candidate_text. Only counts are surfaced, never the values themselves.

    Raises:
        RedactionNotCompletedError: if no redaction record exists (redaction did not run).
        UnredactedExtractionInputError: if a real PII value survives in candidate_text.
        RedisUnavailableError: if Redis is unset or unreachable.
    """
    mapping = _load_record(run_id).get("mapping", {})
    leaked_count = sum(1 for value in mapping.values() if value and value in candidate_text)
    if leaked_count:
        raise UnredactedExtractionInputError(
            f"Extraction input still contains {leaked_count} raw PII value(s) for run {run_id}"
        )


def load_pii_mapping(run_id: str) -> dict[str, str]:
    """Return the placeholder->value mapping for run_id, for final rehydration.

    Returns:
        The mapping, or {} when redaction found no PII.

    Raises:
        RedactionNotCompletedError: if the record is missing or expired.
        RedisUnavailableError: if Redis is unset or unreachable.
    """
    return _load_record(run_id).get("mapping", {})


def delete_pii_mapping(run_id: str) -> None:
    """Delete the run's PII mapping key. Idempotent and best-effort.

    Cleanup must never mask the real error on a failing run, so a down or unset
    Redis is logged and swallowed here rather than raised.
    """
    try:
        _get_redis_client().delete(_mapping_key(run_id))
    except (RedisUnavailableError, RedisError) as error:
        logger.warning("pii_mapping_delete_failed", run_id=run_id, error=str(error))
        return
    logger.info("pii_mapping_deleted", run_id=run_id)


def _load_record(run_id: str) -> dict:
    """Fetch and parse the stored JSON record for run_id.

    Returns a dict with keys 'mapping' and 'redaction_completed'.

    Raises:
        RedactionNotCompletedError: if the key is missing or expired.
        RedisUnavailableError: if Redis is unset or unreachable.
    """
    client = _get_redis_client()
    try:
        raw_record = client.get(_mapping_key(run_id))
    except RedisError as error:
        raise RedisUnavailableError(f"Failed to read PII mapping for run {run_id}") from error
    if raw_record is None:
        raise RedactionNotCompletedError(f"No PII mapping record for run {run_id}")
    return json.loads(raw_record)


def _get_redis_client() -> redis.Redis:
    """Build a Redis client from services.redis_url (decoded string responses).

    Raises:
        RedisUnavailableError: if services.redis_url is unset (fail closed).
    """
    redis_url = get_config().services.redis_url
    if not redis_url:
        raise RedisUnavailableError("services.redis_url is not configured")
    return redis.from_url(redis_url, decode_responses=True)


def _mapping_key(run_id: str) -> str:
    """Return the Redis key for a run's PII mapping."""
    return f"{_KEY_PREFIX}{run_id}"
