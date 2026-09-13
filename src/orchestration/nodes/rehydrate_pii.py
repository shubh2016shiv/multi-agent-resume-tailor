"""Stage 6 PII rehydration node.

Plain code, no agent: after all LLM work and QA are done, the placeholders that
masked PII (e.g. [PERSON_1]) are swapped back to real values. Runs on every path
out of QA -- before the render gate -- so the returned resume carries real values
whether or not it goes on to render. The mapping comes from the run-scoped Redis
store; the LLM never sees it.
"""

from typing import Any

from src.core.logger import get_logger
from src.core.pii_mapping_store import load_pii_mapping
from src.core.settings import get_config
from src.data_models.resume import Resume
from src.orchestration.nodes._stage import pipeline_stage
from src.orchestration.state import ResumeEnhancementPipelineState, require

logger = get_logger(__name__)


@pipeline_stage("rehydrate_pii")
def rehydrate_pii(state: ResumeEnhancementPipelineState) -> dict:
    """Restore real PII into the assembled resume using the run's stored mapping.

    Reads: run_id, optimized_resume (its final_resume).
    Writes: optimized_resume, with placeholders in final_resume replaced by real values.
    Returns: partial state; an unchanged optimized_resume when redaction found no PII.
    Runs on every path out of QA, before the render gate.
    Raises: RedactionNotCompletedError if the mapping is missing or expired.
    """
    optimized_resume = require(state["optimized_resume"], "optimized_resume")
    if not get_config().feature_flags.enable_pii_redaction:
        # PII pipeline disabled: nothing was redacted, so there is nothing to restore.
        logger.debug("pii_rehydration_skipped", reason="pii_redaction_disabled")
        return {"optimized_resume": optimized_resume}
    mapping = load_pii_mapping(state["run_id"])
    if not mapping:
        logger.info("pii_rehydration_skipped", reason="mapping_empty")
        return {"optimized_resume": optimized_resume}
    restored_fields = _replace_in_structure(optimized_resume.final_resume.model_dump(), mapping)
    rehydrated_resume = Resume.model_validate(restored_fields)
    logger.info("pii_rehydration_completed", placeholder_count=len(mapping))
    return {
        "optimized_resume": optimized_resume.model_copy(update={"final_resume": rehydrated_resume})
    }


def _replace_in_structure(value: Any, mapping: dict[str, str]) -> Any:
    """Recursively replace placeholders in every string leaf of a nested structure.

    Walks dicts and lists; replaces placeholders only inside str leaves and leaves
    every other type (int, bool, None) untouched.
    """
    if isinstance(value, str):
        return _replace_placeholders(value, mapping)
    if isinstance(value, dict):
        return {key: _replace_in_structure(item, mapping) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_in_structure(item, mapping) for item in value]
    return value


def _replace_placeholders(text: str, mapping: dict[str, str]) -> str:
    """Replace each placeholder key in text with its real value.

    Replaces longest keys first so a placeholder is never a prefix of another
    (e.g. [PERSON_1] vs [PERSON_10]). Unknown placeholders are left unchanged.
    """
    for placeholder in sorted(mapping, key=len, reverse=True):
        text = text.replace(placeholder, mapping[placeholder])
    return text
