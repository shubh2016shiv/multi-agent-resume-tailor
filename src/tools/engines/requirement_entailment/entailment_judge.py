"""Bounded LLM judge for requirement-to-resume entailment."""

from enum import StrEnum

from pydantic import BaseModel, Field

from src.core.logger import get_logger
from src.core.prompt_catalog import load_tool_prompt
from src.tools.llm_gateway import request_structured_output

logger = get_logger(__name__)

ENGINE_ID = "requirement_entailment_judge"
PROMPT_VERSION = "requirement_entailment.v1"
ENTAILMENT_RUBRIC = load_tool_prompt("requirement_entailment/entailment_judge.md")

# A quality gate must be as reproducible as possible: this bounded judgment runs at
# temperature 0, overriding the higher default agent temperature used for generation.
_JUDGE_TEMPERATURE = 0.0


class EntailmentVerdict(StrEnum):
    """Closed labels returned by the bounded semantic judge."""

    ENTAILED = "entailed"
    NOT_SUPPORTED = "not_supported"
    INCONCLUSIVE = "inconclusive"


class RequirementEntailmentResponse(BaseModel):
    """Structured model response for one requirement-entailment judgment."""

    verdict: EntailmentVerdict = Field(description="Closed entailment label.")
    supporting_quote: str | None = Field(
        default=None,
        description="Exact resume quote required when verdict is entailed.",
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of the closed label.",
    )


def judge_requirement_entailment(
    requirement_text: str,
    resume_text: str,
) -> EntailmentVerdict:
    """Ask the configured LLM whether resume evidence entails one requirement.

    Expects a single requirement and rendered resume text. Returns only a closed
    verdict. Any gateway failure, blank input, or unverifiable supporting quote
    returns INCONCLUSIVE so callers can fail safely or request human review.
    """
    if not requirement_text.strip() or not resume_text.strip():
        return EntailmentVerdict.INCONCLUSIVE
    try:
        response = request_structured_output(
            RequirementEntailmentResponse,
            ENTAILMENT_RUBRIC,
            _build_payload(requirement_text, resume_text),
            temperature=_JUDGE_TEMPERATURE,
        )
    except Exception as error:  # noqa: BLE001 -- semantic gate must fail closed
        logger.warning(
            "requirement_entailment_unavailable",
            engine_id=ENGINE_ID,
            error=str(error),
        )
        return EntailmentVerdict.INCONCLUSIVE
    return _validated_verdict(response, resume_text)


def _build_payload(requirement_text: str, resume_text: str) -> str:
    """Return the bounded judgment payload with prompt-version context."""
    return (
        f"PROMPT_VERSION: {PROMPT_VERSION}\n\n"
        f"REQUIREMENT:\n{requirement_text.strip()}\n\n"
        f"RESUME_TEXT:\n{resume_text.strip()}"
    )


def _validated_verdict(
    response: RequirementEntailmentResponse,
    resume_text: str,
) -> EntailmentVerdict:
    """Return the verdict only after validating required quote evidence."""
    if response.verdict is not EntailmentVerdict.ENTAILED:
        return response.verdict
    if not response.supporting_quote:
        return EntailmentVerdict.INCONCLUSIVE
    if not _quote_exists(response.supporting_quote, resume_text):
        return EntailmentVerdict.INCONCLUSIVE
    return EntailmentVerdict.ENTAILED


def _quote_exists(quote: str, resume_text: str) -> bool:
    """Return whether the model quote is present in the supplied resume text."""
    normalized_quote = " ".join(quote.split())
    normalized_resume = " ".join(resume_text.split())
    return quote in resume_text or normalized_quote in normalized_resume
