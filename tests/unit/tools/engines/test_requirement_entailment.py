"""Contracts for bounded requirement-entailment judgment."""

from unittest.mock import patch

from src.tools.engines.requirement_entailment.entailment_judge import (
    EntailmentVerdict,
    RequirementEntailmentResponse,
    judge_requirement_entailment,
)


def test_entailment_judge_accepts_entailed_quote_present_in_resume() -> None:
    """An entailed verdict is accepted only with a quote present in resume text."""
    resume_text = "Led regulated clinical workflow modernization across three hospitals."
    response = RequirementEntailmentResponse(
        verdict=EntailmentVerdict.ENTAILED,
        supporting_quote="regulated clinical workflow modernization",
        reasoning="The quote directly supports the requirement.",
    )

    with patch(
        "src.tools.engines.requirement_entailment.entailment_judge.request_structured_output",
        return_value=response,
    ):
        verdict = judge_requirement_entailment(
            "Experience managing regulated clinical workflows",
            resume_text,
        )

    assert verdict is EntailmentVerdict.ENTAILED


def test_entailment_judge_rejects_hallucinated_supporting_quote() -> None:
    """A model cannot pass entailment with a quote absent from resume text."""
    response = RequirementEntailmentResponse(
        verdict=EntailmentVerdict.ENTAILED,
        supporting_quote="owned FDA audit remediation",
        reasoning="The model invented stronger evidence.",
    )

    with patch(
        "src.tools.engines.requirement_entailment.entailment_judge.request_structured_output",
        return_value=response,
    ):
        verdict = judge_requirement_entailment(
            "Experience managing regulated clinical workflows",
            "Led hospital workflow modernization.",
        )

    assert verdict is EntailmentVerdict.INCONCLUSIVE


def test_entailment_judge_accepts_not_supported_without_quote() -> None:
    """A not-supported verdict does not need a supporting quote."""
    response = RequirementEntailmentResponse(
        verdict=EntailmentVerdict.NOT_SUPPORTED,
        supporting_quote=None,
        reasoning="No evidence for the requested responsibility.",
    )

    with patch(
        "src.tools.engines.requirement_entailment.entailment_judge.request_structured_output",
        return_value=response,
    ):
        verdict = judge_requirement_entailment(
            "Experience influencing senior stakeholders",
            "Built Python services.",
        )

    assert verdict is EntailmentVerdict.NOT_SUPPORTED


def test_entailment_judge_fails_closed_when_gateway_fails() -> None:
    """Gateway errors return inconclusive rather than a guessed label."""
    with patch(
        "src.tools.engines.requirement_entailment.entailment_judge.request_structured_output",
        side_effect=RuntimeError("model unavailable"),
    ):
        verdict = judge_requirement_entailment("Python", "Built Python services.")

    assert verdict is EntailmentVerdict.INCONCLUSIVE


def test_entailment_judge_skips_blank_inputs() -> None:
    """Blank requirement or evidence never triggers an LLM call."""
    with patch(
        "src.tools.engines.requirement_entailment.entailment_judge.request_structured_output"
    ) as request:
        verdict = judge_requirement_entailment("", "Built Python services.")

    assert verdict is EntailmentVerdict.INCONCLUSIVE
    request.assert_not_called()
