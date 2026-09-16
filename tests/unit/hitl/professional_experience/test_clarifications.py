"""Tests for turning the LLM's fact-gap verdict into candidate questions.

The join is pure code, so these run without an LLM. What they pin down is the
design decision that replaced a pile of defensive checks: a gap is nested and
non-optional, so "a category and a question exist exactly when candidate input
is needed" is enforced by the schema. The old shape allowed a finding that
claimed input was needed but carried no question, and such findings were
dropped with a warning -- meaning the candidate silently never got asked.
"""

from datetime import date

import pytest
from pydantic import ValidationError

from src.agents.professional_experience.models import ExperienceBulletRewrite
from src.data_models.resume import Experience
from src.hitl.professional_experience.clarifications import clarifications_from_findings
from src.hitl.professional_experience.models import (
    CandidateFactGap,
    ExperienceBulletFactGapFinding,
    ExperienceBulletMissingFactCategory,
)

SHIPPED = "Migrated the billing service to the new platform."


def _experience() -> Experience:
    return Experience(
        experience_id="experience-1",
        job_title="Backend Engineer",
        company_name="Acme Corp",
        start_date=date(2023, 1, 1),
        end_date=None,
        location=None,
        description="Owned the billing service.",
        achievements=[SHIPPED],
    )


def _rewrite(bullet_id: str) -> ExperienceBulletRewrite:
    return ExperienceBulletRewrite(
        bullet_id=bullet_id,
        source_bullet="Worked on billing.",
        rewritten_bullet=SHIPPED,
        ownership_level="owned",
    )


def _gap() -> CandidateFactGap:
    return CandidateFactGap(
        gap_category=ExperienceBulletMissingFactCategory.RESULT,
        missing_fact_summary="No measurable outcome is stated.",
        why_flagged="The result appears nowhere in the role evidence.",
        question="What changed after the migration shipped?",
    )


def _join(findings: list[ExperienceBulletFactGapFinding]):
    return clarifications_from_findings(
        experience=_experience(),
        shipped_bullets=[SHIPPED],
        rewritten_bullets=[_rewrite("b::1")],
        findings=findings,
    )


def test_a_gap_carries_role_identity_into_the_question() -> None:
    """The candidate's question must know which role's bullet it belongs to."""
    findings = [
        ExperienceBulletFactGapFinding(bullet_id="b::1", current_bullet=SHIPPED, gap=_gap())
    ]

    [clarification] = _join(findings)

    assert clarification.bullet_id == "b::1"
    assert clarification.company_name == "Acme Corp"
    assert clarification.job_title == "Backend Engineer"
    assert clarification.start_date == "2023-01-01"
    assert clarification.bullet == SHIPPED
    # The gap's own fields survive the join unrenamed -- they are the same fields.
    assert clarification.question == _gap().question
    assert clarification.gap_category is ExperienceBulletMissingFactCategory.RESULT
    assert not clarification.is_answered


def test_a_bullet_with_no_gap_asks_nothing() -> None:
    """A null gap is the whole 'ship as written' signal; no boolean can disagree."""
    findings = [ExperienceBulletFactGapFinding(bullet_id="b::1", current_bullet=SHIPPED, gap=None)]

    assert _join(findings) == []


def test_a_finding_naming_an_unknown_bullet_is_dropped() -> None:
    """A hallucinated bullet_id is the one failure the schema cannot prevent."""
    findings = [
        ExperienceBulletFactGapFinding(
            bullet_id="does-not-exist", current_bullet=SHIPPED, gap=_gap()
        )
    ]

    assert _join(findings) == []


@pytest.mark.parametrize("missing_field", ["gap_category", "question"])
def test_a_partial_gap_cannot_be_constructed(missing_field: str) -> None:
    """The old silent-drop path is gone because this state is unrepresentable.

    Previously the LLM could claim a bullet needed input while omitting the
    question, and the join discarded it with a log line nobody read.
    """
    fields = _gap().model_dump()
    fields.pop(missing_field)

    with pytest.raises(ValidationError):
        CandidateFactGap(**fields)
