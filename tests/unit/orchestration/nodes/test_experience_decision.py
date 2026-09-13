"""Contracts for the role-scoped experience rewrite decision.

_decide_role_rewrite_outcome has four exits and duplicated check blocks (the
first proposal and the repaired one are evaluated the same way). These tests
pin each exit's observable behaviour -- which bullets ship, whether a repair
was requested, and what the optimization note says -- so the duplication can
be refactored without changing what the node produces.

Every LLM boundary is patched: the rewrite agent (_request_role_rewrite_proposal),
the semantic rewrite review (audit_experience_rewrite_quality), and the numeric
truth check (detect_claim_inflation).
"""

from datetime import date
from unittest.mock import patch

from src.agents.professional_experience.models import (
    ExperienceBulletRewrite,
    ExperienceRewriteProposal,
)
from src.data_models.resume import Experience, Resume
from src.hitl.professional_experience.models import build_experience_bullet_id
from src.orchestration.nodes.experience import _decide_role_rewrite_outcome
from src.tools.contracts import (
    Confidence,
    Location,
    ReviewComment,
    ReviewResult,
    Section,
    Severity,
)

EXPERIENCE_MODULE = "src.orchestration.nodes.experience"

SOURCE_BULLETS = ["Worked on backend services.", "Helped with deployments."]


def _experience(achievements: list[str]) -> Experience:
    return Experience(
        experience_id="exp-1",
        job_title="Engineer",
        company_name="Acme",
        start_date=date(2020, 1, 1),
        end_date=None,
        is_current_position=True,
        location=None,
        description="Built backend services for claims operations.",
        achievements=achievements,
        skills_used=["Python"],
    )


def _resume(experience: Experience) -> Resume:
    return Resume(
        full_name="Jane Doe",
        email="jane.doe@example.com",
        phone_number=None,
        location=None,
        website_or_portfolio=None,
        professional_summary="",
        work_experience=[experience],
        education=[],
        skills=[],
        certifications=[],
        languages=[],
    )


def _proposal(
    experience: Experience,
    rewritten_bullets: list[str],
    *,
    bullet_ids: list[str] | None = None,
) -> ExperienceRewriteProposal:
    """Build a proposal, by default with correct 1:1 bullet ids in source order."""
    ids = bullet_ids or [
        build_experience_bullet_id(experience, index) for index, _ in enumerate(rewritten_bullets)
    ]
    return ExperienceRewriteProposal(
        rewritten_bullets=[
            ExperienceBulletRewrite(
                bullet_id=ids[index],
                source_bullet=(
                    experience.achievements[index]
                    if index < len(experience.achievements)
                    else "unmatched source bullet"
                ),
                rewritten_bullet=text,
                supporting_role_evidence=["claims operations"],
                ownership_level="executed",
                clarifying_question=None,
            )
            for index, text in enumerate(rewritten_bullets)
        ],
        optimization_notes="",
    )


def _comment(severity: Severity, message: str = "Bullet is vague") -> ReviewComment:
    return ReviewComment(
        engine_id="experience_rewrite_quality_auditor",
        message=message,
        quoted_text="Worked on backend services.",
        location=Location(
            section=Section.EXPERIENCE,
            bullet_index=0,
            item_id=None,
            character_span=None,
        ),
        severity=severity,
        confidence=Confidence.HIGH,
        advice="Add a concrete result",
        proposed_rewrite=None,
    )


def _review(*comments: ReviewComment) -> ReviewResult:
    return ReviewResult(comments=list(comments), summary="", score=None)


def _clean_inflation() -> ReviewResult:
    """No numeric claim was invented -- the deterministic truth floor passes."""
    return _review()


def test_clean_first_proposal_ships_without_a_repair() -> None:
    """A truthful, substantive first rewrite ships as-is and costs no repair call."""
    experience = _experience(SOURCE_BULLETS)
    proposal = _proposal(experience, ["Built Python services.", "Ran deployments."])

    with (
        patch(f"{EXPERIENCE_MODULE}.detect_claim_inflation", return_value=_clean_inflation()),
        patch(f"{EXPERIENCE_MODULE}.audit_experience_rewrite_quality", return_value=_review()),
        patch(f"{EXPERIENCE_MODULE}._request_role_rewrite_proposal") as request_rewrite,
    ):
        decision = _decide_role_rewrite_outcome(
            proposal,
            "context",
            _resume(experience),
            experience,
            run_id="test-run",
        )

    request_rewrite.assert_not_called()
    shipped = decision.finalized_section.optimized_experiences[0]
    assert shipped.achievements == ["Built Python services.", "Ran deployments."]
    assert decision.selected_rewrite_proposal is proposal
    assert "Rewrote bullets with grounded role evidence" in (
        decision.finalized_section.optimization_notes
    )


def test_flawed_first_proposal_is_repaired_once_and_the_repair_ships() -> None:
    """A MAJOR finding earns exactly one repair; a clean repair is what ships."""
    experience = _experience(SOURCE_BULLETS)
    first = _proposal(experience, ["Vague thing.", "Another vague thing."])
    repaired = _proposal(experience, ["Built Python services.", "Ran deployments."])

    with (
        patch(f"{EXPERIENCE_MODULE}.detect_claim_inflation", return_value=_clean_inflation()),
        patch(
            f"{EXPERIENCE_MODULE}.audit_experience_rewrite_quality",
            side_effect=[_review(_comment(Severity.MAJOR)), _review()],
        ),
        patch(
            f"{EXPERIENCE_MODULE}._request_role_rewrite_proposal",
            return_value=repaired,
        ) as request_rewrite,
    ):
        decision = _decide_role_rewrite_outcome(
            first,
            "context",
            _resume(experience),
            experience,
            run_id="test-run",
        )

    assert request_rewrite.call_count == 1
    shipped = decision.finalized_section.optimized_experiences[0]
    assert shipped.achievements == ["Built Python services.", "Ran deployments."]
    assert decision.selected_rewrite_proposal is repaired
    assert "Repaired once with grounded rewrite feedback" in (
        decision.finalized_section.optimization_notes
    )


def test_repair_that_ships_surfaces_remaining_thin_bullets_to_the_candidate() -> None:
    """MINOR findings that survive the repair become a candidate follow-up note."""
    experience = _experience(SOURCE_BULLETS)
    first = _proposal(experience, ["Vague thing.", "Another vague thing."])
    repaired = _proposal(experience, ["Built Python services.", "Ran deployments."])

    with (
        patch(f"{EXPERIENCE_MODULE}.detect_claim_inflation", return_value=_clean_inflation()),
        patch(
            f"{EXPERIENCE_MODULE}.audit_experience_rewrite_quality",
            side_effect=[
                _review(_comment(Severity.MAJOR)),
                _review(_comment(Severity.MINOR, "Still no measurable result")),
            ],
        ),
        patch(f"{EXPERIENCE_MODULE}._request_role_rewrite_proposal", return_value=repaired),
    ):
        decision = _decide_role_rewrite_outcome(
            first,
            "context",
            _resume(experience),
            experience,
            run_id="test-run",
        )

    notes = decision.finalized_section.optimization_notes
    assert "BULLETS NEEDING YOUR INPUT" in notes
    assert "Still no measurable result" in notes


def test_first_rewrite_is_kept_when_the_repair_makes_it_worse() -> None:
    """A first rewrite that was already safe survives a repair that regresses.

    The first proposal is truthful with only a MINOR finding (which still earns
    a repair), and the repair comes back with a MAJOR one. The safe first
    rewrite ships instead, carrying its own follow-up note.
    """
    experience = _experience(SOURCE_BULLETS)
    first = _proposal(experience, ["Built Python services.", "Ran deployments."])
    repaired = _proposal(experience, ["Overclaimed thing.", "Another overclaim."])

    with (
        patch(f"{EXPERIENCE_MODULE}.detect_claim_inflation", return_value=_clean_inflation()),
        patch(
            f"{EXPERIENCE_MODULE}.audit_experience_rewrite_quality",
            side_effect=[
                _review(_comment(Severity.MINOR, "Could be more concrete")),
                _review(_comment(Severity.MAJOR, "Ownership inflated")),
            ],
        ),
        patch(f"{EXPERIENCE_MODULE}._request_role_rewrite_proposal", return_value=repaired),
    ):
        decision = _decide_role_rewrite_outcome(
            first,
            "context",
            _resume(experience),
            experience,
            run_id="test-run",
        )

    shipped = decision.finalized_section.optimized_experiences[0]
    assert shipped.achievements == ["Built Python services.", "Ran deployments."]
    assert decision.selected_rewrite_proposal is first
    assert "Kept the first grounded rewrite" in decision.finalized_section.optimization_notes


def test_source_bullets_are_preserved_when_no_rewrite_clears_the_truth_floor() -> None:
    """When both the first rewrite and the repair invent numbers, nothing ships."""
    experience = _experience(SOURCE_BULLETS)
    first = _proposal(experience, ["Cut latency by 40%.", "Served 2M users."])
    repaired = _proposal(experience, ["Cut latency by 30%.", "Served 1M users."])

    with (
        patch(
            f"{EXPERIENCE_MODULE}.detect_claim_inflation",
            return_value=_review(_comment(Severity.MAJOR, "Unsupported figure 40%")),
        ),
        patch(f"{EXPERIENCE_MODULE}.audit_experience_rewrite_quality", return_value=_review()),
        patch(f"{EXPERIENCE_MODULE}._request_role_rewrite_proposal", return_value=repaired),
    ):
        decision = _decide_role_rewrite_outcome(
            first,
            "context",
            _resume(experience),
            experience,
            run_id="test-run",
        )

    shipped = decision.finalized_section.optimized_experiences[0]
    assert shipped.achievements == SOURCE_BULLETS
    assert "original bullets" in decision.finalized_section.optimization_notes


def test_a_dropped_bullet_fails_the_truth_floor() -> None:
    """Bullet count parity is part of the deterministic floor, not a style note."""
    experience = _experience(SOURCE_BULLETS)
    first = _proposal(experience, ["Only one bullet came back."])
    repaired = _proposal(experience, ["Still only one."])

    with (
        patch(f"{EXPERIENCE_MODULE}.detect_claim_inflation", return_value=_clean_inflation()),
        patch(f"{EXPERIENCE_MODULE}.audit_experience_rewrite_quality", return_value=_review()),
        patch(f"{EXPERIENCE_MODULE}._request_role_rewrite_proposal", return_value=repaired),
    ):
        decision = _decide_role_rewrite_outcome(
            first,
            "context",
            _resume(experience),
            experience,
            run_id="test-run",
        )

    shipped = decision.finalized_section.optimized_experiences[0]
    assert shipped.achievements == SOURCE_BULLETS
    assert "Bullet count changed" in decision.finalized_section.optimization_notes


def test_reordered_bullet_ids_fail_the_truth_floor() -> None:
    """Bullet identity must survive the rewrite, so ids must come back in source order."""
    experience = _experience(SOURCE_BULLETS)
    reversed_ids = [
        build_experience_bullet_id(experience, 1),
        build_experience_bullet_id(experience, 0),
    ]
    first = _proposal(
        experience,
        ["Ran deployments.", "Built Python services."],
        bullet_ids=reversed_ids,
    )
    repaired = _proposal(
        experience,
        ["Ran deployments again.", "Built more services."],
        bullet_ids=reversed_ids,
    )

    with (
        patch(f"{EXPERIENCE_MODULE}.detect_claim_inflation", return_value=_clean_inflation()),
        patch(f"{EXPERIENCE_MODULE}.audit_experience_rewrite_quality", return_value=_review()),
        patch(f"{EXPERIENCE_MODULE}._request_role_rewrite_proposal", return_value=repaired),
    ):
        decision = _decide_role_rewrite_outcome(
            first,
            "context",
            _resume(experience),
            experience,
            run_id="test-run",
        )

    shipped = decision.finalized_section.optimized_experiences[0]
    assert shipped.achievements == SOURCE_BULLETS
    assert "Bullet IDs changed" in decision.finalized_section.optimization_notes
