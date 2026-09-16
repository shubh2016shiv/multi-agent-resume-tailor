"""Behavior contracts for the simplified professional-experience stage."""

from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

from src.agents.professional_experience.models import (
    ExperienceBulletRewrite,
    ExperienceRewriteProposal,
)
from src.data_models.resume import Experience, Resume
from src.hitl.professional_experience.models import build_experience_bullet_id
from src.orchestration.nodes.professional_experience.await_candidate_clarifications_node import (
    await_candidate_clarifications,
)
from src.orchestration.nodes.professional_experience.experience_truthfulness_rules import (
    select_truthful_rewrite,
)
from src.tools.contracts import Confidence, Location, ReviewComment, ReviewResult, Section, Severity

TRUTH_RULES = "src.orchestration.nodes.professional_experience.experience_truthfulness_rules"
CLARIFICATION_NODE = (
    "src.orchestration.nodes.professional_experience.await_candidate_clarifications_node"
)
SOURCE_BULLETS = ["Worked on backend services.", "Helped with deployments."]


def _experience() -> Experience:
    """Build one source role for rewrite tests."""
    return Experience(
        experience_id="exp-1",
        job_title="Engineer",
        company_name="Acme",
        start_date=date(2020, 1, 1),
        end_date=None,
        is_current_position=True,
        location=None,
        description="Built backend services for claims operations.",
        achievements=SOURCE_BULLETS,
        skills_used=["Python"],
    )


def _resume(role: Experience) -> Resume:
    """Build the evidence resume used by deterministic checks."""
    return Resume(
        full_name="Jane Doe",
        email="jane.doe@example.com",
        phone_number=None,
        location=None,
        website_or_portfolio=None,
        professional_summary="",
        work_experience=[role],
        education=[],
        skills=[],
        certifications=[],
        languages=[],
    )


def _proposal(role: Experience, bullets: list[str]) -> ExperienceRewriteProposal:
    """Build a proposal with source-ordered bullet identifiers."""
    rewrites = [
        ExperienceBulletRewrite(
            bullet_id=build_experience_bullet_id(role, index),
            source_bullet=role.achievements[index],
            rewritten_bullet=bullet,
            supporting_role_evidence=["claims operations"],
            ownership_level="executed",
            clarifying_question=None,
        )
        for index, bullet in enumerate(bullets)
    ]
    return ExperienceRewriteProposal(rewritten_bullets=rewrites, optimization_notes="")


def test_truthful_rewrite_is_selected_once() -> None:
    """A proposal clearing deterministic checks replaces only the bullets."""
    role = _experience()
    proposal = _proposal(role, ["Built claims services.", "Supported deployments."])
    clean_review = ReviewResult(comments=[], summary="", score=None)

    with patch(f"{TRUTH_RULES}.detect_claim_inflation", return_value=clean_review):
        selected, findings = select_truthful_rewrite(_resume(role), proposal, role)

    assert selected.achievements == ["Built claims services.", "Supported deployments."]
    assert selected.company_name == role.company_name
    assert selected.job_title == role.job_title
    assert findings == []


def test_changed_bullet_count_preserves_source_role() -> None:
    """A structurally incomplete proposal cannot replace source evidence."""
    role = _experience()
    proposal = _proposal(role, ["Built claims services."])

    selected, findings = select_truthful_rewrite(_resume(role), proposal, role)

    assert selected is role
    assert findings and "Bullet count changed" in findings[0]


def test_invented_number_preserves_source_role() -> None:
    """A proposal containing unsupported numeric claims falls back to source."""
    role = _experience()
    proposal = _proposal(role, ["Improved throughput by 90%.", "Supported deployments."])
    finding = ReviewComment(
        engine_id="claim_inflation_detector",
        message="Unsupported number",
        quoted_text="90%",
        location=Location(section=Section.EXPERIENCE, bullet_index=0),
        severity=Severity.BLOCKER,
        confidence=Confidence.HIGH,
        advice="Remove the number",
        proposed_rewrite=None,
    )
    review = ReviewResult(comments=[finding], summary="", score=None)

    with patch(f"{TRUTH_RULES}.detect_claim_inflation", return_value=review):
        selected, findings = select_truthful_rewrite(_resume(role), proposal, role)

    assert selected is role
    assert findings == ["Unsupported number. Remove the number"]


def test_clarification_node_continues_when_no_questions_exist() -> None:
    """The graph proceeds immediately when the stage raised no questions."""
    state = {"run_id": "run-1", "experience_clarifications": []}
    assert await_candidate_clarifications(state) == {}  # type: ignore[arg-type]


def test_clarification_node_pauses_for_unanswered_questions() -> None:
    """Unanswered questions produce one LangGraph interrupt payload."""
    question = SimpleNamespace(model_dump=lambda mode: {"question": "What changed?"})
    state = {
        "run_id": "run-1",
        "experience_clarifications": [question],
        "clarification_answers": [],
    }
    with patch(f"{CLARIFICATION_NODE}.interrupt") as graph_interrupt:
        assert await_candidate_clarifications(state) == {}  # type: ignore[arg-type]

    graph_interrupt.assert_called_once_with(
        {
            "type": "candidate_clarifications_required",
            "questions": [{"question": "What changed?"}],
        }
    )
