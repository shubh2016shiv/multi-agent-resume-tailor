"""Tests for the paused-run directory: the sheet, the audit log, and expiry.

The regression these guard is specific. The clarification sheet is an object
with a "clarifications" key, but the web layer used to parse it as a bare list,
so every browser submission raised KeyError while the CLI path worked. The fix
was structural -- one accessor owns the format -- so the tests below exercise
that accessor rather than asserting on the file's bytes.
"""

import json
from datetime import UTC, datetime, timedelta

import pytest

from src.hitl.professional_experience.models import (
    ExperienceBulletClarification,
    ExperienceBulletMissingFactCategory,
    ExperienceClarificationPausedRunManifest,
)
from src.hitl.professional_experience.persistence import (
    PausedRunLayout,
    read_answered_clarifications,
    read_clarification_sheet,
    record_clarification_answers,
    write_clarification_sheet,
)


def _clarification(bullet_id: str) -> ExperienceBulletClarification:
    """One realistic unanswered question."""
    return ExperienceBulletClarification(
        gap_category=ExperienceBulletMissingFactCategory.RESULT,
        missing_fact_summary="No measurable outcome of the migration is stated.",
        why_flagged="The result is not present in any role evidence.",
        question="What changed after the migration shipped?",
        bullet_id=bullet_id,
        company_name="Acme Corp",
        job_title="Backend Engineer",
        start_date="2023-01-01",
        bullet="Migrated the billing service to the new platform.",
    )


@pytest.fixture
def layout(tmp_path) -> PausedRunLayout:
    """A paused-run directory holding two unanswered questions."""
    paused = PausedRunLayout.at(tmp_path / "paused_run_abc")
    write_clarification_sheet(paused, [_clarification("b::1"), _clarification("b::2")])
    return paused


def test_sheet_round_trips_through_the_accessor(layout: PausedRunLayout) -> None:
    """What write_clarification_sheet writes, read_clarification_sheet returns."""
    questions = read_clarification_sheet(layout)

    assert [question.bullet_id for question in questions] == ["b::1", "b::2"]
    assert all(not question.is_answered for question in questions)


def test_sheet_on_disk_is_an_object_not_a_list(layout: PausedRunLayout) -> None:
    """Pin the shape the web layer once guessed wrong.

    The sheet carries instructions for the candidate who edits it by hand via
    the CLI flow, so it cannot be a bare list. Anything reading it positionally
    is reading the wrong thing.
    """
    raw = json.loads(layout.sheet.read_text(encoding="utf-8"))

    assert isinstance(raw, dict)
    assert set(raw) == {"_instructions", "clarifications"}
    assert isinstance(raw["clarifications"], list)


def test_answers_are_recorded_by_bullet_id_not_position(layout: PausedRunLayout) -> None:
    """The second question can be answered without touching the first."""
    record_clarification_answers(
        layout,
        {"b::2": "  Cut p95 latency by 40%.  "},
        answered_by="test",
    )

    answered = {q.bullet_id: q for q in read_answered_clarifications(layout)}
    assert list(answered) == ["b::2"]
    assert answered["b::2"].answer == "Cut p95 latency by 40%."
    assert answered["b::2"].answered_by == "test"
    assert answered["b::2"].answered_at is not None


def test_answering_again_preserves_earlier_answers(layout: PausedRunLayout) -> None:
    """A candidate may answer across several sittings without losing work."""
    record_clarification_answers(layout, {"b::1": "First answer."}, answered_by="test")
    record_clarification_answers(layout, {"b::2": "Second answer."}, answered_by="test")

    answered = {q.bullet_id: q.answer for q in read_answered_clarifications(layout)}
    assert answered == {"b::1": "First answer.", "b::2": "Second answer."}


def test_unknown_bullet_id_is_rejected(layout: PausedRunLayout) -> None:
    """An answer that matches no question is a caller bug, not a silent no-op."""
    with pytest.raises(ValueError, match="No such question"):
        record_clarification_answers(layout, {"nope": "text"}, answered_by="test")


def test_blank_answers_do_not_count_as_answering(layout: PausedRunLayout) -> None:
    """Whitespace is not an answer; resuming on it would waste an LLM round trip."""
    with pytest.raises(ValueError, match="at least one"):
        record_clarification_answers(layout, {"b::1": "   "}, answered_by="test")


def test_audit_log_is_append_only(layout: PausedRunLayout) -> None:
    """Every accepted answer leaves a record the mutable sheet cannot overwrite."""
    record_clarification_answers(layout, {"b::1": "First."}, answered_by="cli")
    record_clarification_answers(layout, {"b::1": "Corrected."}, answered_by="web")

    entries = [
        json.loads(line) for line in layout.audit_log.read_text(encoding="utf-8").splitlines()
    ]
    assert [entry["answer"] for entry in entries] == ["First.", "Corrected."]
    assert [entry["answered_by"] for entry in entries] == ["cli", "web"]
    # The sheet holds only the latest answer; the audit holds the history.
    assert read_clarification_sheet(layout)[0].answer == "Corrected."


def _manifest(expires_in: timedelta) -> ExperienceClarificationPausedRunManifest:
    now = datetime.now(UTC)
    return ExperienceClarificationPausedRunManifest(
        run_id="run-1",
        resume_path="resume.pdf",
        jd_path="jd.txt",
        paused_at=now,
        expires_at=now + expires_in,
    )


def test_paused_run_expiry_is_computed_from_the_clock() -> None:
    """Expiry is derived, so it can never disagree with a stored status field."""
    assert not _manifest(timedelta(hours=1)).is_expired
    assert _manifest(timedelta(hours=-1)).is_expired
