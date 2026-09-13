"""Contracts for the runner's checkpoint-settling and PII-cleanup decisions.

These four helpers encode the run lifecycle rules that are easy to get subtly
wrong and expensive when wrong: a deleted checkpoint makes a paused run
unresumable, and a deleted PII mapping makes every later resume unable to
rehydrate. They are pinned here before runner.py is simplified, because the
planned refactor merges the two entry points' shared try/except/finally and
would otherwise be free to change these rules unnoticed.
"""

from pathlib import Path

from src.data_models.job import JobDescription, JobLevel
from src.data_models.orchestration import OrchestrationResult, RunDisposition
from src.data_models.resume import Resume
from src.data_models.strategy import AlignmentStrategy
from src.hitl.professional_experience.persistence import PausedRunLayout
from src.orchestration.runner import (
    _result_output_dir,
    _settle_fresh_run_checkpoint,
    _settle_resumed_run_checkpoint,
    _should_cleanup_pii_mapping,
    _should_cleanup_pii_mapping_after_resume,
)


def _resume_model(full_name: str = "Jane Doe") -> Resume:
    return Resume(
        full_name=full_name,
        email="jane.doe@example.com",
        phone_number=None,
        location=None,
        website_or_portfolio=None,
        professional_summary="",
        work_experience=[],
        education=[],
        skills=[],
        certifications=[],
        languages=[],
    )


def _job() -> JobDescription:
    return JobDescription(
        job_title="Backend Engineer",
        company_name="Acme",
        job_level=JobLevel.MID,
        location=None,
        summary="Build services.",
        full_text="Build services.",
        requirements=[],
        ats_keywords=[],
    )


def _strategy() -> AlignmentStrategy:
    return AlignmentStrategy(
        summary_of_strategy="Lead with backend depth.",
        identified_matches=[],
        identified_gaps=[],
        keywords_to_integrate=[],
        professional_summary_guidance="Open with backend scope.",
        experience_guidance="Foreground service work.",
        skills_guidance="Order Python first.",
        overall_fit_score=50.0,
    )


def _result(
    *,
    disposition: RunDisposition,
    paused_run_path: str | None = None,
) -> OrchestrationResult:
    return OrchestrationResult(
        original_resume=_resume_model(),
        job_description=_job(),
        strategy=_strategy(),
        optimized_resume=None,
        quality_report=None,
        rendered_artifacts=None,
        clarifications_requested=[],
        disposition=disposition,
        paused_run_path=paused_run_path,
    )


def _paused_result(paused_run_path: str) -> OrchestrationResult:
    return _result(
        disposition=RunDisposition.NEEDS_CANDIDATE_INPUT,
        paused_run_path=paused_run_path,
    )


# --- fresh-run checkpoint settling -------------------------------------------


def test_fresh_run_that_paused_archives_its_checkpoint_into_the_paused_run(
    tmp_path: Path,
) -> None:
    """A paused run keeps its checkpoint, moved next to the sheet it belongs to."""
    checkpoint_db = tmp_path / "checkpoints" / "run.sqlite3"
    checkpoint_db.parent.mkdir(parents=True)
    checkpoint_db.write_bytes(b"checkpoint")
    paused_run = tmp_path / "paused_run_abc"

    _settle_fresh_run_checkpoint(checkpoint_db, _paused_result(str(paused_run)))

    assert not checkpoint_db.exists()
    assert PausedRunLayout.at(paused_run).checkpoint_db.read_bytes() == b"checkpoint"


def test_fresh_run_that_completed_deletes_its_checkpoint(tmp_path: Path) -> None:
    """A terminal run has nothing to resume, so its checkpoint history is removed."""
    checkpoint_db = tmp_path / "run.sqlite3"
    checkpoint_db.write_bytes(b"checkpoint")

    _settle_fresh_run_checkpoint(checkpoint_db, _result(disposition=RunDisposition.RENDERED))

    assert not checkpoint_db.exists()


def test_fresh_run_that_crashed_deletes_its_checkpoint(tmp_path: Path) -> None:
    """result=None means the run raised; there is no pause to resume from."""
    checkpoint_db = tmp_path / "run.sqlite3"
    checkpoint_db.write_bytes(b"checkpoint")

    _settle_fresh_run_checkpoint(checkpoint_db, None)

    assert not checkpoint_db.exists()


def test_settling_a_missing_checkpoint_is_not_an_error(tmp_path: Path) -> None:
    """The finally block runs even when the checkpoint was never created."""
    _settle_fresh_run_checkpoint(tmp_path / "never_created.sqlite3", None)


# --- resumed-run checkpoint settling ----------------------------------------


def test_resumed_run_that_completed_deletes_the_paused_run_checkpoint(tmp_path: Path) -> None:
    """Only a genuinely finished resume retires the paused run's checkpoint."""
    layout = PausedRunLayout.at(tmp_path / "paused_run_abc")
    layout.root.mkdir(parents=True)
    layout.checkpoint_db.write_bytes(b"checkpoint")

    _settle_resumed_run_checkpoint(layout, _result(disposition=RunDisposition.RENDERED))

    assert not layout.checkpoint_db.exists()


def test_resumed_run_that_paused_again_keeps_its_checkpoint(tmp_path: Path) -> None:
    """A re-paused resume must stay resumable after the candidate answers more."""
    layout = PausedRunLayout.at(tmp_path / "paused_run_abc")
    layout.root.mkdir(parents=True)
    layout.checkpoint_db.write_bytes(b"checkpoint")

    _settle_resumed_run_checkpoint(layout, _paused_result(str(layout.root)))

    assert layout.checkpoint_db.exists()


def test_resumed_run_that_crashed_keeps_its_checkpoint(tmp_path: Path) -> None:
    """A failed resume (bad sheet, transient error) must remain resumable."""
    layout = PausedRunLayout.at(tmp_path / "paused_run_abc")
    layout.root.mkdir(parents=True)
    layout.checkpoint_db.write_bytes(b"checkpoint")

    _settle_resumed_run_checkpoint(layout, None)

    assert layout.checkpoint_db.exists()


# --- PII mapping cleanup ----------------------------------------------------


def test_fresh_run_keeps_the_pii_mapping_only_while_the_run_is_resumable() -> None:
    """A paused fresh run still needs its mapping; every other outcome cleans up.

    A crashed fresh run (None) cleans up too: there is no paused run left that
    rehydrate_pii could need the mapping for.
    """
    paused = _paused_result("paused_run_abc")

    assert _should_cleanup_pii_mapping(paused) is False
    assert _should_cleanup_pii_mapping(_result(disposition=RunDisposition.RENDERED)) is True
    assert _should_cleanup_pii_mapping(None) is True


def test_resumed_run_keeps_the_pii_mapping_unless_the_run_is_terminal() -> None:
    """Unlike a fresh run, a crashed resume KEEPS the mapping.

    The paused run survives a failed resume, so deleting the mapping here would
    leave every later resume unable to rehydrate PII.
    """
    paused = _paused_result("paused_run_abc")

    assert _should_cleanup_pii_mapping_after_resume(None) is False
    assert _should_cleanup_pii_mapping_after_resume(paused) is False
    assert (
        _should_cleanup_pii_mapping_after_resume(_result(disposition=RunDisposition.RENDERED))
        is True
    )


# --- result output directory ------------------------------------------------


def test_paused_result_is_persisted_inside_its_paused_run_directory() -> None:
    """A paused run's result JSON lands in the folder the candidate works in."""
    assert _result_output_dir(_paused_result("some/paused_run_abc")) == Path("some/paused_run_abc")


def test_completed_result_is_persisted_beside_the_rendered_artifacts(
    monkeypatch, tmp_path: Path
) -> None:
    """A terminal run's result JSON lands in <output_dir>/<candidate>/<designation>."""
    monkeypatch.setattr(
        "src.orchestration.runner.get_config",
        lambda: _FakeConfig(tmp_path),
    )

    output_dir = _result_output_dir(_result(disposition=RunDisposition.RENDERED))

    assert output_dir == tmp_path / "Jane_Doe" / "Backend_Engineer"


class _FakeConfig:
    """Minimal stand-in for the settings object runner reads output_dir from."""

    def __init__(self, output_dir: Path) -> None:
        self.file_paths = _FakeFilePaths(output_dir)


class _FakeFilePaths:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = str(output_dir)
