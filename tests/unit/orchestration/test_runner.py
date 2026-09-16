"""Contracts for the checkpoint and result helpers used by the runner.

These encode checkpoint retention and public-result rules that are expensive to get
wrong because deleting a paused checkpoint makes a run unresumable.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.data_models.job import JobDescription, JobLevel
from src.data_models.orchestration import OrchestrationResult, RunDisposition
from src.data_models.resume import Resume
from src.data_models.strategy import AlignmentStrategy
from src.hitl.professional_experience.persistence import PausedRunLayout
from src.orchestration.checkpointing import _settle_fresh_checkpoint, _settle_resumed_checkpoint
from src.orchestration.run_results import (
    _derive_run_disposition,
    _persist_result,
    _result_output_dir,
)
from src.orchestration.runner import _execute_run, _invoke_pipeline, _RunPlan


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


# --- runner safety ----------------------------------------------------------


def test_compile_failure_closes_and_settles_checkpoint(tmp_path: Path) -> None:
    """A graph compilation error must not leak the run's SQLite connection."""
    checkpointer = MagicMock()
    checkpoint_path = tmp_path / "run.sqlite3"
    plan = _RunPlan(
        run_id="run-1",
        resume_path="resume.pdf",
        jd_path="job.txt",
        checkpointer=checkpointer,
        pipeline_input={},
        started_log_fields={},
        in_flight_checkpoint_db=checkpoint_path,
    )
    with (
        patch(
            "src.orchestration.runner.build_resume_enhancement_graph",
            side_effect=RuntimeError("compile failed"),
        ),
        patch("src.orchestration.runner.close_checkpoint_database") as close_checkpoint,
        patch("src.orchestration.runner.settle_checkpoint") as settle,
    ):
        with pytest.raises(RuntimeError, match="compile failed"):
            _execute_run(plan, None)

    close_checkpoint.assert_called_once_with(checkpointer)
    settle.assert_called_once_with(
        None,
        in_flight_path=checkpoint_path,
        paused_layout=None,
    )


def test_progress_callback_failure_does_not_abort_pipeline() -> None:
    """A broken UI callback must not discard a valid pipeline result."""
    pipeline = MagicMock()
    pipeline.stream.return_value = iter(
        [("tasks", {"name": "extract_resume"}), ("values", {"run_id": "run-1"})]
    )
    callback = MagicMock(side_effect=RuntimeError("UI disconnected"))

    with patch("src.orchestration.runner.logger.warning") as warning:
        output = _invoke_pipeline(
            pipeline,
            {},
            {"configurable": {"thread_id": "run-1"}},
            callback,
        )

    assert output == {"run_id": "run-1"}
    callback.assert_called_once_with("started", "extract_resume")
    warning.assert_called_once()


# --- fresh-run checkpoint settling -------------------------------------------


def test_fresh_run_that_paused_archives_its_checkpoint_into_the_paused_run(
    tmp_path: Path,
) -> None:
    """A paused run keeps its checkpoint, moved next to the sheet it belongs to."""
    checkpoint_db = tmp_path / "checkpoints" / "run.sqlite3"
    checkpoint_db.parent.mkdir(parents=True)
    checkpoint_db.write_bytes(b"checkpoint")
    paused_run = tmp_path / "paused_run_abc"

    _settle_fresh_checkpoint(checkpoint_db, _paused_result(str(paused_run)))

    assert not checkpoint_db.exists()
    assert PausedRunLayout.at(paused_run).checkpoint_db.read_bytes() == b"checkpoint"


def test_fresh_run_that_completed_deletes_its_checkpoint(tmp_path: Path) -> None:
    """A terminal run has nothing to resume, so its checkpoint history is removed."""
    checkpoint_db = tmp_path / "run.sqlite3"
    checkpoint_db.write_bytes(b"checkpoint")

    _settle_fresh_checkpoint(checkpoint_db, _result(disposition=RunDisposition.RENDERED))

    assert not checkpoint_db.exists()


def test_fresh_run_that_crashed_deletes_its_checkpoint(tmp_path: Path) -> None:
    """result=None means the run raised; there is no pause to resume from."""
    checkpoint_db = tmp_path / "run.sqlite3"
    checkpoint_db.write_bytes(b"checkpoint")

    _settle_fresh_checkpoint(checkpoint_db, None)

    assert not checkpoint_db.exists()


def test_settling_a_missing_checkpoint_is_not_an_error(tmp_path: Path) -> None:
    """The finally block runs even when the checkpoint was never created."""
    _settle_fresh_checkpoint(tmp_path / "never_created.sqlite3", None)


# --- resumed-run checkpoint settling ----------------------------------------


def test_resumed_run_that_completed_deletes_the_paused_run_checkpoint(tmp_path: Path) -> None:
    """Only a genuinely finished resume retires the paused run's checkpoint."""
    layout = PausedRunLayout.at(tmp_path / "paused_run_abc")
    layout.root.mkdir(parents=True)
    layout.checkpoint_db.write_bytes(b"checkpoint")

    _settle_resumed_checkpoint(layout, _result(disposition=RunDisposition.RENDERED))

    assert not layout.checkpoint_db.exists()


def test_resumed_run_that_paused_again_keeps_its_checkpoint(tmp_path: Path) -> None:
    """A re-paused resume must stay resumable after the candidate answers more."""
    layout = PausedRunLayout.at(tmp_path / "paused_run_abc")
    layout.root.mkdir(parents=True)
    layout.checkpoint_db.write_bytes(b"checkpoint")

    _settle_resumed_checkpoint(layout, _paused_result(str(layout.root)))

    assert layout.checkpoint_db.exists()


def test_resumed_run_that_crashed_keeps_its_checkpoint(tmp_path: Path) -> None:
    """A failed resume (bad sheet, transient error) must remain resumable."""
    layout = PausedRunLayout.at(tmp_path / "paused_run_abc")
    layout.root.mkdir(parents=True)
    layout.checkpoint_db.write_bytes(b"checkpoint")

    _settle_resumed_checkpoint(layout, None)

    assert layout.checkpoint_db.exists()


# --- result output directory ------------------------------------------------


def test_run_disposition_uses_blocking_precedence() -> None:
    """Human review, gate failure, and candidate input should win in that order."""
    assert _derive_run_disposition(True, True, False) is RunDisposition.NEEDS_HUMAN_REVIEW
    assert _derive_run_disposition(False, False, True) is RunDisposition.QUALITY_GATE_FAILED
    assert _derive_run_disposition(False, True, True) is RunDisposition.NEEDS_CANDIDATE_INPUT
    assert _derive_run_disposition(False, True, False) is RunDisposition.RENDERED


def test_paused_result_is_persisted_inside_its_paused_run_directory() -> None:
    """A paused run's result JSON lands in the folder the candidate works in."""
    assert _result_output_dir(_paused_result("some/paused_run_abc")) == Path("some/paused_run_abc")


def test_result_persistence_reuses_one_path_per_run(tmp_path: Path) -> None:
    """A retry overwrites the same result JSON instead of creating a duplicate."""
    result = _paused_result(str(tmp_path))

    _persist_result(result, "run-1")
    _persist_result(result, "run-1")

    assert list(tmp_path.glob("run_*.json")) == [tmp_path / "run_run-1.json"]


def test_completed_result_is_persisted_beside_the_rendered_artifacts(
    monkeypatch, tmp_path: Path
) -> None:
    """A terminal run's result JSON lands in <output_dir>/<candidate>/<designation>."""
    monkeypatch.setattr(
        "src.orchestration.run_results.get_config",
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
