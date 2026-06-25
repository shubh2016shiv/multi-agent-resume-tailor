"""Unit tests for the run-id binding behavior."""

import threading

import pytest

from src.core.run_id_binding import MissingRunIdError, bind_run_id, get_current_run_id


def test_get_current_run_id_returns_bound_run_id_inside_binding_scope() -> None:
    """A bound run_id is readable for the entire active binding scope."""
    with bind_run_id("run-123"):
        assert get_current_run_id() == "run-123"


def test_get_current_run_id_raises_when_no_run_id_is_bound() -> None:
    """Reading the current run_id outside a binding scope fails closed."""
    with pytest.raises(MissingRunIdError):
        get_current_run_id()


def test_bind_run_id_restores_previous_run_id_after_nested_binding() -> None:
    """Leaving an inner binding restores the outer run_id."""
    with bind_run_id("outer-run"):
        with bind_run_id("inner-run"):
            assert get_current_run_id() == "inner-run"
        assert get_current_run_id() == "outer-run"


def test_bind_run_id_rejects_empty_run_id() -> None:
    """The binding contract rejects an empty run_id."""
    with pytest.raises(ValueError):
        with bind_run_id(""):
            pass


def test_bound_run_id_is_visible_to_worker_threads() -> None:
    """A worker thread can read the run_id bound by the ingestion thread.

    Reproduces the real failure: a ContextVar bound here would be unset inside a
    worker thread. The run_id must be readable from that thread.
    """
    observed_run_id_by_thread: dict[str, str] = {}

    def read_bound_run_id_from_worker_thread() -> None:
        observed_run_id_by_thread["run_id"] = get_current_run_id()

    with bind_run_id("run-threaded"):
        worker_thread = threading.Thread(target=read_bound_run_id_from_worker_thread)
        worker_thread.start()
        worker_thread.join()

    assert observed_run_id_by_thread["run_id"] == "run-threaded"
