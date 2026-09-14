"""How to run, and resume, a LangGraph pipeline that can pause mid-way. Start here.

THE TWO PUBLIC FUNCTIONS
    tailor_resume(resume_path, jd_path)   -- start a fresh run
    resume_paused_run(paused_run_path)    -- continue a run that paused for answers

THIS FILE IS THE GENERAL PATTERN; THE OTHER TWO FILES ARE THIS PROJECT'S BOOKKEEPING
    Everything in this file is the reusable shape: open a checkpointer, key a run by
    thread_id, hand the graph its input, detect that a node paused it, and resume
    that same thread_id later with a Command. That shape would look almost the same
    in any LangGraph project that needs to pause for human input.

    What is deliberately NOT in this file: turning the graph's raw output into this
    project's own OrchestrationResult type (see run_results.py), and deciding what
    happens to a run's checkpoint file and PII mapping once it ends (see
    run_lifecycle.py). Both of those are full of resume-tailoring-specific
    decisions; nothing in them generalizes to your own project the way this file does.

READ IN THIS ORDER: tailor_resume(), then resume_paused_run(), then _execute_run()
    Both entry points do the same three things -- open a checkpointer, describe what
    they are running, hand it to _execute_run() -- and differ only in how they build
    those three things. _execute_run() is the one function that actually drives the
    graph; read it once and you have read the whole pattern.

THE FRESH-RUN / RESUME FORK, IN ONE PICTURE
    build the starting input   ->   build_resume_enhancement_graph()
    (a dict, or a Command)          (see graph.py)
              |                              |
              v                              v
                     compiled_graph.invoke(input)
                              |
          +-------------------+-------------------+
          |                                        |
    a node called interrupt()             the graph ran to the end
    (see nodes/experience/node.py)                  |
          |                                         v
          v                           the final state dict -- every field every
    state was saved to the                 node along the way filled in
    checkpoint (see checkpointing.py)
    -- the process can exit here;
    resume_paused_run() picks this
    same thread_id back up later
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from src.core.logger import get_logger
from src.data_models.orchestration import OrchestrationResult
from src.hitl.professional_experience.persistence import (
    SHEET_FILENAME,
    PausedRunLayout,
    load_paused_run,
    read_answered_clarifications,
)
from src.observability import init_observability
from src.orchestration.checkpointing import close_checkpoint_database, open_checkpoint_database
from src.orchestration.graph import build_resume_enhancement_graph
from src.orchestration.run_lifecycle import in_flight_checkpoint_db_path, settle_run_state
from src.orchestration.run_results import finalize_pipeline_output, log_pipeline_completion
from src.orchestration.state import new_pipeline_state

logger = get_logger(__name__)

ProgressCallback = Callable[[str, str], None]


def tailor_resume(
    resume_path: str,
    jd_path: str,
    progress_callback: ProgressCallback | None = None,
) -> OrchestrationResult:
    """Run a fresh pipeline execution from source documents."""
    init_observability("resume-tailor-agents")
    run_id = uuid4().hex
    checkpoint_db_path = in_flight_checkpoint_db_path(run_id)
    return _execute_run(
        _RunPlan(
            run_id=run_id,
            resume_path=resume_path,
            jd_path=jd_path,
            checkpointer=open_checkpoint_database(checkpoint_db_path),
            pipeline_input=new_pipeline_state(run_id, resume_path, jd_path),
            started_log_fields={"resume_mode": "fresh"},
            in_flight_checkpoint_db=checkpoint_db_path,
        ),
        progress_callback,
    )


# HITL COMPONENT 6 -- RESUME. See
# src/hitl/professional_experience/README.md#9-component-6--resume-mechanism
def resume_paused_run(
    paused_run_path: str,
    progress_callback: ProgressCallback | None = None,
) -> OrchestrationResult:
    """Resume a previously paused professional-experience clarification run.

    Everything that can refuse the resume is checked before any pipeline
    machinery is opened: an expired run and an unanswered sheet both fail here,
    cheaply, instead of after a checkpoint connection and a graph compile.
    """
    init_observability("resume-tailor-agents")
    layout, manifest = load_paused_run(paused_run_path)
    if manifest.is_expired:
        raise ValueError(
            f"This paused run expired on {manifest.expires_at.isoformat()} and can no "
            "longer be resumed. Start a fresh run to tailor this resume again."
        )
    answered_clarifications = read_answered_clarifications(layout)
    if not answered_clarifications:
        raise ValueError(
            f"{SHEET_FILENAME} has no answered questions yet; answer at least "
            "one clarification before resuming the paused run."
        )

    return _execute_run(
        _RunPlan(
            run_id=manifest.run_id,
            resume_path=manifest.resume_path,
            jd_path=manifest.jd_path,
            checkpointer=open_checkpoint_database(layout.checkpoint_db),
            # Command tells LangGraph: reload thread_id's saved state from the
            # checkpointer, merge `update` into it, then continue from the exact
            # node that called interrupt() -- instead of starting over at START.
            pipeline_input=Command(
                resume={"status": "candidate_answers_submitted"},
                update={"clarification_answers": answered_clarifications},
            ),
            started_log_fields={
                "resume_mode": "paused_run_resume",
                "paused_run_path": paused_run_path,
                "answered_clarifications": len(answered_clarifications),
            },
            paused_run_layout=layout,
        ),
        progress_callback,
    )


@dataclass(frozen=True)
class _RunPlan:
    """Everything _execute_run needs, assembled by whichever entry point was called.

    Exists purely to avoid passing eight separate arguments to _execute_run; it
    never leaves this file. Exactly one of the last two fields is set, and that is
    what marks the mode:
      * in_flight_checkpoint_db -- a fresh run. Its checkpoint sits in a scratch
        directory and moves into a paused-run folder, or is deleted, when the run ends.
      * paused_run_layout -- a resume. Its checkpoint already lives in the paused-run
        folder, and the same folder is reused if the run pauses again.
    """

    run_id: str
    resume_path: str
    jd_path: str
    checkpointer: SqliteSaver
    pipeline_input: Any
    started_log_fields: dict[str, Any]
    in_flight_checkpoint_db: Path | None = None
    paused_run_layout: PausedRunLayout | None = None


def _execute_run(
    plan: _RunPlan,
    progress_callback: ProgressCallback | None,
) -> OrchestrationResult:
    """Compile, invoke, and settle one run. Both entry points end up here.

    The caller opens the checkpointer, this function always closes it. Opening it
    here instead would mean a failed open landed in the finally block, which would
    then try to close a connection that was never established.
    """
    start_time = time.monotonic()
    result: OrchestrationResult | None = None
    pipeline = build_resume_enhancement_graph(checkpointer=plan.checkpointer)
    config: RunnableConfig = {"configurable": {"thread_id": plan.run_id}}
    logger.info(
        "pipeline_run_started",
        run_id=plan.run_id,
        resume_path=plan.resume_path,
        jd_path=plan.jd_path,
        **plan.started_log_fields,
    )
    try:
        output = _invoke_pipeline(pipeline, plan.pipeline_input, config, progress_callback)
        result = finalize_pipeline_output(
            pipeline=pipeline,
            config=config,
            output=output,
            run_id=plan.run_id,
            resume_path=plan.resume_path,
            jd_path=plan.jd_path,
            paused_run_layout=plan.paused_run_layout,
        )
        log_pipeline_completion(plan.run_id, result, start_time)
        return result
    except Exception:
        logger.exception(
            "pipeline_run_failed",
            run_id=plan.run_id,
            duration_ms=round((time.monotonic() - start_time) * 1000),
        )
        raise
    finally:
        close_checkpoint_database(plan.checkpointer)
        settle_run_state(
            plan.run_id,
            result,
            in_flight_checkpoint_db=plan.in_flight_checkpoint_db,
            paused_run_layout=plan.paused_run_layout,
        )


def _invoke_pipeline(
    pipeline: CompiledStateGraph,
    pipeline_input: Any,
    config: RunnableConfig,
    progress_callback: ProgressCallback | None,
) -> dict:
    """Invoke normally, or stream task lifecycle events to a caller that wants them.

    stream_mode=["tasks", "values"] asks LangGraph for two interleaved kinds of
    event as each node runs: a "tasks" event when a node starts or finishes (used
    here only to call progress_callback), and a "values" event carrying the state
    after each merge -- the last one of those is the same final output invoke()
    would have returned directly.
    """
    if progress_callback is None:
        return cast(dict, pipeline.invoke(pipeline_input, config=config))
    output: dict | None = None
    for mode, event in pipeline.stream(
        pipeline_input, config=config, stream_mode=["tasks", "values"]
    ):
        if mode == "tasks":
            _report_task_event(progress_callback, cast(dict, event))
        else:
            output = cast(dict, event)
    if output is None:
        raise RuntimeError("Pipeline completed without emitting state")
    return output


def _report_task_event(callback: ProgressCallback, event: dict[str, Any]) -> None:
    """Translate one LangGraph task event into a stable UI lifecycle event."""
    if "result" not in event and "error" not in event:
        callback("started", str(event["name"]))
        return
    callback("failed" if event.get("error") else "completed", str(event["name"]))
