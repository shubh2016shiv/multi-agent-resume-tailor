"""
Public entry point for the resume enhancement pipeline.

Call tailor_resume(resume_path, jd_path) -- it builds the graph, invokes it,
and returns an OrchestrationResult with the optimized resume, QA report, and
(when the quality gate passed) the rendered file paths.

Every run is also written to <output_dir>/<candidate>/<designation>/run_<timestamp>.json
before returning, so the result survives a caller crash and can be re-read for free
without running the (paid) pipeline again -- including gate-fail runs, whose JSON still
carries the qa_report explaining why.
"""

from datetime import datetime
from pathlib import Path
from typing import cast
from uuid import uuid4

from langgraph.graph.state import CompiledStateGraph

from src.core.logger import get_logger
from src.core.pii_mapping_store import delete_pii_mapping
from src.core.settings import get_config
from src.data_models.orchestration import OrchestrationResult
from src.observability import init_observability
from src.orchestration.graph import build_resume_enhancement_graph
from src.orchestration.state import ResumeEnhancementPipelineState
from src.tools.document_rendering.output_location import resume_output_dir

logger = get_logger(__name__)

# Initialize LangSmith tracing once at module load.
init_observability("resume-tailor-agents")

# Build the graph once at module load -- it is stateless and safe to share.
# Each tailor_resume() call gets a fresh initial state but reuses this graph.
_PIPELINE: CompiledStateGraph = build_resume_enhancement_graph()


def tailor_resume(resume_path: str, jd_path: str) -> OrchestrationResult:
    """Run the full resume enhancement pipeline and return the final result.

    Precondition: resume_path and jd_path are paths to readable files
                  in a format supported by convert_document_to_markdown.
    Returns: OrchestrationResult with the optimized resume and QA report.
    Raises: ValueError if any agent node fails to produce a valid typed output.
    """
    run_id = uuid4().hex
    initial_state: ResumeEnhancementPipelineState = {
        "run_id": run_id,
        "resume_path": resume_path,
        "jd_path": jd_path,
        "resume": None,
        "job_description": None,
        "requirement_match_report": None,
        "alignment_strategy": None,
        "professional_summary": None,
        "optimized_experience": None,
        "optimized_skills": None,
        "optimized_resume": None,
        "qa_report": None,
        "ats_rendered_outcome": None,
        "human_review_required": False,
        "rendered_artifacts": None,
    }

    # The run's PII mapping is sensitive run-state; delete it on every terminal
    # path (success, quality-gate end, or exception) once rehydration is done.
    try:
        final_state = cast(ResumeEnhancementPipelineState, _PIPELINE.invoke(initial_state))
        result = _build_orchestration_result(final_state)
        _persist_result(result)
        return result
    finally:
        if get_config().feature_flags.enable_pii_redaction:
            delete_pii_mapping(run_id)


def _persist_result(result: OrchestrationResult) -> str:
    """Write the full run result to a JSON file so it survives a caller crash and can be
    inspected later without paying for another run.

    Saved on EVERY run, whether or not the quality gate passed -- a gate-fail result still
    carries the qa_report explaining why. Lands next to any rendered files at
    <output_dir>/<candidate>/<designation>/run_<timestamp>.json. Returns the written path.
    """
    base_dir = Path(get_config().file_paths.output_dir)
    output_dir = resume_output_dir(result.optimized_resume.final_resume, result.job_description, base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Run result saved", path=str(path), gate_passed=result.qa_report.passed_quality_threshold)
    return str(path)


def _build_orchestration_result(
    state: ResumeEnhancementPipelineState,
) -> OrchestrationResult:
    """Assemble the final OrchestrationResult from the completed pipeline state.

    Precondition: all state fields are non-None (the graph ran to completion).
    Raises: ValueError if any required field is still None after the graph finishes.
    """
    assert state["resume"] is not None, "Pipeline completed but 'resume' is still None"
    assert state["job_description"] is not None, "Pipeline completed but 'job_description' is still None"
    assert state["alignment_strategy"] is not None, "Pipeline completed but 'alignment_strategy' is still None"
    assert state["optimized_resume"] is not None, "Pipeline completed but 'optimized_resume' is still None"
    assert state["qa_report"] is not None, "Pipeline completed but 'qa_report' is still None"

    # TODO: surface state["human_review_required"] on OrchestrationResult so callers can
    #       distinguish "rendered" / "rejected on score" / "needs a human" (the terminal
    #       ATS-unrecoverable and INCONCLUSIVE cases). Proposed: add a bool field to
    #       OrchestrationResult. Deferred: the field exists in state now (Phase 2); adding
    #       the result-model field is its own small change once a caller needs to read it.
    return OrchestrationResult(
        original_resume=state["resume"],
        job_description=state["job_description"],
        strategy=state["alignment_strategy"],
        optimized_resume=state["optimized_resume"],
        qa_report=state["qa_report"],
        rendered_artifacts=state["rendered_artifacts"],
    )
