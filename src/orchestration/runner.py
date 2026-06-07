"""
Public entry point for the resume enhancement pipeline.

Call tailor_resume(resume_path, jd_path) -- it builds the graph, invokes it,
and returns an OrchestrationResult with the optimized resume, QA report, and
(when the quality gate passed) the rendered PDF path.
"""

from langgraph.graph.state import CompiledStateGraph

from src.data_models.orchestration import OrchestrationResult
from src.orchestration.graph import build_resume_enhancement_graph
from src.orchestration.state import ResumeEnhancementPipelineState

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
    initial_state: ResumeEnhancementPipelineState = {
        "resume_path": resume_path,
        "jd_path": jd_path,
        "resume": None,
        "job_description": None,
        "alignment_strategy": None,
        "professional_summary": None,
        "optimized_experience": None,
        "optimized_skills": None,
        "optimized_resume": None,
        "qa_report": None,
        "rendered_resume_path": None,
    }

    final_state: ResumeEnhancementPipelineState = _PIPELINE.invoke(initial_state)

    return _build_orchestration_result(final_state)


def _build_orchestration_result(
    state: ResumeEnhancementPipelineState,
) -> OrchestrationResult:
    """Assemble the final OrchestrationResult from the completed pipeline state.

    Precondition: all state fields are non-None (the graph ran to completion).
    Raises: ValueError if any required field is still None after the graph finishes.
    """
    required_fields = [
        "resume",
        "job_description",
        "alignment_strategy",
        "optimized_resume",
        "qa_report",
    ]
    missing = [f for f in required_fields if state.get(f) is None]
    if missing:
        raise ValueError(
            f"Pipeline completed but these fields are still None: {missing}. "
            "Check the node that should produce each field."
        )

    return OrchestrationResult(
        original_resume=state["resume"],
        job_description=state["job_description"],
        strategy=state["alignment_strategy"],
        optimized_resume=state["optimized_resume"],
        qa_report=state["qa_report"],
        rendered_resume_path=state["rendered_resume_path"],
    )
