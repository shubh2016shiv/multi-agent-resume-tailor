"""
Resume enhancement pipeline as a LangGraph StateGraph.

Reading this file gives the complete pipeline topology at a glance.
No agent logic lives here -- all agent calls are in nodes.py.

Stage 1 (parallel):   extract_resume + analyze_job
Stage 2 (sequential): run_gap_analysis          (waits for Stage 1)
Stage 3 (parallel):   write_professional_summary + optimize_experience + optimize_skills
Stage 4 (sequential): assemble_ats_resume        (waits for Stage 3)
Stage 5 (sequential): evaluate_resume_quality
Stage 6 (sequential): rehydrate_pii              (restore PII after QA, on every path)
Stage 7 (conditional): render_final_resume       (gate passed, or render_draft_on_gate_fail=True)
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.core.logger import get_logger
from src.core.settings import get_config
from src.data_models.evaluation import AtsCheckStatus
from src.orchestration import nodes
from src.orchestration.state import ResumeEnhancementPipelineState
from src.resume_quality_evaluation import should_render_resume

logger = get_logger(__name__)


def _route_after_ats_check(state: ResumeEnhancementPipelineState) -> str:
    """Route on the rendered-ATS verdict produced by the QA node.

    Returns "patch" when the ATS check FAILed -- a recoverable case where an essential
    section rendered empty and can be restored from typed upstream state. Returns "continue"
    otherwise: PASS proceeds to the render gate, and INCONCLUSIVE was already flagged for
    human review in the QA node (nothing was built to patch), so it falls through to end.

    Precondition: evaluate_resume_quality has populated rendered_structure_evaluation.
    """
    ats_outcome = state["rendered_structure_evaluation"]
    if ats_outcome is None:
        raise ValueError("rendered_structure_evaluation is None after quality evaluation.")
    decision = "patch" if ats_outcome.status is AtsCheckStatus.FAIL else "continue"
    logger.info(
        "graph_routing_decision",
        from_node="evaluate_resume_quality",
        decision=decision,
        ats_status=ats_outcome.status.value,
    )
    return decision


def _route_after_quality(state: ResumeEnhancementPipelineState) -> str:
    """Route on the code-owned quality gate after QA.

    Returns "render" when the QA report passed the threshold. Also returns "render"
    when it failed but render_draft_on_gate_fail is True (development mode: write md+docx
    so the draft is viewable without inspecting the JSON). Returns "end" otherwise.

    Precondition: evaluate_resume_quality has populated quality_report.
    """
    quality_report = state["quality_report"]
    if quality_report is None:
        raise ValueError(
            "quality_report is None after quality evaluation; cannot route render gate."
        )
    if should_render_resume(quality_report):
        decision = "render"
        reason = "gate_passed"
    elif get_config().feature_flags.render_draft_on_gate_fail:
        decision = "render"
        reason = "draft_on_gate_fail"
    else:
        decision = "end"
        reason = "gate_failed"
    logger.info(
        "graph_routing_decision",
        from_node="rehydrate_pii",
        decision=decision,
        reason=reason,
        overall_score=quality_report.overall_quality_score,
    )
    return decision


def build_resume_enhancement_graph() -> CompiledStateGraph:
    """Construct and compile the resume enhancement pipeline.

    Returns: a CompiledStateGraph ready to invoke with ResumeEnhancementPipelineState.
             Each parallel stage runs concurrently. LangGraph fans out on multiple
             outgoing edges and waits for all incoming edges before entering the
             next sequential node (fan-in).
    """
    graph = StateGraph(ResumeEnhancementPipelineState)

    # -- register every node (name -> function) --
    graph.add_node("extract_resume", nodes.extract_resume)
    graph.add_node("analyze_job", nodes.analyze_job)
    graph.add_node("run_gap_analysis", nodes.run_gap_analysis)
    graph.add_node("write_professional_summary", nodes.write_professional_summary)
    graph.add_node("optimize_experience", nodes.optimize_experience)
    graph.add_node("optimize_skills", nodes.optimize_skills)
    graph.add_node("assemble_ats_resume", nodes.assemble_ats_resume)
    graph.add_node("evaluate_resume_quality", nodes.evaluate_resume_quality)
    graph.add_node("patch_ats_assembly", nodes.patch_ats_assembly)
    graph.add_node("rehydrate_pii", nodes.rehydrate_pii)
    graph.add_node("render_final_resume", nodes.render_final_resume)

    # -- Stage 1: parallel fan-out from START --
    graph.add_edge(START, "extract_resume")
    graph.add_edge(START, "analyze_job")

    # -- Stage 2: fan-in -- both Stage 1 nodes must finish before gap analysis starts --
    graph.add_edge("extract_resume", "run_gap_analysis")
    graph.add_edge("analyze_job", "run_gap_analysis")

    # -- Stage 3: parallel fan-out from gap analysis --
    graph.add_edge("run_gap_analysis", "write_professional_summary")
    graph.add_edge("run_gap_analysis", "optimize_experience")
    graph.add_edge("run_gap_analysis", "optimize_skills")

    # -- Stage 4: fan-in -- all Stage 3 nodes must finish before ATS assembly starts --
    graph.add_edge("write_professional_summary", "assemble_ats_resume")
    graph.add_edge("optimize_experience", "assemble_ats_resume")
    graph.add_edge("optimize_skills", "assemble_ats_resume")

    # -- Stage 5: sequential quality assurance --
    graph.add_edge("assemble_ats_resume", "evaluate_resume_quality")

    # -- Stage 5b: conditional ATS section recovery -- a FAIL (an essential section
    # rendered empty) routes through the deterministic patch; PASS/INCONCLUSIVE skip it.
    graph.add_conditional_edges(
        "evaluate_resume_quality",
        _route_after_ats_check,
        {"patch": "patch_ats_assembly", "continue": "rehydrate_pii"},
    )
    graph.add_edge("patch_ats_assembly", "rehydrate_pii")

    # -- Stage 6: rehydrate PII on every path -- the returned resume must carry
    # real values whether or not it goes on to render, so this runs before the gate.

    # -- Stage 7: conditional render -- only when the QA gate passed --
    graph.add_conditional_edges(
        "rehydrate_pii",
        _route_after_quality,
        {"render": "render_final_resume", "end": END},
    )
    graph.add_edge("render_final_resume", END)

    return graph.compile()
