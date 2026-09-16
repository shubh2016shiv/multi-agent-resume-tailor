"""Contracts for the active resume-enhancement graph topology."""

from src.orchestration.graph import build_resume_enhancement_graph

ACTIVE_NODES = {
    "analyze_job",
    "assemble_ats_resume",
    "await_candidate_clarifications",
    "evaluate_resume_quality",
    "extract_resume",
    "optimize_experience",
    "optimize_skills",
    "patch_ats_assembly",
    "render_final_resume",
    "run_gap_analysis",
    "write_professional_summary",
}


def test_graph_registers_every_active_node() -> None:
    """The compiled graph should expose every public pipeline node exactly once."""
    graph = build_resume_enhancement_graph().get_graph()
    registered = set(graph.nodes) - {"__start__", "__end__"}
    assert registered == ACTIVE_NODES


def test_graph_connects_the_main_pipeline_stages() -> None:
    """The primary pipeline should remain connected from ingestion through rendering."""
    graph = build_resume_enhancement_graph().get_graph()
    edges = {(edge.source, edge.target) for edge in graph.edges}
    expected = {
        ("__start__", "extract_resume"),
        ("__start__", "analyze_job"),
        ("extract_resume", "run_gap_analysis"),
        ("analyze_job", "run_gap_analysis"),
        ("assemble_ats_resume", "evaluate_resume_quality"),
        ("evaluate_resume_quality", "render_final_resume"),
        ("patch_ats_assembly", "render_final_resume"),
        ("render_final_resume", "__end__"),
    }
    assert expected <= edges
