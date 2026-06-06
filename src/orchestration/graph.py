"""
Resume enhancement pipeline as a LangGraph StateGraph.

Reading this file gives the complete pipeline topology at a glance.
No agent logic lives here -- all agent calls are in nodes.py.

Stage 1 (parallel):   extract_resume + analyze_job
Stage 2 (sequential): run_gap_analysis          (waits for Stage 1)
Stage 3 (parallel):   write_professional_summary + optimize_experience + optimize_skills
Stage 4 (sequential): assemble_ats_resume        (waits for Stage 3)
Stage 5 (sequential): run_quality_assurance
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.orchestration import nodes
from src.orchestration.state import ResumeEnhancementPipelineState


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
    graph.add_node("run_quality_assurance", nodes.run_quality_assurance)

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

    # -- Stage 5: sequential quality assurance then done --
    graph.add_edge("assemble_ats_resume", "run_quality_assurance")
    graph.add_edge("run_quality_assurance", END)

    return graph.compile()
