"""Graph node that extracts structured resume content from a document."""

from src.agents.resume_parser import create_resume_extractor_agent
from src.core.run_id_binding import bind_run_id
from src.data_models.resume import Resume
from src.orchestration.crew_task_execution import run_agent_task
from src.orchestration.nodes.node_contract import StateUpdate
from src.orchestration.nodes.node_logging import log_node_execution
from src.orchestration.state import ResumeEnhancementPipelineState
from src.tools.engines.document_ingestion.resume_extraction import assign_experience_ids


@log_node_execution("extract_resume")
def extract_resume(state: ResumeEnhancementPipelineState) -> StateUpdate:
    """Extract a resume from `resume_path` and return its structured state update."""
    agent = create_resume_extractor_agent()
    with bind_run_id(state["run_id"]):
        resume = run_agent_task(
            agent=agent,
            task_name="extract_resume_content_task",
            context=f"RESUME FILE PATH: {state['resume_path']}",
            output_model=Resume,
            run_id=state["run_id"],
        )
    return {"resume": assign_experience_ids(resume)}


__all__ = ["extract_resume"]
