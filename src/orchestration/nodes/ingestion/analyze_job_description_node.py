"""Graph node that extracts structured content from a job description."""

from src.agents.job_description_analyser import create_job_analyzer_agent
from src.data_models.job import JobDescription
from src.orchestration.crew_task_execution import run_agent_task
from src.orchestration.nodes.node_contract import StateUpdate
from src.orchestration.nodes.node_logging import log_node_execution
from src.orchestration.state import ResumeEnhancementPipelineState
from src.tools.engines.document_ingestion.document_conversion import (
    convert_document_to_markdown,
)


@log_node_execution("analyze_job")
def analyze_job(state: ResumeEnhancementPipelineState) -> StateUpdate:
    """Analyze the document at `jd_path` and return its structured state update."""
    job_description = run_agent_task(
        agent=create_job_analyzer_agent(),
        task_name="analyze_job_description_task",
        context=f"JOB DESCRIPTION:\n{convert_document_to_markdown(state['jd_path'])}",
        output_model=JobDescription,
        run_id=state["run_id"],
    )
    return {"job_description": job_description}


__all__ = ["analyze_job"]
