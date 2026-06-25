"""Stage 1 ingestion nodes for the resume enhancement graph."""

import time

from src.agents.job_description_analyser import create_job_analyzer_agent

# src/agents/resume_parser is the authoritative resume extraction agent.
# It owns all 4 document-ingestion tools (convert -> quality check -> redact -> extract).
# The node passes a file path -- the agent reasons through the steps itself.
from src.agents.resume_parser import create_resume_extractor_agent
from src.core.logger import get_logger
from src.core.run_id_binding import bind_run_id
from src.data_models.job import JobDescription
from src.data_models.resume import Resume
from src.orchestration.crew_task_execution import run_agent_task
from src.orchestration.state import ResumeEnhancementPipelineState
from src.tools.engines.document_ingestion.document_conversion import convert_document_to_markdown
from src.tools.engines.document_ingestion.resume_extraction import assign_experience_ids

logger = get_logger(__name__)


def extract_resume(state: ResumeEnhancementPipelineState) -> dict:
    """Hand the resume file path to the resume_parser agent.

    Reads: resume_path, a file path to a PDF or DOCX; run_id, to scope PII storage.
    Writes: resume.
    Returns: partial state with a structured Resume whose experiences have IDs.

    Binds run_id so the agent's redaction tool can store the PII mapping under it;
    the agent itself never sees the run_id.
    """
    start_time = time.monotonic()
    logger.info(
        "pipeline_stage_started",
        stage="extract_resume",
        run_id=state["run_id"],
    )
    agent = create_resume_extractor_agent()
    with bind_run_id(state["run_id"]):
        resume = run_agent_task(
            agent=agent,
            task_name="extract_resume_content_task",
            context=f"RESUME FILE PATH: {state['resume_path']}",
            output_model=Resume,
        )
    result = {"resume": assign_experience_ids(resume)}
    duration_ms = round((time.monotonic() - start_time) * 1000)
    logger.info(
        "pipeline_stage_completed",
        stage="extract_resume",
        run_id=state["run_id"],
        duration_ms=duration_ms,
    )
    return result


def analyze_job(state: ResumeEnhancementPipelineState) -> dict:
    """Convert the JD file to Markdown and extract a structured JobDescription.

    Reads: jd_path, a file path to a PDF, DOCX, or plain text file.
    Writes: job_description.
    Returns: partial state with the structured JobDescription.
    """
    start_time = time.monotonic()
    logger.info(
        "pipeline_stage_started",
        stage="analyze_job",
        run_id=state["run_id"],
    )
    jd_markdown = convert_document_to_markdown(state["jd_path"])
    agent = create_job_analyzer_agent()
    job_description = run_agent_task(
        agent=agent,
        task_name="analyze_job_description_task",
        context=f"JOB DESCRIPTION:\n{jd_markdown}",
        output_model=JobDescription,
    )
    result = {"job_description": job_description}
    duration_ms = round((time.monotonic() - start_time) * 1000)
    logger.info(
        "pipeline_stage_completed",
        stage="analyze_job",
        run_id=state["run_id"],
        duration_ms=duration_ms,
    )
    return result
