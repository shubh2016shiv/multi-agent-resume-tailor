"""Stage 1 ingestion nodes for the resume enhancement graph."""

from src.agents.job_description_analyser import create_job_analyzer_agent

# src/agents/resume_parser is the authoritative resume extraction agent.
# It owns all 4 document-ingestion tools (convert -> quality check -> redact -> extract).
# The node passes a file path -- the agent reasons through the steps itself.
from src.agents.resume_parser import create_resume_extractor_agent
from src.data_models.job import JobDescription
from src.data_models.resume import Resume
from src.orchestration.crew_task_execution import run_agent_task
from src.orchestration.state import ResumeEnhancementPipelineState
from src.tools.document_ingestion.document_converter import convert_document_to_markdown
from src.tools.document_ingestion.resume_section_extractor import assign_experience_ids


def extract_resume(state: ResumeEnhancementPipelineState) -> dict:
    """Hand the resume file path to the resume_parser agent.

    Reads: resume_path, a file path to a PDF or DOCX.
    Writes: resume.
    Returns: partial state with a structured Resume whose experiences have IDs.
    """
    agent = create_resume_extractor_agent()
    resume = run_agent_task(
        agent=agent,
        task_name="extract_resume_content_task",
        context=f"RESUME FILE PATH: {state['resume_path']}",
        output_model=Resume,
    )
    return {"resume": assign_experience_ids(resume)}


def analyze_job(state: ResumeEnhancementPipelineState) -> dict:
    """Convert the JD file to Markdown and extract a structured JobDescription.

    Reads: jd_path, a file path to a PDF, DOCX, or plain text file.
    Writes: job_description.
    Returns: partial state with the structured JobDescription.
    """
    jd_markdown = convert_document_to_markdown(state["jd_path"])
    agent = create_job_analyzer_agent()
    job_description = run_agent_task(
        agent=agent,
        task_name="analyze_job_description_task",
        context=f"JOB DESCRIPTION:\n{jd_markdown}",
        output_model=JobDescription,
    )
    return {"job_description": job_description}
