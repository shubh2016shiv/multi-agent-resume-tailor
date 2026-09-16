"""Graph node that generates the final resume documents."""

from pathlib import Path

from src.core.logger import get_logger
from src.core.settings import get_config
from src.orchestration.nodes.node_contract import StateUpdate
from src.orchestration.nodes.node_logging import log_node_execution
from src.orchestration.nodes.render_resume.generate_resume_documents import (
    generate_resume_documents,
)
from src.orchestration.state import ResumeEnhancementPipelineState, require

logger = get_logger(__name__)


@log_node_execution("render_final_resume")
def render_final_resume(state: ResumeEnhancementPipelineState) -> StateUpdate:
    """Generate resume files from the quality-approved optimized resume."""
    optimized_resume = require(state["optimized_resume"], "optimized_resume")
    job = require(state["job_description"], "job_description")
    artifacts = generate_resume_documents(
        optimized_resume.final_resume,
        job,
        Path(get_config().file_paths.output_dir),
        state["run_id"],
    )
    logger.info(
        "resume_artifacts_written",
        run_id=state["run_id"],
        markdown_path=artifacts.markdown_path,
        docx_path=artifacts.docx_path,
        pdf_rendered=artifacts.pdf_path is not None,
    )
    return {"rendered_artifacts": artifacts}


__all__ = ["render_final_resume"]
