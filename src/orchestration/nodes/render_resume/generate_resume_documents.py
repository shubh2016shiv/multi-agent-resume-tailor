"""Generate Markdown, DOCX, and optional PDF resume documents."""

from pathlib import Path

from src.core.logger import get_logger
from src.data_models.job import JobDescription
from src.data_models.rendering import RenderedResumeArtifacts
from src.data_models.resume import Resume
from src.tools.engines.document_rendering import is_render_available, render_resume_document
from src.tools.engines.document_rendering.docx_renderer import render_resume_docx
from src.tools.engines.document_rendering.markdown_renderer import build_resume_markdown
from src.tools.engines.document_rendering.output_paths import resume_filename, resume_output_dir

logger = get_logger(__name__)


def generate_resume_documents(
    resume: Resume,
    job: JobDescription,
    base_output_dir: Path,
    run_id: str,
) -> RenderedResumeArtifacts:
    """Generate final resume documents and return the paths that were produced."""
    output_dir = resume_output_dir(resume, job, base_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = _write_markdown(resume, job, output_dir, run_id)
    docx_path = _write_docx(resume, job, output_dir, run_id)
    pdf_path, skipped_reason = _try_write_pdf(resume, job, output_dir, run_id)
    return RenderedResumeArtifacts(
        markdown_path=markdown_path,
        docx_path=docx_path,
        pdf_path=pdf_path,
        pdf_skipped_reason=skipped_reason,
    )


def _write_markdown(resume: Resume, job: JobDescription, output_dir: Path, run_id: str) -> str:
    """Write the Markdown document and return its path."""
    path = output_dir / resume_filename(resume, job, "md", run_id)
    path.write_text(build_resume_markdown(resume), encoding="utf-8")
    return str(path)


def _write_docx(resume: Resume, job: JobDescription, output_dir: Path, run_id: str) -> str:
    """Write the DOCX document and return its path."""
    path = output_dir / resume_filename(resume, job, "docx", run_id)
    render_resume_docx(resume, path)
    return str(path)


def _try_write_pdf(
    resume: Resume, job: JobDescription, output_dir: Path, run_id: str
) -> tuple[str | None, str | None]:
    """Write a PDF when the toolchain is available, otherwise return the reason."""
    if not is_render_available():
        reason = "tectonic (LaTeX toolchain) is not installed"
        logger.warning("pdf_render_skipped", reason=reason)
        return None, reason
    try:
        path = output_dir / resume_filename(resume, job, "pdf", run_id)
        return str(render_resume_document(resume, path)), None
    except RuntimeError as error:
        logger.warning("pdf_render_failed", error=str(error))
        return None, f"PDF render failed: {error}"


__all__ = ["generate_resume_documents"]
