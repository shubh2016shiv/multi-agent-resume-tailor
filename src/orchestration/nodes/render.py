"""Stage 6 final resume render node.

Plain code, no agent: rendering an already-decided Resume into files is mechanical.
This node runs when the quality gate passed, or when render_draft_on_gate_fail is True
(see _route_after_quality in graph.py). When it runs on a gate-fail, the files are drafts
and quality_report.passes_quality_gate is False -- the caller decides what to do with them.

It always writes Markdown and DOCX (both pure Python, every OS). PDF is best-effort: if the
LaTeX toolchain (tectonic) is absent or compilation fails, the PDF is skipped with a recorded
reason and the run still succeeds -- md and docx are always there. Files are written under
the configured output_dir, nested <output_dir>/<candidate>/<designation>/ with a
self-describing file name (see document_rendering.output_location).
"""

import time
from datetime import datetime
from pathlib import Path

from src.core.logger import get_logger
from src.core.settings import get_config
from src.data_models.job import JobDescription
from src.data_models.rendering import RenderedResumeArtifacts
from src.data_models.resume import Resume
from src.orchestration.state import ResumeEnhancementPipelineState
from src.tools.engines.document_rendering import is_render_available, render_resume_document
from src.tools.engines.document_rendering.docx_renderer import render_resume_docx
from src.tools.engines.document_rendering.markdown_renderer import build_resume_markdown
from src.tools.engines.document_rendering.output_paths import resume_filename, resume_output_dir

logger = get_logger(__name__)


def render_final_resume(state: ResumeEnhancementPipelineState) -> dict:
    """Write the assembled resume to disk (Markdown always, PDF best-effort).

    Reads: optimized_resume (its final_resume) and job_description (for naming).
    Writes: rendered_artifacts.
    Returns: partial state with a RenderedResumeArtifacts.

    Precondition: rehydrate_pii has already restored real PII into final_resume.
    """
    start_time = time.monotonic()
    logger.info(
        "pipeline_stage_started",
        stage="render_final_resume",
        run_id=state["run_id"],
    )
    assert state["optimized_resume"] is not None, "optimized_resume must be set before rendering"
    assert state["job_description"] is not None, "job_description must be set before rendering"
    final_resume = state["optimized_resume"].final_resume
    job = state["job_description"]
    output_dir = resume_output_dir(final_resume, job, Path(get_config().file_paths.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    when = datetime.now()

    markdown_path = _write_markdown(final_resume, job, output_dir, when)
    docx_path = _write_docx(final_resume, job, output_dir, when)
    pdf_path, pdf_skipped_reason = _try_render_pdf(final_resume, job, output_dir, when)
    duration_ms = round((time.monotonic() - start_time) * 1000)
    logger.info(
        "pipeline_stage_completed",
        stage="render_final_resume",
        run_id=state["run_id"],
        duration_ms=duration_ms,
        md_path=str(markdown_path),
        docx_path=str(docx_path),
        pdf_rendered=pdf_path is not None,
    )
    return {
        "rendered_artifacts": RenderedResumeArtifacts(
            markdown_path=markdown_path,
            docx_path=docx_path,
            pdf_path=pdf_path,
            pdf_skipped_reason=pdf_skipped_reason,
        )
    }


def _write_markdown(resume: Resume, job: JobDescription, output_dir: Path, when: datetime) -> str:
    """Write the Markdown resume and return its path (always succeeds)."""
    path = output_dir / resume_filename(resume, job, "md", when)
    path.write_text(build_resume_markdown(resume), encoding="utf-8")
    return str(path)


def _write_docx(resume: Resume, job: JobDescription, output_dir: Path, when: datetime) -> str:
    """Write the Word (.docx) resume and return its path (always succeeds)."""
    path = output_dir / resume_filename(resume, job, "docx", when)
    render_resume_docx(resume, path)
    return str(path)


def _try_render_pdf(
    resume: Resume, job: JobDescription, output_dir: Path, when: datetime
) -> tuple[str | None, str | None]:
    """Render the PDF if possible; otherwise skip with a recorded reason.

    Returns (pdf_path, skipped_reason) where exactly one element is non-None: a path on
    success, or a human-readable reason when the toolchain is missing or compilation fails.
    """
    if not is_render_available():
        reason = "tectonic (LaTeX toolchain) is not installed; run scripts/install-tectonic.sh"
        logger.warning("PDF render skipped", reason=reason)
        return None, reason
    try:
        target = output_dir / resume_filename(resume, job, "pdf", when)
        return str(render_resume_document(resume, target)), None
    except RuntimeError as error:
        logger.warning("PDF render failed", error=str(error))
        return None, f"PDF render failed: {error}"
