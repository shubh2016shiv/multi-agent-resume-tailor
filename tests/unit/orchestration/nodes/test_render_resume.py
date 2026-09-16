"""Contracts for final resume document generation."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.data_models.job import JobDescription
from src.data_models.rendering import RenderedResumeArtifacts
from src.data_models.resume import Resume
from src.orchestration.nodes.render_resume import render_final_resume
from src.orchestration.nodes.render_resume.generate_resume_documents import (
    generate_resume_documents,
)


def test_render_node_delegates_document_generation() -> None:
    """The graph node should only read state and delegate document generation."""
    artifacts = RenderedResumeArtifacts(markdown_path="resume.md", docx_path="resume.docx")
    final_resume = object()
    optimized = SimpleNamespace(final_resume=final_resume)
    state = {
        "run_id": "run-1",
        "optimized_resume": optimized,
        "job_description": object(),
    }
    with (
        patch(
            "src.orchestration.nodes.render_resume.render_final_resume_node.get_config"
        ) as config,
        patch(
            "src.orchestration.nodes.render_resume.render_final_resume_node."
            "generate_resume_documents",
            return_value=artifacts,
        ) as generate,
    ):
        config.return_value.file_paths.output_dir = "output"
        result = render_final_resume(state)  # type: ignore[arg-type]

    assert result == {"rendered_artifacts": artifacts}
    generate.assert_called_once_with(
        final_resume, state["job_description"], Path("output"), "run-1"
    )


def test_generation_skips_pdf_when_toolchain_is_unavailable(tmp_path) -> None:
    """Markdown and DOCX should still be returned when PDF cannot be generated."""
    resume = Resume(
        full_name="Candidate",
        email="candidate@example.com",
        phone_number=None,
        location=None,
        website_or_portfolio=None,
        professional_summary="Software engineer.",
    )
    job = JobDescription(
        job_title="Software Engineer",
        company_name="Acme",
        location=None,
        summary="Build software.",
        full_text="Software Engineer at Acme.",
    )
    with (
        patch(
            "src.orchestration.nodes.render_resume.generate_resume_documents.render_resume_docx"
        ) as write_docx,
        patch(
            "src.orchestration.nodes.render_resume.generate_resume_documents.is_render_available",
            return_value=False,
        ),
    ):
        artifacts = generate_resume_documents(resume, job, tmp_path, "run-1")
        retry_artifacts = generate_resume_documents(resume, job, tmp_path, "run-1")
        next_run_artifacts = generate_resume_documents(resume, job, tmp_path, "run-2")

    assert artifacts.markdown_path.endswith("Candidate_Software_Engineer_run_1.md")
    assert artifacts.docx_path.endswith("Candidate_Software_Engineer_run_1.docx")
    assert artifacts.pdf_path is None
    assert artifacts.pdf_skipped_reason
    assert retry_artifacts == artifacts
    assert next_run_artifacts.markdown_path != artifacts.markdown_path
    write_docx.assert_called()
