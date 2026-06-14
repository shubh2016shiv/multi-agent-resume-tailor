"""Contract for the files produced when a tailored resume is rendered.

One run can yield several formats. Markdown is the guaranteed baseline (pure Python,
every OS); PDF is best-effort (it needs the LaTeX toolchain). DOCX is added in a later
phase. The pipeline records paths and skip reasons here so a caller always knows what
was produced and why anything was not.
"""

from pydantic import BaseModel, Field


class RenderedResumeArtifacts(BaseModel):
    """Filesystem paths to the artifacts produced for one tailored resume."""

    markdown_path: str = Field(
        description="Path to the Markdown resume. Always produced (no toolchain needed)."
    )
    docx_path: str = Field(
        description="Path to the Word (.docx) resume. Always produced (python-docx, no toolchain)."
    )
    pdf_path: str | None = Field(
        default=None,
        description="Path to the PDF resume, or None when PDF rendering was skipped or failed.",
    )
    pdf_skipped_reason: str | None = Field(
        default=None,
        description="Why the PDF was not produced (e.g. tectonic missing); None when the PDF exists.",
    )
