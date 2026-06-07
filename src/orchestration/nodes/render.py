"""Stage 6 final PDF render node.

Plain code, no agent: rendering an already-decided Resume into a PDF is mechanical.
This node only runs when the quality gate passed (see the conditional edge in
graph.py); it does not re-check the gate.
"""

from pathlib import Path

from src.orchestration.state import ResumeEnhancementPipelineState
from src.tools.document_rendering import is_render_available, render_resume_document

_OUTPUT_DIR = Path("output")


def render_final_resume(state: ResumeEnhancementPipelineState) -> dict:
    """Render the assembled resume to a PDF and record its path.

    Reads: optimized_resume (its final_resume) and resume_path (for the file name).
    Writes: rendered_resume_path.
    Returns: partial state with the PDF path.
    Raises: RuntimeError if the LaTeX toolchain is unavailable.

    TODO: rehydrate PII placeholders before rendering.
          Proposed: apply the redaction map captured at extraction time.
          Deferred: the extraction path does not yet thread the PII map through state.
    """
    if not is_render_available():
        raise RuntimeError(
            "Cannot render PDF: the LaTeX toolchain (tectonic) is not installed. "
            "Install it, or skip rendering."
        )

    final_resume = state["optimized_resume"].final_resume
    output_path = _build_output_path(state["resume_path"])
    rendered = render_resume_document(final_resume, output_path)
    return {"rendered_resume_path": str(rendered)}


def _build_output_path(source_resume_path: str) -> Path:
    """Derive the output PDF path from the source resume's file name.

    Returns: output/<source-stem>_optimized.pdf, with the directory created.
    """
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(source_resume_path).stem or "resume"
    return _OUTPUT_DIR / f"{stem}_optimized.pdf"
