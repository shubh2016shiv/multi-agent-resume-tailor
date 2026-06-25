"""Resume rendering capabilities."""

from .docx_renderer import render_resume_docx
from .markdown_renderer import build_resume_markdown
from .output_paths import resume_filename, resume_output_dir
from .resume_renderer import build_resume_tex, render_resume_document
from .section_layout import RenderProfile
from .sidecar import is_render_available

__all__ = [
    "RenderProfile",
    "build_resume_markdown",
    "build_resume_tex",
    "is_render_available",
    "render_resume_docx",
    "render_resume_document",
    "resume_filename",
    "resume_output_dir",
]
