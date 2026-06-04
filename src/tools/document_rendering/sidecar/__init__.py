"""
Rendering sidecar: the toolchain-dependent compilation concern, isolated.

Everything else in document_rendering/ is pure Python (escaping, layout policy,
templating) and runs anywhere. Turning LaTeX into a PDF needs an external runtime
-- the `tectonic` binary -- which is a cross-cutting infrastructure dependency, not
application logic. Following the sidecar pattern, that concern lives behind this
small, swappable interface:

    is_render_available() -> bool
    compile_tex_to_pdf(tex_source, output_path) -> Path

The default implementation shells out to a co-located tectonic binary (baked into
the image with a pre-warmed cache). Because the renderer depends only on this
interface, the backend can later become a real out-of-process sidecar (e.g. a
tectonic microservice) without any change to the rendering core.
"""

from .latex_compiler import compile_tex_to_pdf, is_render_available

__all__ = [
    "compile_tex_to_pdf",
    "is_render_available",
]
