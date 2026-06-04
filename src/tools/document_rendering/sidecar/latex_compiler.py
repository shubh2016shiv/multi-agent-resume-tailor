"""
Tectonic compilation backend for the rendering sidecar.

Tectonic is a self-contained LaTeX engine: reproducible, fetches packages on
demand, no system texlive required. This is the only module in the renderer that
touches the external toolchain; the pure build_resume_tex path never imports it,
so escaping/policy/templating stay testable without any TeX installed.
"""

import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

TECTONIC_BINARY = "tectonic"
COMPILE_TIMEOUT_SECONDS = 60


def is_render_available() -> bool:
    """Return True if the tectonic binary is on PATH (PDF rendering is possible)."""
    return shutil.which(TECTONIC_BINARY) is not None


def compile_tex_to_pdf(tex_source: str, output_path: Path) -> Path:
    """Compile a LaTeX document to a PDF at output_path and return it.

    Args:
        tex_source: A complete LaTeX document (from build_resume_tex).
        output_path: Where to write the .pdf; parent directories are created.

    Returns:
        output_path, now pointing at a non-empty PDF.

    Raises:
        RuntimeError: If tectonic is missing, times out, exits non-zero, or
            produces no PDF. The tectonic log is included on a compile failure.
    """
    if not is_render_available():
        raise RuntimeError(
            f"'{TECTONIC_BINARY}' not found on PATH. Install it (e.g. "
            "'brew install tectonic', 'conda install -c conda-forge tectonic', or "
            "the release binary) to render PDFs."
        )
    with TemporaryDirectory() as workdir:
        tex_path = Path(workdir) / "resume.tex"
        tex_path.write_text(tex_source, encoding="utf-8")
        result = _run_tectonic(tex_path)
        produced = tex_path.with_suffix(".pdf")
        if result.returncode != 0 or not produced.exists() or produced.stat().st_size == 0:
            raise RuntimeError(f"tectonic failed to produce a PDF:\n{result.stderr}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(produced, output_path)
    return output_path


def _run_tectonic(tex_path: Path) -> subprocess.CompletedProcess:
    """Invoke tectonic on tex_path, writing output beside it; raise on timeout."""
    try:
        return subprocess.run(
            [TECTONIC_BINARY, "--outdir", str(tex_path.parent), str(tex_path)],
            capture_output=True,
            text=True,
            timeout=COMPILE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"tectonic timed out after {COMPILE_TIMEOUT_SECONDS}s") from exc
