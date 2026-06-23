"""Deterministic ATS grading: does the REAL rendered resume carry its section headers?

This rubric is the fix for the ATS root cause: the QA agent was checking its lossy
TOON context (where headers look like `[Skills]`), which the header validator's regex
can never match -- so it reported "missing headers" on every run, a structural false
positive. Here we build the actual artifact (`build_resume_tex`), project its real
`\\section*{TITLE}` macros into the plain standalone header lines the validator was
designed to read, and audit those. No LLM: the verdict is mechanical and authoritative.

Scope is deliberately narrow. We check section-header presence only. We do NOT run the
formatting/multi-column heuristics here: those exist to flag ATS problems when INGESTING
an uploaded resume, and would only produce noise against our own code-generated output
from a provably single-column template (resume.tex.j2 is `extarticle`, one column).
"""

import re

from src.core.logger import get_logger
from src.data_models.evaluation import AtsCheckStatus, AtsRenderedOutcome
from src.data_models.resume import Resume
from src.tools.contracts import Severity
from src.tools.engines.ats_compliance import audit_section_headers
from src.tools.engines.document_rendering.resume_renderer import build_resume_tex

logger = get_logger(__name__)

# A FAIL credits no ATS compatibility for the structural dimension; each missing
# essential header is a hard parsing blocker, so any FAIL collapses the score to 0.
# (The hard gate in the QA node is what actually blocks release; this score only feeds
# the weighted overall blend so the number the user sees is honest, not inflated.)
_FAIL_ATS_SCORE = 0.0
_PASS_ATS_SCORE = 100.0
_INCONCLUSIVE_ATS_SCORE = 0.0

# The .tex template emits each section as `\section*{TITLE}`. Project the TITLE out so
# the text-based header validator sees one standalone header line per section.
_SECTION_TITLE_PATTERN = re.compile(r"\\section\*\{([^}]+)\}")


def grade_ats(final_resume: Resume) -> AtsRenderedOutcome:
    """Grade ATS compatibility by auditing the rendered .tex artifact's section headers.

    Expects the final, assembled resume (the one that will become the PDF).
    Returns an AtsRenderedOutcome:
      - INCONCLUSIVE if the .tex cannot be built (we cannot inspect what we cannot render,
        so we never silently PASS -- this routes to secondary/human review downstream).
      - FAIL if any essential section header (experience/education/skills) is missing.
      - PASS otherwise, with ats_score 100.
    """
    # Build the real artifact. Any build failure means we cannot verify the resume, which
    # is INCONCLUSIVE -- an explicit "cannot confirm" sentinel, never a guessed PASS. We
    # catch broad Exception on purpose: whatever build_resume_tex raises (template, data,
    # escaping), the correct ATS verdict is the same "could not inspect" outcome.
    try:
        tex = build_resume_tex(final_resume)
    except Exception as error:  # noqa: BLE001 -- any build failure maps to INCONCLUSIVE
        logger.warning("ATS check inconclusive: could not build .tex", error=str(error))
        return AtsRenderedOutcome(
            status=AtsCheckStatus.INCONCLUSIVE,
            violations=[],
            ats_score=_INCONCLUSIVE_ATS_SCORE,
            detail=f"Could not render the resume to .tex for inspection ({error}).",
        )

    projected_headers = _project_section_titles(tex)
    header_review = audit_section_headers(projected_headers)
    # Only MAJOR comments are missing ESSENTIAL sections; SUGGESTION comments are missing
    # OPTIONAL ones and must not fail the gate.
    violations = [
        comment.message for comment in header_review.comments if comment.severity is Severity.MAJOR
    ]
    if violations:
        return AtsRenderedOutcome(
            status=AtsCheckStatus.FAIL,
            violations=violations,
            ats_score=_FAIL_ATS_SCORE,
            detail=f"Rendered resume is missing {len(violations)} essential section header(s).",
        )
    return AtsRenderedOutcome(
        status=AtsCheckStatus.PASS,
        violations=[],
        ats_score=_PASS_ATS_SCORE,
        detail="Rendered resume carries every essential section header.",
    )


def _project_section_titles(tex: str) -> str:
    """Extract each `\\section*{TITLE}` from the .tex as a standalone TITLE line.

    The header validator audits plain header lines (e.g. "SKILLS"), not LaTeX macros, so
    this bridges the two: it returns the section titles, one per line, in document order.
    Expects a complete .tex string. Returns the titles joined by newlines (empty string
    if the document declares no sections).
    """
    titles = _SECTION_TITLE_PATTERN.findall(tex)
    return "\n".join(titles)
