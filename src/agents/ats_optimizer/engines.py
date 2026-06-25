"""
Post-hoc quality engines for AtsOptimizedResume validation.

These are NOT called during agent execution. They measure the agent's output
after the fact -- the code-owned counterpart to the read-only tools the agent
consults while reasoning. Useful for tests, monitoring, and a QA gate.

All functions are pure: Pydantic in, dict out. They reuse the existing mechanical
ATS engines (no LLM, no I/O) so scoring is deterministic and never self-graded.
"""

from src.agents.ats_optimizer.models import AtsOptimizedResume
from src.data_models.job import JobDescription
from src.tools.contracts import ReviewResult, Severity
from src.tools.engines.ats_compliance import audit_ats_formatting, audit_section_headers
from src.tools.engines.document_rendering.resume_text_renderer import render_resume
from src.tools.engines.job_matching import analyze_keyword_coverage

_SERIOUS_SEVERITIES = {Severity.BLOCKER, Severity.MAJOR}


def check_ats_quality(optimized: AtsOptimizedResume, job: JobDescription) -> dict:
    """Measure an assembled resume against mechanical ATS standards.

    Renders the final resume to text, then runs the formatting, section-header,
    and keyword-coverage engines on it.

    Expects: a validated AtsOptimizedResume and the JobDescription it targeted.
    Returns: dict with overall_status ('pass'/'needs_review'), keyword_coverage
             (0.0-1.0 or None), per-engine issue messages, and the list of
             serious (BLOCKER/MAJOR) findings that would block submission.
    """
    ####################################################
    # STEP 1: RENDER THE FINAL RESUME TO PLAIN TEXT
    ####################################################
    resume_text = render_resume(optimized.final_resume)

    ####################################################
    # STEP 2: RUN THE THREE MECHANICAL ATS ENGINES
    ####################################################
    formatting = audit_ats_formatting(resume_text)
    headers = audit_section_headers(resume_text)
    coverage = analyze_keyword_coverage(resume_text, job.ats_keywords)

    ####################################################
    # STEP 3: COLLECT BLOCKERS AND BUILD THE RESULT
    ####################################################
    serious_findings = _collect_blocker_and_major_findings([formatting, headers, coverage])

    return {
        "overall_status": "pass" if not serious_findings else "needs_review",
        "keyword_coverage": coverage.score,
        "formatting_issues": _format_review_comments(formatting),
        "header_issues": _format_review_comments(headers),
        "keyword_findings": _format_review_comments(coverage),
        "serious_findings": serious_findings,
    }


def _format_review_comments(result: ReviewResult) -> list[str]:
    """Format each comment in a ReviewResult as 'SEVERITY: message'."""
    return [f"{comment.severity.value}: {comment.message}" for comment in result.comments]


def _collect_blocker_and_major_findings(results: list[ReviewResult]) -> list[str]:
    """Return formatted BLOCKER/MAJOR comment messages across several ReviewResults."""
    serious: list[str] = []
    for result in results:
        for comment in result.comments:
            if comment.severity in _SERIOUS_SEVERITIES:
                serious.append(f"{comment.severity.value}: {comment.message}")
    return serious
