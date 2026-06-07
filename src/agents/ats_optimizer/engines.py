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
from src.tools.ats_compliance import audit_ats_formatting, audit_section_headers
from src.tools.job_matching import analyze_keyword_coverage
from src.tools.review_contract.review_models import ReviewResult, Severity
from src.tools.shared.resume_rendering import render_resume

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
    resume_text = render_resume(optimized.final_resume)

    formatting = audit_ats_formatting(resume_text)
    headers = audit_section_headers(resume_text)
    coverage = analyze_keyword_coverage(resume_text, job.ats_keywords)

    serious = _collect_serious_findings([formatting, headers, coverage])

    return {
        "overall_status": "pass" if not serious else "needs_review",
        "keyword_coverage": coverage.score,
        "formatting_issues": _messages(formatting),
        "header_issues": _messages(headers),
        "keyword_findings": _messages(coverage),
        "serious_findings": serious,
    }


def _messages(result: ReviewResult) -> list[str]:
    """Return each comment in a ReviewResult as 'SEVERITY: message'."""
    return [f"{comment.severity.value}: {comment.message}" for comment in result.comments]


def _collect_serious_findings(results: list[ReviewResult]) -> list[str]:
    """Return BLOCKER/MAJOR comment messages across several ReviewResults."""
    serious: list[str] = []
    for result in results:
        for comment in result.comments:
            if comment.severity in _SERIOUS_SEVERITIES:
                serious.append(f"{comment.severity.value}: {comment.message}")
    return serious
