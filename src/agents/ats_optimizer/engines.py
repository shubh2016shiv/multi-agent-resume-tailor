"""
Post-hoc quality engines for AtsOptimizedResume validation.

These are NOT called during agent execution. They measure the agent's output
after the fact -- the code-owned counterpart to the read-only tools the agent
consults while reasoning. Useful for tests, monitoring, and a QA gate.

All functions are pure: Pydantic in, dict out. They reuse the existing mechanical
ATS engines (no LLM, no I/O) so scoring is deterministic and never self-graded.
"""

from src.agents.ats_optimizer.models import AtsOptimizedResume  # the agent's output contract
from src.data_models.job import JobDescription  # supplies the target ATS keyword list
from src.tools.contracts import ReviewResult, Severity  # shared review-finding shape
from src.tools.engines.ats_compliance import audit_ats_formatting, audit_section_headers
from src.tools.engines.document_rendering.resume_text_renderer import render_resume
from src.tools.engines.job_matching import analyze_keyword_coverage

# Findings at these severities are treated as submission-blocking.
SERIOUS_FINDING_SEVERITIES = {Severity.BLOCKER, Severity.MAJOR}


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
    # ATS engines operate on rendered text, not the Pydantic model, since that's
    # what a real applicant-tracking system would actually parse.
    ####################################################
    resume_text = render_resume(optimized.final_resume)

    ####################################################
    # STEP 2: RUN THE THREE MECHANICAL ATS ENGINES
    ####################################################
    formatting = audit_ats_formatting(resume_text)  # spacing/bullets/fonts-as-text issues
    headers = audit_section_headers(resume_text)  # are section headers ATS-standard?
    coverage = analyze_keyword_coverage(resume_text, job.ats_keywords)  # % of required keywords present

    ####################################################
    # STEP 3: COLLECT BLOCKERS AND BUILD THE RESULT
    ####################################################
    serious_findings = collect_blocker_and_major_findings([formatting, headers, coverage])

    return {
        "overall_status": "pass" if not serious_findings else "needs_review",
        "keyword_coverage": coverage.score,
        "formatting_issues": format_review_comments(formatting),
        "header_issues": format_review_comments(headers),
        "keyword_findings": format_review_comments(coverage),
        "serious_findings": serious_findings,
    }


def format_review_comments(result: ReviewResult) -> list[str]:
    """Format each comment in a ReviewResult as 'SEVERITY: message'."""
    return [f"{comment.severity.value}: {comment.message}" for comment in result.comments]


def collect_blocker_and_major_findings(results: list[ReviewResult]) -> list[str]:
    """Return formatted BLOCKER/MAJOR comment messages across several ReviewResults."""
    serious_messages: list[str] = []
    for result in results:
        for comment in result.comments:
            if comment.severity in SERIOUS_FINDING_SEVERITIES:
                serious_messages.append(f"{comment.severity.value}: {comment.message}")
    return serious_messages
