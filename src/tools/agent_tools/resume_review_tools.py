"""Agent-facing review tools that the LLM is allowed to call."""

from collections.abc import Callable

from crewai.tools import tool
from pydantic import ValidationError

from src.data_models.resume import Resume
from src.tools.contracts import ReviewResult
from src.tools.engines.ats_compliance import audit_ats_formatting, audit_section_headers
from src.tools.engines.job_matching import analyze_keyword_coverage
from src.tools.engines.resume_diagnostics import audit_summary_quality
from src.tools.engines.truthfulness import (
    detect_claim_inflation,
    detect_rewrite_drift,
    validate_skills_evidence,
)


def render_review_result(review_result: ReviewResult, title: str) -> str:
    """Turn a structured review result into plain text for the agent."""
    lines = [f"=== {title} ===", review_result.summary or "(no summary)"]
    if review_result.score is not None:
        lines.append(f"Score: {review_result.score:.2f}")
    if not review_result.comments:
        lines.append("No issues found.")

    for comment in review_result.comments:
        lines.append(
            f"[{comment.severity.value}/{comment.confidence.value}] "
            f"({comment.location.section.value}) {comment.message}"
        )
        lines.append(f"    advice: {comment.advice}")
        if comment.proposed_rewrite:
            lines.append(f"    rewrite: {comment.proposed_rewrite}")
    return "\n".join(lines)


def merge_review_results(review_results: list[ReviewResult]) -> ReviewResult:
    """Combine several review results into one agent-facing result."""
    all_comments = [
        comment for review_result in review_results for comment in review_result.comments
    ]
    combined_summary = "; ".join(
        review_result.summary for review_result in review_results if review_result.summary
    )
    first_score = next(
        (
            review_result.score
            for review_result in review_results
            if review_result.score is not None
        ),
        None,
    )
    return ReviewResult(comments=all_comments, summary=combined_summary, score=first_score)


def run_resume_review_tool(
    resume_json: str,
    review_runner: Callable[[Resume], ReviewResult],
    title: str,
) -> str:
    """Parse resume JSON, run one review function, and render the result."""
    try:
        resume = Resume.model_validate_json(resume_json)
    except ValidationError as error:
        return f"Error: could not parse resume JSON ({error.error_count()} validation error(s))."
    return render_review_result(review_runner(resume), title)


@tool("Audit Summary Quality")
def audit_summary(resume_json: str) -> str:
    """Review the professional summary for obvious quality issues."""
    return run_resume_review_tool(resume_json, audit_summary_quality, "Summary Quality")


@tool("Check Skills Evidence")
def check_skills_evidence(resume_json: str) -> str:
    """Flag listed skills that the resume does not support with evidence."""
    return run_resume_review_tool(resume_json, validate_skills_evidence, "Skills Evidence")


@tool("Audit Truthfulness")
def audit_truthfulness(original_resume_json: str, revised_resume_json: str) -> str:
    """Compare the original and revised resumes for invented or drifted claims."""
    try:
        original_resume = Resume.model_validate_json(original_resume_json)
        revised_resume = Resume.model_validate_json(revised_resume_json)
    except ValidationError as error:
        return f"Error: could not parse resume JSON ({error.error_count()} validation error(s))."

    merged_review = merge_review_results(
        [
            detect_claim_inflation(original_resume, revised_resume),
            detect_rewrite_drift(original_resume, revised_resume),
        ]
    )
    return render_review_result(merged_review, "Truthfulness")


@tool("Validate ATS Compliance")
def validate_ats_compliance(resume_text: str) -> str:
    """Check ATS safety by combining formatting and section-header checks."""
    merged_review = merge_review_results(
        [audit_ats_formatting(resume_text), audit_section_headers(resume_text)]
    )
    return render_review_result(merged_review, "ATS Compliance")


@tool("Analyze Keyword Coverage")
def analyze_jd_keyword_coverage(resume_text: str, keywords_csv: str) -> str:
    """Explain how well the resume covers the requested job keywords."""
    clean_keywords = [keyword.strip() for keyword in keywords_csv.split(",") if keyword.strip()]
    keyword_review = analyze_keyword_coverage(resume_text, clean_keywords)
    return render_review_result(keyword_review, "Keyword Coverage")
