"""
Agent-facing tools: the coarse instruments agents actually call.

Agents receive ~7 of these, not the ~19 underlying engines. Each tool composes one
or more engines, merges their `ReviewResult`s, and renders a single readable string
(CrewAI tools return strings to the LLM). Inputs arrive as JSON/text from the agent,
so each tool parses defensively and reports a clear error instead of crashing.

The engines stay pure and typed (`Resume`/`JobDescription` in, `ReviewResult` out);
this layer is the serialization + composition + rendering boundary between them and
the CrewAI agents.
"""

from collections.abc import Callable

from crewai.tools import tool
from pydantic import ValidationError

from src.data_models.job import JobDescription
from src.data_models.resume import Resume
from src.tools.ats_compliance import audit_ats_formatting, audit_section_headers
from src.tools.document_ingestion import (
    audit_extraction_quality,
    convert_document_to_markdown,
    extract_resume,
    redact_pii,
)
from src.tools.job_matching import analyze_keyword_coverage, match_requirements
from src.tools.resume_diagnostics import audit_summary_quality
from src.tools.review_contract.review_models import ReviewResult
from src.tools.shared.resume_rendering import render_resume
from src.tools.truthfulness import (
    detect_claim_inflation,
    detect_rewrite_drift,
    validate_skills_evidence,
)

# ── rendering + composition helpers ──────────────────────────────────────────


def _render_review_result(result: ReviewResult, title: str) -> str:
    """Render a ReviewResult as an agent-readable report string."""
    lines = [f"=== {title} ===", result.summary or "(no summary)"]
    if result.score is not None:
        lines.append(f"Score: {result.score:.2f}")
    if not result.comments:
        lines.append("No issues found.")
    for comment in result.comments:
        location = comment.location.section.value
        lines.append(
            f"[{comment.severity.value}/{comment.confidence.value}] "
            f"({location}) {comment.message}"
        )
        lines.append(f"    advice: {comment.advice}")
        if comment.proposed_rewrite:
            lines.append(f"    rewrite: {comment.proposed_rewrite}")
    return "\n".join(lines)


def _merge(results: list[ReviewResult]) -> ReviewResult:
    """Concatenate several engines' comments into one ReviewResult.

    Carries forward the first engine's score (the lead engine in a composite owns
    the headline metric — e.g. requirements_matcher's must-have coverage); without
    this the merged result silently drops every score.
    """
    comments = [comment for result in results for comment in result.comments]
    summary = "; ".join(result.summary for result in results if result.summary)
    score = next((result.score for result in results if result.score is not None), None)
    return ReviewResult(comments=comments, summary=summary, score=score)


def _run_resume_tool(resume_json: str, runner: Callable[[Resume], ReviewResult], title: str) -> str:
    """Parse a Resume from JSON, run a review, and render it; report parse errors."""
    try:
        resume = Resume.model_validate_json(resume_json)
    except ValidationError as error:
        return f"Error: could not parse resume JSON ({error.error_count()} validation error(s))."
    return _render_review_result(runner(resume), title)


# ── agent-facing tools ───────────────────────────────────────────────────────


@tool("Audit Summary Quality")
def audit_summary(resume_json: str) -> str:
    """Review the professional summary for length, first-person voice, and generic phrasing.

    Args:
        resume_json: A Resume serialized as JSON.

    Returns:
        The summary-quality report.
    """
    return _run_resume_tool(resume_json, audit_summary_quality, "Summary Quality")


@tool("Check Skills Evidence")
def check_skills_evidence(resume_json: str) -> str:
    """Flag listed skills that nothing in the resume evidences.

    Args:
        resume_json: A Resume serialized as JSON.

    Returns:
        The skills-evidence report.
    """
    return _run_resume_tool(resume_json, validate_skills_evidence, "Skills Evidence")


@tool("Audit Truthfulness")
def audit_truthfulness(original_resume_json: str, revised_resume_json: str) -> str:
    """Compare an original and a revised resume for invented facts and semantic drift.

    Args:
        original_resume_json: The original Resume as JSON (source of truth).
        revised_resume_json: The rewritten Resume as JSON.

    Returns:
        A combined report from the claim-inflation (mechanical) and rewrite-drift
        (judgment) engines.
    """
    try:
        original = Resume.model_validate_json(original_resume_json)
        revised = Resume.model_validate_json(revised_resume_json)
    except ValidationError as error:
        return f"Error: could not parse resume JSON ({error.error_count()} validation error(s))."
    merged = _merge([detect_claim_inflation(original, revised), detect_rewrite_drift(original, revised)])
    return _render_review_result(merged, "Truthfulness")


@tool("Match Job Requirements")
def match_job_requirements(resume_json: str, job_json: str) -> str:
    """Classify how well the resume evidences each job requirement, plus keyword coverage.

    Args:
        resume_json: A Resume serialized as JSON.
        job_json: A JobDescription serialized as JSON.

    Returns:
        A combined report from the requirements-matcher (judgment) and keyword-coverage
        (mechanical) engines.
    """
    try:
        resume = Resume.model_validate_json(resume_json)
        job = JobDescription.model_validate_json(job_json)
    except ValidationError as error:
        return f"Error: could not parse input JSON ({error.error_count()} validation error(s))."
    merged = _merge(
        [
            match_requirements(resume, job),
            analyze_keyword_coverage(render_resume(resume), job.ats_keywords),
        ]
    )
    return _render_review_result(merged, "Job Requirements Match")


@tool("Validate ATS Compliance")
def validate_ats_compliance(resume_text: str) -> str:
    """Check resume text for ATS-breaking formatting and missing standard section headers.

    Args:
        resume_text: The resume as plain text or Markdown.

    Returns:
        A combined report from the formatting and section-header engines.
    """
    merged = _merge([audit_ats_formatting(resume_text), audit_section_headers(resume_text)])
    return _render_review_result(merged, "ATS Compliance")


@tool("Analyze Keyword Coverage")
def analyze_jd_keyword_coverage(resume_text: str, keywords_csv: str) -> str:
    """Report how well the resume covers a comma-separated list of job keywords.

    Args:
        resume_text: The resume as plain text.
        keywords_csv: Job keywords as a comma-separated string.

    Returns:
        The keyword-coverage report (coverage %, density, missing keywords).
    """
    keywords = [keyword.strip() for keyword in keywords_csv.split(",") if keyword.strip()]
    return _render_review_result(
        analyze_keyword_coverage(resume_text, keywords), "Keyword Coverage"
    )


@tool("Convert Resume Document to Markdown")
def convert_resume_document_to_markdown(file_path: str) -> str:
    """Convert a resume PDF or DOCX file to Markdown text.

    Call this first. Pass the file path. The Markdown it returns is what
    all subsequent steps (quality check, PII redaction, extraction) operate on.

    Args:
        file_path: Absolute or relative path to the resume PDF/DOCX file.

    Returns:
        The document content as a Markdown string.
    """
    return convert_document_to_markdown(file_path)


@tool("Redact PII from Resume Markdown")
def redact_pii_from_resume_markdown(markdown: str) -> str:
    """Mask personally identifiable information before sending text to an LLM.

    Call this after the quality check passes. Returns the redacted Markdown
    ready for extraction. PII (name, email, phone) is replaced with placeholders.

    Args:
        markdown: The Markdown text from convert_resume_document_to_markdown.

    Returns:
        The Markdown with PII replaced by placeholder tokens.
    """
    redacted_markdown, _pii_mapping = redact_pii(markdown)
    return redacted_markdown


@tool("Extract Structured Resume from Markdown")
def extract_structured_resume_from_markdown(redacted_markdown: str) -> str:
    """Extract a validated Resume object from PII-redacted Markdown.

    Call this last, after redaction. Uses a schema-constrained LLM call to
    produce a Resume that matches the exact Resume data model fields.

    Args:
        redacted_markdown: The redacted Markdown from redact_pii_from_resume_markdown.

    Returns:
        The extracted Resume as a JSON string matching the Resume schema.
    """
    resume = extract_resume(redacted_markdown)
    return resume.model_dump_json()


@tool("Check Resume Markdown Quality")
def check_resume_markdown_quality(markdown: str) -> str:
    """Inspect converted resume Markdown for extraction blockers.

    Use this after PDF conversion, before extracting. A BLOCKER means the
    conversion failed and the resume cannot be reliably extracted — stop
    and report the problem. An empty result means the text is usable.

    Args:
        markdown: The Markdown text produced from the resume PDF/DOCX.

    Returns:
        A quality report listing issues found (volume, fragmentation,
        conversion artifacts). Empty result means no issues found.
    """
    return _render_review_result(
        audit_extraction_quality(markdown), "Resume Markdown Quality"
    )
