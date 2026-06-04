"""
Keyword coverage analysis: how well a resume covers a job description's keywords.

This is a JD-mode tool. Keywords come from the job description, never from a
static list, so it works for any domain.
"""

import re

from crewai.tools import tool

from src.tools.review_contract.review_models import (
    Confidence,
    Location,
    ReviewComment,
    ReviewResult,
    Section,
    Severity,
)

ENGINE_ID = "keyword_coverage_analyzer"

MIN_KEYWORD_DENSITY = 0.02  # 2 percent: minimum for ATS effectiveness
MAX_KEYWORD_DENSITY = 0.05  # 5 percent: maximum before keyword stuffing


@tool("Calculate Keyword Density")
def calculate_keyword_density(resume_text: str, required_keywords: list[str]) -> str:
    """Analyse keyword usage and density against a list of required keywords.

    Args:
        resume_text: Complete resume content as text.
        required_keywords: Keywords drawn from the job description.

    Returns:
        Formatted density report string. Returns an error line for empty input.
    """
    if not resume_text.strip():
        return "Error: Resume text is empty"
    keyword_counts = _count_keywords(resume_text, required_keywords)
    metrics = _compute_density_metrics(resume_text, required_keywords, keyword_counts)
    return _format_density_report(metrics, required_keywords, keyword_counts)


def analyze_keyword_coverage(resume_text: str, required_keywords: list[str]) -> ReviewResult:
    """Engine surface: same analysis as calculate_keyword_density, as a ReviewResult.

    Args:
        resume_text: Complete resume content as text.
        required_keywords: Keywords drawn from the job description.

    Returns:
        A ReviewResult whose score is the coverage fraction (0.0-1.0). Comments
        flag absent JD keywords (MAJOR) and non-optimal density (MINOR). An empty
        comment list means full coverage at an optimal density.
    """
    if not resume_text.strip():
        return ReviewResult(comments=[], summary="Empty input: nothing to audit")
    if not required_keywords:
        return ReviewResult(comments=[], summary="No keywords provided to match")
    keyword_counts = _count_keywords(resume_text, required_keywords)
    metrics = _compute_density_metrics(resume_text, required_keywords, keyword_counts)
    comments = _build_coverage_comments(metrics, required_keywords, keyword_counts)
    return ReviewResult(
        comments=comments,
        summary=f"{metrics['coverage']:.0%} keyword coverage",
        score=metrics["coverage"],
    )


def get_optimal_keyword_density_range() -> tuple[float, float]:
    """Return the (min, max) optimal keyword density as fractions (0.0 to 1.0)."""
    return (MIN_KEYWORD_DENSITY, MAX_KEYWORD_DENSITY)


def _count_keywords(resume_text: str, required_keywords: list[str]) -> dict[str, int]:
    """Count case-insensitive occurrences of each keyword present in the text."""
    resume_lower = resume_text.lower()
    return {
        keyword: resume_lower.count(keyword.lower())
        for keyword in required_keywords
        if keyword.lower() in resume_lower
    }


def _compute_density_metrics(
    resume_text: str,
    required_keywords: list[str],
    keyword_counts: dict[str, int],
) -> dict:
    """Compute keyword density and coverage metrics.

    Returns a dict with keys: total_words (int), total_instances (int),
    unique_found (int), density (float), coverage (float), is_optimal (bool).
    """
    total_words = len(re.findall(r"\b\w+\b", resume_text.lower()))
    total_instances = sum(keyword_counts.values())
    density = total_instances / total_words if total_words else 0
    return {
        "total_words": total_words,
        "total_instances": total_instances,
        "unique_found": len(keyword_counts),
        "density": density,
        "coverage": len(keyword_counts) / len(required_keywords) if required_keywords else 0,
        "is_optimal": MIN_KEYWORD_DENSITY <= density <= MAX_KEYWORD_DENSITY,
    }


def _build_coverage_comments(
    metrics: dict,
    required_keywords: list[str],
    keyword_counts: dict[str, int],
) -> list[ReviewComment]:
    """Flag absent JD keywords and keyword density outside the optimal band."""
    comments = []
    missing = [keyword for keyword in required_keywords if keyword not in keyword_counts]
    if missing:
        comments.append(
            _make_finding(
                message=f"{len(missing)} of {len(required_keywords)} JD keywords absent",
                severity=Severity.MAJOR,
                advice=f"Work these JD keywords into bullets where true: {', '.join(missing)}.",
            )
        )
    # Only meaningful once at least one keyword is actually present; a 0% density
    # when nothing matched is noise -- the missing-keywords comment already says it.
    if metrics["total_instances"] > 0 and not metrics["is_optimal"]:
        comments.append(
            _make_finding(
                message=(
                    f"Keyword density {metrics['density']:.1%} is outside the optimal "
                    f"{MIN_KEYWORD_DENSITY:.0%}-{MAX_KEYWORD_DENSITY:.0%} band"
                ),
                severity=Severity.MINOR,
                advice="Move keyword usage toward the optimal band; avoid keyword stuffing.",
            )
        )
    return comments


def _make_finding(message: str, severity: Severity, advice: str) -> ReviewComment:
    """Build a document-level coverage comment (mechanical, so HIGH confidence).

    Keyword coverage spans the whole resume, so comments anchor to Section.OTHER.
    """
    return ReviewComment(
        engine_id=ENGINE_ID,
        message=message,
        quoted_text="",
        location=Location(section=Section.OTHER),
        severity=severity,
        confidence=Confidence.HIGH,
        advice=advice,
    )


def _format_density_report(
    metrics: dict,
    required_keywords: list[str],
    keyword_counts: dict[str, int],
) -> str:
    """Render the metrics and keyword breakdown into a readable report string."""
    missing_keywords = [keyword for keyword in required_keywords if keyword not in keyword_counts]
    top_keywords = sorted(keyword_counts.items(), key=lambda item: item[1], reverse=True)[:10]
    status = "[OK] OPTIMAL" if metrics["is_optimal"] else "[!] NEEDS ADJUSTMENT"
    top_lines = "\n".join(f"  {keyword}: {count}x" for keyword, count in top_keywords)
    return (
        f"Keyword Density Analysis:\n"
        f"========================\n"
        f"Total Words: {metrics['total_words']}\n"
        f"Total Keyword Instances: {metrics['total_instances']}\n"
        f"Unique Keywords Found: {metrics['unique_found']}/{len(required_keywords)}\n"
        f"Keyword Density: {metrics['density']:.1%}\n"
        f"Keyword Coverage: {metrics['coverage']:.1%}\n"
        f"Status: {status}\n\n"
        f"Optimal Range: {MIN_KEYWORD_DENSITY:.1%} - {MAX_KEYWORD_DENSITY:.1%}\n\n"
        f"Missing Keywords: {', '.join(missing_keywords) if missing_keywords else 'None'}\n\n"
        f"Top Keywords:\n{top_lines}"
    )
