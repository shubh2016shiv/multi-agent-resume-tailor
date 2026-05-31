"""
Keyword coverage analysis: how well a resume covers a job description's keywords.

This is a JD-mode tool. Keywords come from the job description, never from a
static list, so it works for any domain.
"""

import re

from crewai.tools import tool

# TODO: Return a ReviewResult instead of a formatted string.
#       Proposed: rename to analyze_keyword_coverage() returning ReviewResult,
#                 per the tooling plan, once judgment tools migrate to the contract.
#       Deferred: the contract migration is a separate feature; keep the string
#                 return now so the ATS agent that calls this keeps working.

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
