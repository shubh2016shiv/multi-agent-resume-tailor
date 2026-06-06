"""
Post-hoc quality engines for AlignmentStrategy validation.

These are NOT called during agent execution. They validate the agent's
output after the fact — useful for tests, monitoring, and QA pipelines.

All functions are pure: Pydantic in, dict out. No framework, no LLM, no I/O.
"""

from src.data_models.strategy import AlignmentStrategy

# ── quality check ─────────────────────────────────────────────────────────────


def check_strategy_quality(strategy: AlignmentStrategy) -> dict:
    """Score the completeness and consistency of an AlignmentStrategy.

    Checks: score range, presence of matches/gaps/keywords, guidance length,
    and logical consistency (high score + many gaps is suspicious).

    Expects: a validated AlignmentStrategy.
    Returns: dict with quality (str), score (int 0-100), issues, warnings.
    """
    issues: list[str] = []
    warnings: list[str] = []
    score = 100

    # score range
    if not (0 <= strategy.overall_fit_score <= 100):
        issues.append(f"Fit score out of range: {strategy.overall_fit_score}")
        score -= 40

    # presence checks
    if not strategy.identified_matches:
        warnings.append("No skill matches identified")
        score -= 15
    if not strategy.identified_gaps:
        warnings.append("No skill gaps identified")
        score -= 10
    if not strategy.keywords_to_integrate:
        issues.append("No keywords to integrate")
        score -= 25

    # guidance length
    for field_name, label, deduct in [
        ("professional_summary_guidance", "Professional summary guidance", 15),
        ("experience_guidance", "Experience guidance", 15),
        ("skills_guidance", "Skills guidance", 15),
    ]:
        value = getattr(strategy, field_name, "")
        if not value or len(value) < 20:
            issues.append(f"{label} is missing or too brief")
            score -= deduct

    # logical consistency
    if strategy.overall_fit_score > 90 and len(strategy.identified_gaps) > 5:
        warnings.append("High fit score but many gaps — may be inconsistent")
        score -= 10
    if strategy.overall_fit_score < 50 and len(strategy.identified_gaps) == 0:
        warnings.append("Low fit score but no gaps — may be inconsistent")
        score -= 15

    score = max(0, score)

    if score >= 90:
        quality = "excellent"
    elif score >= 70:
        quality = "good"
    elif score >= 50:
        quality = "fair"
    else:
        quality = "poor"

    return {
        "quality": quality,
        "score": score,
        "issues": issues,
        "warnings": warnings,
    }


# ── coverage stats ────────────────────────────────────────────────────────────


def calculate_coverage_stats(strategy: AlignmentStrategy) -> dict:
    """Derive summary statistics from an AlignmentStrategy.

    Expects: a validated AlignmentStrategy.
    Returns: dict with total_matches, total_gaps, keywords_to_integrate,
             coverage_ratio, and fit_score.
    """
    total_matches = len(strategy.identified_matches)
    total_gaps = len(strategy.identified_gaps)
    total_keywords = len(strategy.keywords_to_integrate)

    total_requirements = total_matches + total_gaps
    coverage_ratio = total_matches / total_requirements if total_requirements > 0 else 0.0

    return {
        "total_matches": total_matches,
        "total_gaps": total_gaps,
        "keywords_to_integrate": total_keywords,
        "coverage_ratio": round(coverage_ratio, 2),
        "fit_score": round(strategy.overall_fit_score, 1),
    }
