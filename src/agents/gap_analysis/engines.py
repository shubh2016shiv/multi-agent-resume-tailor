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
    quality_score = 100

    ####################################################
    # STEP 1: VALIDATE THE FIT SCORE IS IN RANGE
    ####################################################
    if not (0 <= strategy.overall_fit_score <= 100):
        issues.append(f"Fit score out of range: {strategy.overall_fit_score}")
        quality_score -= 40

    ####################################################
    # STEP 2: CHECK THAT MATCHES, GAPS, AND KEYWORDS ARE PRESENT
    ####################################################
    if not strategy.identified_matches:
        warnings.append("No skill matches identified")
        quality_score -= 15
    if not strategy.identified_gaps:
        warnings.append("No skill gaps identified")
        quality_score -= 10
    if not strategy.keywords_to_integrate:
        issues.append("No keywords to integrate")
        quality_score -= 25

    ####################################################
    # STEP 3: CHECK THAT ALL THREE GUIDANCE FIELDS ARE SUBSTANTIVE
    ####################################################
    if (
        not strategy.professional_summary_guidance
        or len(strategy.professional_summary_guidance) < 20
    ):
        issues.append("Professional summary guidance is missing or too brief")
        quality_score -= 15
    if not strategy.experience_guidance or len(strategy.experience_guidance) < 20:
        issues.append("Experience guidance is missing or too brief")
        quality_score -= 15
    if not strategy.skills_guidance or len(strategy.skills_guidance) < 20:
        issues.append("Skills guidance is missing or too brief")
        quality_score -= 15

    ####################################################
    # STEP 4: FLAG LOGICAL CONTRADICTIONS IN THE STRATEGY
    # A high fit score with many gaps, or a low score with no gaps,
    # suggests the agent produced an internally inconsistent strategy.
    ####################################################
    if strategy.overall_fit_score > 90 and len(strategy.identified_gaps) > 5:
        warnings.append("High fit score but many gaps — may be inconsistent")
        quality_score -= 10
    if strategy.overall_fit_score < 50 and len(strategy.identified_gaps) == 0:
        warnings.append("Low fit score but no gaps — may be inconsistent")
        quality_score -= 15

    ####################################################
    # STEP 5: DERIVE THE QUALITY LABEL AND RETURN
    ####################################################
    quality_score = max(0, quality_score)

    if quality_score >= 90:
        quality_label = "excellent"
    elif quality_score >= 70:
        quality_label = "good"
    elif quality_score >= 50:
        quality_label = "fair"
    else:
        quality_label = "poor"

    return {
        "quality": quality_label,
        "score": quality_score,
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
