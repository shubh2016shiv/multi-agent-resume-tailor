"""
Post-hoc quality engines for ProfessionalSummary validation.

These are NOT called during agent execution. They validate the agent's
output after the fact — useful for tests, monitoring, and QA pipelines.

All functions are pure: Pydantic in, dict out. No framework, no LLM, no I/O.
"""

from src.agents.professional_summary.models import ProfessionalSummary
from src.data_models.strategy import AlignmentStrategy

# ── quality check ─────────────────────────────────────────────────────────────

# Mirrors the summary-writing task contract in src/config/tasks/professional_summary.yaml.
MIN_SUMMARY_WORDS = 80
MAX_SUMMARY_WORDS = 110


def check_summary_quality(summary: ProfessionalSummary, strategy: AlignmentStrategy) -> dict:
    """Score the quality of each draft in a ProfessionalSummary.

    Checks: word count range and supported role vocabulary presence.

    Expects: a validated ProfessionalSummary and the AlignmentStrategy that guided it.
    Returns: dict with per-draft evaluations and overall status.
    """
    evaluations: list[dict] = []

    for draft in summary.drafts:
        score = 100
        issues: list[str] = []
        warnings: list[str] = []

        word_count = len(draft.content.split())

        # word count
        if word_count < MIN_SUMMARY_WORDS:
            issues.append(f"Too short ({word_count} words)")
            score -= 30
        elif word_count > MAX_SUMMARY_WORDS:
            issues.append(f"Too long ({word_count} words)")
            score -= 25

        # supported role vocabulary presence (top 5 from strategy)
        required = {k.lower() for k in strategy.keywords_to_integrate[:5]}
        draft_lower = draft.content.lower()
        missing = [k for k in required if k not in draft_lower]
        integrated_count = len(required) - len(missing)

        if required and integrated_count == 0:
            issues.append(f"Summary loses role relevance; no supported role vocabulary present: {missing}")
            score -= 20
        elif missing:
            warnings.append(f"Some supported role vocabulary is absent: {missing}")
            score -= 5

        score = max(0, score)

        if score >= 90:
            quality = "excellent"
        elif score >= 75:
            quality = "good"
        elif score >= 60:
            quality = "fair"
        else:
            quality = "poor"

        evaluations.append(
            {
                "version": draft.version_name,
                "quality": quality,
                "score": score,
                "word_count": word_count,
                "issues": issues,
                "warnings": warnings,
            }
        )

    return {
        "overall_status": "complete",
        "draft_count": len(summary.drafts),
        "recommended": summary.recommended_version,
        "evaluations": evaluations,
    }


# ── keyword integration ───────────────────────────────────────────────────────


def analyze_keyword_integration(summary_text: str, required_keywords: list[str]) -> dict:
    """Measure how many required keywords appear in a summary text.

    Expects: a summary string and a list of keywords.
    Returns: dict with total_required, total_integrated, missing_keywords,
             integration_rate (0.0-1.0).
    """
    summary_lower = summary_text.lower()
    integrated: list[str] = []
    missing: list[str] = []

    for kw in required_keywords:
        if kw.lower() in summary_lower:
            integrated.append(kw)
        else:
            missing.append(kw)

    total = len(required_keywords)
    rate = len(integrated) / total if total > 0 else 0.0

    return {
        "total_required": total,
        "total_integrated": len(integrated),
        "integrated_keywords": integrated,
        "missing_keywords": missing,
        "integration_rate": round(rate, 2),
    }
