"""
Code-owned quality gate for the Quality Assessment agent.

This is the post-hoc, deterministic counterpart to the agent's judgment. The agent
produces the three dimension scores and an overall_quality_score; THIS module decides
pass/fail from that score against a fixed threshold. It is the single authoritative
source for passed_quality_threshold -- the boolean that gates PDF rendering.

Pure: QualityReport in, QualityReport out. No LLM, no I/O. The gate must never be a
heuristic the LLM applies inconsistently; it is one comparison against one constant.
"""

from src.data_models.evaluation import QualityReport

# The pass mark stated in the agent's goal (agents.yaml: quality_assurance_reviewer).
# Kept here as the single code-owned source of truth for the render gate.
QUALITY_PASS_THRESHOLD = 80.0


def apply_quality_gate(
    report: QualityReport, threshold: float = QUALITY_PASS_THRESHOLD
) -> QualityReport:
    """Set passed_quality_threshold authoritatively from the overall score.

    The agent may set passed_quality_threshold itself, but its value is not trusted:
    this function overwrites it with overall_quality_score >= threshold so the render
    gate reads a deterministic, code-owned boolean.

    Expects: a validated QualityReport with overall_quality_score in 0-100.
    Returns: a copy of the report with passed_quality_threshold recomputed.
    """
    passed = report.overall_quality_score >= threshold
    return report.model_copy(update={"passed_quality_threshold": passed})


def should_render_resume(report: QualityReport) -> bool:
    """Return whether the optimized resume is cleared for PDF rendering.

    This is the exact condition the render node's conditional edge reads. It assumes
    apply_quality_gate has already set passed_quality_threshold; it does not recompute.

    Expects: a QualityReport whose gate has been applied.
    Returns: True only when the report passed the quality threshold.
    """
    return report.passed_quality_threshold
