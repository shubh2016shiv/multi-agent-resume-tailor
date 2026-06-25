"""Public surface for deterministic resume quality evaluation.

    from src.resume_quality_evaluation import evaluate_resume_truthfulness

Evidence engines detect facts; these evaluators convert that evidence into quality
metrics. The quality decision module combines those metrics using documented product
weights. No LLM-owned score or gate is exposed from this package.
"""

from src.resume_quality_evaluation.job_alignment import evaluate_job_alignment
from src.resume_quality_evaluation.quality_decision import (
    QUALITY_PASS_THRESHOLD,
    apply_resume_quality_gate,
    calculate_overall_quality_score,
    should_render_resume,
)
from src.resume_quality_evaluation.rendered_structure import evaluate_rendered_structure
from src.resume_quality_evaluation.truthfulness import evaluate_resume_truthfulness

__all__ = [
    "QUALITY_PASS_THRESHOLD",
    "apply_resume_quality_gate",
    "calculate_overall_quality_score",
    "evaluate_job_alignment",
    "evaluate_rendered_structure",
    "evaluate_resume_truthfulness",
    "should_render_resume",
]
