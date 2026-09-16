"""Deterministically calculate and re-grade resume quality scores."""

from src.core.logger import get_logger
from src.data_models.evaluation import (
    AtsCheckStatus,
    ATSMetrics,
    JobAlignmentEvaluation,
    QualityFeedback,
    RenderedStructureEvaluation,
    ResumeQualityReport,
    TruthfulnessEvaluation,
)
from src.data_models.job import JobDescription
from src.data_models.resume import Resume
from src.resume_quality_evaluation import (
    apply_release_hard_blocks,
    apply_resume_quality_gate,
    calculate_overall_quality_score,
    evaluate_job_alignment,
    evaluate_rendered_structure,
    evaluate_resume_truthfulness,
)

logger = get_logger(__name__)


def ground_quality_scores(
    original_resume: Resume,
    revised_resume: Resume,
    job: JobDescription,
) -> tuple[ResumeQualityReport, RenderedStructureEvaluation]:
    """Build the quality report from deterministic accuracy, relevance, and ATS checks."""
    accuracy = evaluate_resume_truthfulness(original_resume, revised_resume)
    relevance = evaluate_job_alignment(revised_resume, job)
    structure = evaluate_rendered_structure(revised_resume)
    ats = _build_ats_metrics(structure, relevance.ats_keyword_coverage)
    overall = calculate_overall_quality_score(
        accuracy.accuracy_score,
        relevance.relevance_score,
        structure.ats_score,
    )
    feedback = _build_quality_feedback(accuracy, relevance, structure, overall)
    report = ResumeQualityReport(
        accuracy=accuracy,
        relevance=relevance,
        ats_optimization=ats,
        overall_quality_score=overall,
        passes_quality_gate=False,
        assessment_summary=feedback.assessment_summary,
        feedback_for_improvement=feedback.feedback_for_improvement,
    )
    report = apply_resume_quality_gate(report)
    if not report.passes_quality_gate and feedback.feedback_for_improvement is None:
        report = report.model_copy(
            update={
                "feedback_for_improvement": (
                    "Review the dimension justifications that fall below the quality gate."
                )
            }
        )
    _log_quality_scores(report)
    return apply_release_hard_blocks(report, structure), structure


def _build_quality_feedback(
    accuracy: TruthfulnessEvaluation,
    relevance: JobAlignmentEvaluation,
    structure: RenderedStructureEvaluation,
    overall: float,
) -> QualityFeedback:
    """Build concise feedback directly from deterministic findings."""
    issues = [
        *(f"Unsupported claim: {claim}" for claim in accuracy.exaggerated_claims),
        *(f"Unsupported skill: {skill}" for skill in accuracy.unsupported_skills),
        *(f"Missing requirement: {requirement}" for requirement in relevance.missed_requirements),
        *(f"ATS issue: {violation}" for violation in structure.violations),
    ]
    if not relevance.is_conclusive:
        issues.append("Job alignment could not be scored conclusively.")
    if structure.status is not AtsCheckStatus.PASS and not structure.violations:
        issues.append(structure.detail)
    return QualityFeedback(
        assessment_summary=(
            f"Deterministic quality scores: accuracy {accuracy.accuracy_score:.1f}, "
            f"relevance {relevance.relevance_score:.1f}, ATS {structure.ats_score:.1f}, "
            f"overall {overall:.1f}."
        ),
        feedback_for_improvement="\n".join(f"- {issue}" for issue in issues) or None,
    )


def _build_ats_metrics(
    structure: RenderedStructureEvaluation,
    keyword_coverage: float,
) -> ATSMetrics:
    """Translate rendered-structure results into the report's ATS dimension."""
    return ATSMetrics(
        ats_score=structure.ats_score,
        keyword_coverage=keyword_coverage,
        formatting_issues=structure.violations,
        justification=structure.detail,
    )


def _log_quality_scores(report: ResumeQualityReport) -> None:
    """Record the grounded scores and resulting gate decision."""
    logger.info(
        "quality_scores_computed",
        accuracy_score=report.accuracy.accuracy_score,
        relevance_score=report.relevance.relevance_score,
        ats_score=report.ats_optimization.ats_score,
        overall_score=report.overall_quality_score,
        passes_gate=report.passes_quality_gate,
    )


__all__ = ["ground_quality_scores"]
