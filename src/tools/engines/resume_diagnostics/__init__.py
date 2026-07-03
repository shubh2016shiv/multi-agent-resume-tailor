"""Resume quality checks that inspect writing quality and structure."""

from src.data_models.resume import Experience
from src.tools.contracts import ReviewResult

from .bullet_structure import audit_bullet_structure, audit_bullet_structure_for_experiences
from .consistency_checks import audit_consistency, audit_consistency_for_experiences
from .experience_rewrite_quality import audit_experience_rewrite_quality
from .language_quality import audit_language_quality, audit_language_quality_for_experiences
from .quantification import audit_quantification, audit_quantification_for_experiences
from .summary_quality import audit_summary_quality


def audit_experience_quality_for_experiences(
    experiences: list[Experience],
) -> ReviewResult:
    """Run every experience-quality check and merge the findings."""
    review_results = [
        audit_bullet_structure_for_experiences(experiences),
        audit_consistency_for_experiences(experiences),
        audit_quantification_for_experiences(experiences),
        audit_language_quality_for_experiences(experiences),
    ]
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


__all__ = [
    "audit_bullet_structure",
    "audit_bullet_structure_for_experiences",
    "audit_consistency",
    "audit_consistency_for_experiences",
    "audit_experience_quality_for_experiences",
    "audit_experience_rewrite_quality",
    "audit_language_quality",
    "audit_language_quality_for_experiences",
    "audit_quantification",
    "audit_quantification_for_experiences",
    "audit_summary_quality",
]
