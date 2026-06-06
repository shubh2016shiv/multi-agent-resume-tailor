from src.data_models.resume import Experience
from src.tools.review_contract.review_models import ReviewResult

from .bullet_structure_auditor import (
    audit_bullet_structure,
    audit_bullet_structure_for_experiences,
)
from .consistency_auditor import audit_consistency, audit_consistency_for_experiences
from .language_quality_auditor import (
    audit_language_quality,
    audit_language_quality_for_experiences,
)
from .quantification_auditor import (
    audit_quantification,
    audit_quantification_for_experiences,
)
from .summary_quality_auditor import audit_summary_quality


# Stage 3 / Step 3.4: Final place where experience quality checks run
# Receives from: typed list[Experience] returned by the professional experience LLM.
# Sends to: ReviewResult used by orchestration to accept or request one rewrite.
#
# IMPORTANT:
# These four checks are not orphaned and are not delegated to the writer LLM.
# They run here, in code, after OptimizedExperienceSection exists:
# - audit_bullet_structure_for_experiences
# - audit_consistency_for_experiences
# - audit_quantification_for_experiences
# - audit_language_quality_for_experiences
#
# Why here:
# The writer LLM receives TOON for reading, but these checks need typed Experience
# objects. Running them here avoids forcing the LLM to convert TOON/draft text
# into JSON tool arguments.
def audit_experience_quality_for_experiences(
    experiences: list[Experience],
) -> ReviewResult:
    """Run all professional-experience quality checks on typed entries.

    Expects one or more Experience objects returned by the writer agent.
    Returns one merged ReviewResult for orchestration decisions.
    """
    results = [
        audit_bullet_structure_for_experiences(experiences),
        audit_consistency_for_experiences(experiences),
        audit_quantification_for_experiences(experiences),
        audit_language_quality_for_experiences(experiences),
    ]
    comments = [comment for result in results for comment in result.comments]
    summary = "; ".join(result.summary for result in results if result.summary)
    score = next((result.score for result in results if result.score is not None), None)
    return ReviewResult(comments=comments, summary=summary, score=score)


__all__ = [
    "audit_bullet_structure",
    "audit_bullet_structure_for_experiences",
    "audit_consistency",
    "audit_consistency_for_experiences",
    "audit_experience_quality_for_experiences",
    "audit_language_quality",
    "audit_language_quality_for_experiences",
    "audit_quantification",
    "audit_quantification_for_experiences",
    "audit_summary_quality",
]
