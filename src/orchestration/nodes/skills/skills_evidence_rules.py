"""Evidence and preservation rules for optimized resume skills."""

from src.data_models.resume import OptimizedSkillsSection, Resume, Skill
from src.tools.contracts import Confidence, ReviewComment, ReviewResult, Severity
from src.tools.engines.truthfulness.skills_evidence import validate_skills_evidence

_SERIOUS_SEVERITIES = {Severity.BLOCKER, Severity.MAJOR}


def audit_skills_section(
    original_resume: Resume,
    optimized_skills: OptimizedSkillsSection,
) -> ReviewResult:
    """Audit optimized skills against evidence in the original resume."""
    audit_resume = original_resume.model_copy(update={"skills": optimized_skills.optimized_skills})
    return validate_skills_evidence(audit_resume)


def is_confident_unsupported(comment: ReviewComment) -> bool:
    """Return whether an unsupported-skill finding is safe to act upon."""
    return comment.severity in _SERIOUS_SEVERITIES and comment.confidence == Confidence.HIGH


def skills_audit_needs_rewrite(audit: ReviewResult) -> bool:
    """Return whether the audit warrants one corrective rewrite."""
    return any(is_confident_unsupported(comment) for comment in audit.comments)


def flagged_skill_names(audit: ReviewResult) -> list[str]:
    """Return names of skills confidently identified as unsupported."""
    return [
        comment.quoted_text or comment.message
        for comment in audit.comments
        if is_confident_unsupported(comment)
    ]


def preserve_original_skills(
    optimized: OptimizedSkillsSection,
    original_resume: Resume,
) -> OptimizedSkillsSection:
    """Restore original skills dropped by optimization and reconcile removals."""
    dropped = _find_dropped_skills(optimized, original_resume)
    if not dropped:
        return optimized

    reinstated_names = {skill.skill_name.casefold() for skill in dropped}
    surviving_removals = [
        name for name in optimized.removed_skills if name.casefold() not in reinstated_names
    ]
    return optimized.model_copy(
        update={
            "optimized_skills": optimized.optimized_skills + dropped,
            "removed_skills": surviving_removals,
        }
    )


def _find_dropped_skills(
    optimized: OptimizedSkillsSection,
    original_resume: Resume,
) -> list[Skill]:
    """Return original skills absent from the optimized section."""
    optimized_names = {skill.skill_name.casefold() for skill in optimized.optimized_skills}
    return [
        skill
        for skill in original_resume.skills
        if skill.skill_name.casefold() not in optimized_names
    ]


__all__ = [
    "audit_skills_section",
    "flagged_skill_names",
    "is_confident_unsupported",
    "preserve_original_skills",
    "skills_audit_needs_rewrite",
]
