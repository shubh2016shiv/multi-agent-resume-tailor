"""Deterministic truthfulness rules for professional-experience rewrites."""

from src.agents.professional_experience.models import ExperienceRewriteProposal
from src.data_models.resume import Experience, Resume
from src.hitl.professional_experience.models import build_experience_bullet_id
from src.tools.contracts import ReviewComment
from src.tools.engines.truthfulness import detect_claim_inflation


def collect_truthfulness_findings(
    source_role_resume: Resume,
    rewrite_proposal: ExperienceRewriteProposal,
    rewritten_role: Experience,
    original_experience: Experience,
) -> list[str]:
    """Return mechanical findings that prevent a role rewrite from shipping."""
    source_count = len(original_experience.achievements)
    rewritten_count = len(rewritten_role.achievements)
    if rewritten_count != source_count:
        return [
            f"Bullet count changed: source role has {source_count} bullet(s), "
            f"rewrite has {rewritten_count}. Rewrite each source bullet one-for-one."
        ]

    expected_ids = [
        build_experience_bullet_id(original_experience, index)
        for index, _ in enumerate(original_experience.achievements)
    ]
    returned_ids = [item.bullet_id for item in rewrite_proposal.rewritten_bullets]
    if returned_ids != expected_ids:
        return [
            "Bullet IDs changed or were returned out of source order. "
            "Copy each bullet_id exactly from the matching source bullet record."
        ]

    rewritten_resume = source_role_resume.model_copy(update={"work_experience": [rewritten_role]})
    inflation = detect_claim_inflation(source_role_resume, rewritten_resume)
    return findings_from_comments(inflation.comments)


def role_with_rewritten_bullets(
    proposal: ExperienceRewriteProposal,
    original_experience: Experience,
) -> Experience:
    """Apply only proposed bullet text while preserving all source-role metadata."""
    achievements = [item.rewritten_bullet for item in proposal.rewritten_bullets]
    return original_experience.model_copy(update={"achievements": achievements})


def select_truthful_rewrite(
    source_role_resume: Resume,
    proposal: ExperienceRewriteProposal,
    source_role: Experience,
) -> tuple[Experience, list[str]]:
    """Return the rewritten role when safe, otherwise the untouched source role."""
    rewritten_role = role_with_rewritten_bullets(proposal, source_role)
    findings = collect_truthfulness_findings(
        source_role_resume,
        proposal,
        rewritten_role,
        source_role,
    )
    return (source_role if findings else rewritten_role), findings


def findings_from_comments(comments: list[ReviewComment]) -> list[str]:
    """Flatten review comments into actionable finding text."""
    return [f"{comment.message}. {comment.advice}" for comment in comments]


__all__ = [
    "collect_truthfulness_findings",
    "findings_from_comments",
    "role_with_rewritten_bullets",
    "select_truthful_rewrite",
]
