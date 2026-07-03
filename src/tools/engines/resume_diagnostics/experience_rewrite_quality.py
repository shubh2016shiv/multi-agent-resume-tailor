"""
Semantic review for grounded experience rewrites.

This judgment engine reviews one role's rewritten bullets against the same role's
source bullets, description, and skills_used. It checks the rewrite behavior the
experience node actually cares about: supported specificity, ownership
preservation, plain recruiter-readable tone, and whether the rewritten bullets
would make a serious recruiter consider the candidate for interview.
"""

from src.agents.professional_experience.models import ExperienceBulletRewrite
from src.core.prompt_catalog import load_tool_prompt
from src.data_models.resume import Experience
from src.tools.contracts import ReviewResult
from src.tools.llm_gateway import request_review

ENGINE_ID = "experience_rewrite_quality_auditor"

EXPERIENCE_REWRITE_QUALITY_RUBRIC = load_tool_prompt(
    "resume_diagnostics/experience_rewrite_quality.md"
)


def audit_experience_rewrite_quality(
    source_experience: Experience,
    rewritten_bullets: list[ExperienceBulletRewrite],
) -> ReviewResult:
    """Review one role's rewritten bullets for supported specificity and tone.

    Args:
        source_experience: The original experience entry being rewritten.
        rewritten_bullets: The rewrite proposal records for that same role.

    Returns:
        A ReviewResult with role-specific rewrite findings and an interview-worthiness
        score. An empty result means there were no rewritten bullets to review.
    """
    if not rewritten_bullets:
        return ReviewResult(comments=[], summary="No rewritten bullets to review")

    review_input = _build_experience_rewrite_review_input(
        source_experience, rewritten_bullets
    )
    return request_review(
        ENGINE_ID,
        EXPERIENCE_REWRITE_QUALITY_RUBRIC,
        review_input,
    )


def _build_experience_rewrite_review_input(
    source_experience: Experience,
    rewritten_bullets: list[ExperienceBulletRewrite],
) -> str:
    """Render one role's source evidence and rewrites into a compact review payload."""
    bullet_blocks = []
    for bullet_number, rewritten_bullet in enumerate(rewritten_bullets, start=1):
        supporting_evidence = (
            "\n".join(
                f"    - {evidence_item}"
                for evidence_item in rewritten_bullet.supporting_role_evidence
            )
            or "    - (none provided)"
        )
        bullet_blocks.append(
            "\n".join(
                [
                    f"BULLET {bullet_number}",
                    f"  SOURCE_BULLET: {rewritten_bullet.source_bullet}",
                    f"  REWRITTEN_BULLET: {rewritten_bullet.rewritten_bullet}",
                    f"  DECLARED_OWNERSHIP_LEVEL: {rewritten_bullet.ownership_level}",
                    "  SUPPORTING_ROLE_EVIDENCE:",
                    supporting_evidence,
                ]
            )
        )

    role_skills = (
        ", ".join(source_experience.skills_used)
        if source_experience.skills_used
        else "(none listed)"
    )
    return "\n".join(
        [
            "ROLE CONTEXT",
            f"JOB_TITLE: {source_experience.job_title}",
            f"COMPANY_NAME: {source_experience.company_name}",
            f"ROLE_DESCRIPTION: {source_experience.description}",
            f"ROLE_SKILLS_USED: {role_skills}",
            "",
            "REWRITE PROPOSAL",
            "\n\n".join(bullet_blocks),
        ]
    )
