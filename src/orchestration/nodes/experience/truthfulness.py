"""The deterministic floor a rewrite must clear before it may ship. No LLM here.

Two checks, both mechanical and both about the rewrite being the same claims as the
source, better worded:

    bullet parity   -- same number of bullets, same ids, same order as the source
    claim inflation -- no figure that the source role does not state

A proposal that fails either one cannot ship, whatever an LLM review thinks of its
prose. Non-numeric invention (a tool or team the candidate never had) is governed by
the writer's prompt and temperature-0 decoding instead: an LLM drift check on this
path penalised the context-mining that makes bullets specific, so it was removed.
"""

from src.agents.professional_experience.models import ExperienceRewriteProposal
from src.data_models.resume import Experience, Resume
from src.hitl.professional_experience.models import build_experience_bullet_id
from src.orchestration.nodes.experience.findings import findings_from_comments
from src.tools.engines.truthfulness import detect_claim_inflation


def collect_truthfulness_findings(
    source_role_resume: Resume,
    rewrite_proposal: ExperienceRewriteProposal,
    rewritten_role: Experience,
    original_experience: Experience,
) -> list[str]:
    """Return truthfulness findings for one rewritten role (empty list = safe to ship).

    Two deterministic checks, no LLM -- this is the non-negotiable floor a rewrite
    must clear to ship:
    1. Bullet count parity -- the 1:1 rewrite contract, a closed countable fact.
    2. detect_claim_inflation -- any figure present in the rewrite but absent from
       the source role. Numbers are the highest-risk, most verifiable fabrication.

    Non-numeric invention (a fake tool, team, or scale) is governed by the writer's
    prompt and temperature-0 decoding, not an LLM guard here: an LLM drift check on
    this path penalised the very context-mining that makes bullets specific, so it
    was removed for a net gain in quality without loosening the number floor.
    """
    source_count = len(original_experience.achievements)
    rewritten_count = len(rewritten_role.achievements)
    if rewritten_count != source_count:
        return [
            f"Bullet count changed: source role has {source_count} bullet(s), "
            f"rewrite has {rewritten_count}. Rewrite each source bullet one-for-one."
        ]

    expected_bullet_ids = [
        build_experience_bullet_id(original_experience, bullet_index)
        for bullet_index, _ in enumerate(original_experience.achievements)
    ]
    returned_bullet_ids = [
        bullet_rewrite.bullet_id for bullet_rewrite in rewrite_proposal.rewritten_bullets
    ]
    if returned_bullet_ids != expected_bullet_ids:
        return [
            "Bullet IDs changed or were returned out of source order. "
            "Copy each bullet_id exactly from the matching source bullet record."
        ]

    rewritten_role_resume = source_role_resume.model_copy(
        update={"work_experience": [rewritten_role]}
    )
    inflation = detect_claim_inflation(source_role_resume, rewritten_role_resume)
    return findings_from_comments(inflation.comments)


def role_with_rewritten_bullets(
    rewrite_proposal: ExperienceRewriteProposal,
    original_experience: Experience,
) -> Experience:
    """Keep only the rewritten bullet text; every other field comes from the source.

    This is the containment boundary: metadata, description, and skills_used can
    never be changed by the LLM, whatever the rewrite proposal contains.
    """
    rewritten_achievements = [
        bullet_rewrite.rewritten_bullet for bullet_rewrite in rewrite_proposal.rewritten_bullets
    ]
    return original_experience.model_copy(update={"achievements": rewritten_achievements})
