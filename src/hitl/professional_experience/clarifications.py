"""Candidate clarification questions for professional-experience bullets.

This module is the single home for "how does a candidate question get made".
One semantic LLM review decides, per shipped bullet, whether a candidate-owned
fact (artifact, result, user/scope, or scale) is still missing -- and phrases
the final candidate-facing question in the same pass. Pure code then joins the
findings back to the rewrite records by bullet_id and builds the sheet entries.

Flow:
  rewrite ships (src/orchestration/nodes/experience.py)
    -> audit_experience_candidate_fact_gaps()   ONE LLM call; rubric lives at
       src/config/tool_prompts/hitl/experience_candidate_fact_gap.md
    -> clarifications_from_findings()           deterministic join, no LLM
    -> list[ExperienceBulletClarification]      -> clarification sheet + pause
"""

from src.agents.professional_experience.models import ExperienceBulletRewrite
from src.core.logger import get_logger
from src.core.prompt_catalog import load_tool_prompt
from src.data_models.resume import Experience
from src.hitl.professional_experience.models import (
    ExperienceBulletClarification,
    ExperienceBulletFactGapFinding,
    ExperienceBulletFactGapReview,
)
from src.tools.llm_gateway import request_structured_output

logger = get_logger(__name__)

EXPERIENCE_CANDIDATE_FACT_GAP_RUBRIC = load_tool_prompt(
    "hitl/experience_candidate_fact_gap.md"
)


def build_bullet_clarifications(
    experience: Experience,
    shipped_bullets: list[str],
    rewritten_bullets: list[ExperienceBulletRewrite],
    run_id: str = "unknown",
) -> list[ExperienceBulletClarification]:
    """Return candidate questions for bullets that still need candidate-owned facts.

    Expects the role's shipped bullets and the rewrite records that produced them.
    When candidate answers from a previous round exist, pass the evidence-augmented
    experience so already-answered facts count as role evidence and are not re-asked.
    """
    if not shipped_bullets or not rewritten_bullets:
        return []
    fact_gap_review = audit_experience_candidate_fact_gaps(
        source_experience=experience,
        shipped_bullets=shipped_bullets,
        rewritten_bullets=rewritten_bullets,
    )
    clarifications = clarifications_from_findings(
        experience=experience,
        shipped_bullets=shipped_bullets,
        rewritten_bullets=rewritten_bullets,
        findings=fact_gap_review.findings,
    )
    logger.info(
        "experience_clarifications_built",
        run_id=run_id,
        company=experience.company_name,
        findings=len(fact_gap_review.findings),
        questions=len(clarifications),
    )
    return clarifications


def audit_experience_candidate_fact_gaps(
    source_experience: Experience,
    shipped_bullets: list[str],
    rewritten_bullets: list[ExperienceBulletRewrite],
) -> ExperienceBulletFactGapReview:
    """Decide which shipped bullets need candidate facts and phrase each question.

    One structured-output call. The rubric owns both decisions -- whether a
    candidate-owned fact is missing, and the exact question that asks for it --
    because they are one judgment: what to ask follows directly from what is missing.
    """
    review_input = _build_fact_gap_review_input(
        source_experience,
        shipped_bullets,
        rewritten_bullets,
    )
    return request_structured_output(
        ExperienceBulletFactGapReview,
        EXPERIENCE_CANDIDATE_FACT_GAP_RUBRIC,
        review_input,
        temperature=0.0,
    )


def clarifications_from_findings(
    experience: Experience,
    shipped_bullets: list[str],
    rewritten_bullets: list[ExperienceBulletRewrite],
    findings: list[ExperienceBulletFactGapFinding],
) -> list[ExperienceBulletClarification]:
    """Join fact-gap findings back to their rewrite records and build sheet entries.

    Pure code, joined by bullet_id. A finding that names an unknown bullet_id or
    arrives incomplete (no category or no question despite requiring input) is
    dropped with a warning rather than shipped as a broken sheet entry.
    """
    rewrite_by_bullet_id = {
        bullet_rewrite.bullet_id: bullet_rewrite
        for bullet_rewrite in rewritten_bullets
    }
    clarifications: list[ExperienceBulletClarification] = []
    for finding in findings:
        if not finding.requires_candidate_input:
            continue
        bullet_rewrite = rewrite_by_bullet_id.get(finding.bullet_id)
        question = (finding.question or "").strip()
        if bullet_rewrite is None or finding.gap_category is None or not question:
            logger.warning(
                "experience_clarification_finding_dropped",
                bullet_id=finding.bullet_id,
                known_bullet=bullet_rewrite is not None,
                has_category=finding.gap_category is not None,
                has_question=bool(question),
            )
            continue
        clarifications.append(
            ExperienceBulletClarification(
                company_name=experience.company_name,
                bullet_id=finding.bullet_id,
                job_title=experience.job_title,
                start_date=experience.start_date.isoformat(),
                bullet=_shipped_bullet_text(bullet_rewrite, shipped_bullets),
                why_flagged=finding.why_candidate_input_is_needed,
                gap_category=finding.gap_category,
                missing_fact_summary=finding.missing_fact_summary,
                question=question,
            )
        )
    return clarifications


def _shipped_bullet_text(
    bullet_rewrite: ExperienceBulletRewrite,
    shipped_bullets: list[str],
) -> str:
    """The text the candidate sees: the rewrite when it shipped, else the source bullet.

    The source-preserved fallback ships the original bullets, so the rewrite record's
    rewritten text is absent from shipped_bullets exactly then.
    """
    if bullet_rewrite.rewritten_bullet in shipped_bullets:
        return bullet_rewrite.rewritten_bullet
    return bullet_rewrite.source_bullet


def _build_fact_gap_review_input(
    source_experience: Experience,
    shipped_bullets: list[str],
    rewritten_bullets: list[ExperienceBulletRewrite],
) -> str:
    """Render role evidence and shipped bullets into the semantic-review input.

    Iterates the rewrite records -- the owners of bullet identity -- so every block
    carries its true bullet_id and the exact text that shipped for that bullet.
    """
    bullet_blocks = [
        _render_bullet_block(block_number, bullet_rewrite, shipped_bullets)
        for block_number, bullet_rewrite in enumerate(rewritten_bullets, start=1)
    ]
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
            "SHIPPED BULLETS TO REVIEW",
            "\n\n".join(bullet_blocks),
        ]
    )


def _render_bullet_block(
    block_number: int,
    bullet_rewrite: ExperienceBulletRewrite,
    shipped_bullets: list[str],
) -> str:
    """Render one bullet with its identity, shipped text, and supporting evidence."""
    supporting_evidence = (
        "\n".join(
            f"    - {evidence_item}"
            for evidence_item in bullet_rewrite.supporting_role_evidence
        )
        or "    - (none provided)"
    )
    writer_hint = bullet_rewrite.clarifying_question or "(none)"
    return "\n".join(
        [
            f"BULLET {block_number}",
            f"  BULLET_ID: {bullet_rewrite.bullet_id}",
            f"  SOURCE_BULLET: {bullet_rewrite.source_bullet}",
            f"  CURRENT_SHIPPED_BULLET: {_shipped_bullet_text(bullet_rewrite, shipped_bullets)}",
            f"  DECLARED_OWNERSHIP_LEVEL: {bullet_rewrite.ownership_level}",
            "  SUPPORTING_ROLE_EVIDENCE:",
            supporting_evidence,
            f"  WRITER_QUESTION_HINT: {writer_hint}",
        ]
    )
