"""Candidate clarification questions for professional-experience bullets.

WHAT THIS MODULE IS
-------------------
The single home for "how does a candidate question get made". It is HITL
Component 1 -- the TRIGGER: the code that decides whether the pipeline pauses.

THE ALGORITHM, IN FOUR STEPS
----------------------------
    build_bullet_clarifications()                <- the only public entry point
      |
      +-- STEP 1  cheap guard: no bullets -> return []           (this file)
      +-- STEP 2  audit_experience_candidate_fact_gaps()          (this file)
      |             one LLM call, temperature 0, returns one verdict per bullet
      +-- STEP 3  clarifications_from_findings()                  (this file)
      |             pure code: attach role identity to each verdict that has a gap
      +-- STEP 4  log one telemetry line, return the list

WHO CALLS INTO HERE
-------------------
    _run_single_experience_optimization()   [src/orchestration/nodes/experience.py]
        calls build_bullet_clarifications() once per work-experience role, right
        after that role's rewrite has been finalised.

WHAT COMES OUT
--------------
    list[ExperienceBulletClarification]  -- zero or more questions. The list then
    travels up to the graph, is written to the clarification sheet by the runner,
    and the pause node interrupts the graph if the list is non-empty.
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

# Loaded ONCE at import time from a Markdown file, not inlined as a Python string.
# The decision logic ("when is a bullet too thin?") is therefore reviewable prose
# under version control, editable by anyone who reads English.
EXPERIENCE_CANDIDATE_FACT_GAP_RUBRIC = load_tool_prompt("hitl/experience_candidate_fact_gap.md")


# =============================================================================
# build_bullet_clarifications()  --  PUBLIC ENTRY POINT of the trigger component
# -----------------------------------------------------------------------------
# CALLED BY : _run_single_experience_optimization()
#             [src/orchestration/nodes/experience.py], once per role.
# CALLS     : audit_experience_candidate_fact_gaps()  (STEP 2, the LLM call)
#             clarifications_from_findings()          (STEP 3, pure-code join)
# RETURNS   : list[ExperienceBulletClarification]. Empty == "ask nothing here".
# =============================================================================
def build_bullet_clarifications(
    experience: Experience,  # the role being reviewed
    shipped_bullets: list[str],  # the bullet text that actually shipped
    rewritten_bullets: list[ExperienceBulletRewrite],  # the rewrite records that produced them
    run_id: str = "unknown",  # for telemetry correlation only
) -> list[ExperienceBulletClarification]:
    """Return candidate questions for bullets that still need candidate-owned facts.

    Expects the role's shipped bullets and the rewrite records that produced them.
    When candidate answers from a previous round exist, pass the evidence-augmented
    experience so already-answered facts count as role evidence and are not re-asked.

    Asking for clarification is an enhancement layered on top of an already
    truthful rewrite, not a safety gate -- truthfulness is enforced deterministically
    upstream. So a failure of this review degrades to "ask nothing" rather than
    discarding a role's completed work.
    """
    # ---- STEP 1: cheap guard -------------------------------------------------
    # A source-preserved fallback role, or a role with no bullets, has nothing to
    # review. Return early before spending an LLM call.
    if not shipped_bullets or not rewritten_bullets:
        return []

    # ---- STEP 2: the semantic verdict (ONE LLM call; may raise) ------------
    # try/except is deliberate, not lazy. This whole component is an *enhancement*
    # on top of an already-truthful rewrite. If the LLM call fails (timeout, bad
    # JSON twice, budget), the correct behaviour is to ship the good rewrite and
    # ask nothing -- never to throw away a role's finished work.
    try:
        fact_gap_review = audit_experience_candidate_fact_gaps(
            source_experience=experience,
            shipped_bullets=shipped_bullets,
            rewritten_bullets=rewritten_bullets,
        )
    except Exception:
        logger.exception(
            "experience_fact_gap_review_failed",
            run_id=run_id,
            company=experience.company_name,
            shipped_bullets=len(shipped_bullets),
        )
        return []

    # ---- STEP 3: turn verdicts into routable questions (pure code, no LLM) --
    # fact_gap_review.findings is one verdict per bullet. This join keeps only the
    # ones with a non-null gap and stamps each with the role identity an answer
    # needs to find its way home after the pause.
    clarifications = clarifications_from_findings(
        experience=experience,
        shipped_bullets=shipped_bullets,
        rewritten_bullets=rewritten_bullets,
        findings=fact_gap_review.findings,
    )

    # ---- STEP 4: one telemetry line so a run's decisions are greppable -----
    # "findings" is every bullet the model looked at; "questions" is how many it
    # decided to ask about. The gap between them is how selective the trigger was.
    logger.info(
        "experience_clarifications_built",
        run_id=run_id,
        company=experience.company_name,
        findings=len(fact_gap_review.findings),
        questions=len(clarifications),
    )
    return clarifications


# =============================================================================
# audit_experience_candidate_fact_gaps()  --  STEP 2: the one decision-making call
# -----------------------------------------------------------------------------
# THIS IS THE TRIGGER. It is the only place in the entire pipeline that decides
# whether a human is asked anything. Everything downstream -- the pause, the
# sheet, the resume -- exists only because this function returned a gap.
# See README.md section 7 (Component 1).
#
# CALLED BY : build_bullet_clarifications()  (STEP 2 above)
# CALLS     : _build_fact_gap_review_input()  -> assembles the prompt text
#             request_structured_output()     -> the LLM gateway
# RETURNS   : ExperienceBulletFactGapReview -- a wrapper holding one
#             ExperienceBulletFactGapFinding per shipped bullet.
# =============================================================================
def audit_experience_candidate_fact_gaps(
    source_experience: Experience,
    shipped_bullets: list[str],
    rewritten_bullets: list[ExperienceBulletRewrite],
) -> ExperienceBulletFactGapReview:
    """Decide which shipped bullets need candidate facts and phrase each question.

    One structured-output call. temperature=0.0 because the same bullet must not
    flip between "fine" and "needs a question" across identical runs.
    """
    # ---- 2a: render the role + its bullets into one plain-text block --------
    # This is the literal text the model reads. NOT the raw resume object.
    review_input = _build_fact_gap_review_input(
        source_experience,
        shipped_bullets,
        rewritten_bullets,
    )

    # ---- 2b: the call. Four arguments in, one validated object out. ---------
    # request_structured_output(output_model, system_prompt, user_content, temperature):
    #   - forces the model's reply to match the output_model schema (retries once
    #     on malformed JSON, then raises)
    #   - returns a *validated instance*, never raw text
    return request_structured_output(
        ExperienceBulletFactGapReview,  # output_model : the schema the reply must match
        EXPERIENCE_CANDIDATE_FACT_GAP_RUBRIC,  # system_prompt: the rubric loaded at module top
        review_input,  # user_content : this role's bullets (from 2a)
        temperature=0.0,  # determinism  : identical input -> identical verdict
    )


# =============================================================================
# clarifications_from_findings()  --  STEP 3: verdict -> routable question
# -----------------------------------------------------------------------------
# Pure code. No LLM. Takes the model's per-bullet verdicts and turns the ones
# that carry a gap into ExperienceBulletClarification objects, stamped with the
# role's identity (company, title, start_date) so an answer can be routed back
# to the exact bullet after a days-long pause.
#
# CALLED BY : build_bullet_clarifications()  (STEP 3)
#             test_clarifications.py         (directly, since it needs no LLM)
# CALLS     : _shipped_bullet_text()  -> picks the right text to show the candidate
# RETURNS   : list[ExperienceBulletClarification]
# =============================================================================
def clarifications_from_findings(
    experience: Experience,
    shipped_bullets: list[str],
    rewritten_bullets: list[ExperienceBulletRewrite],
    findings: list[ExperienceBulletFactGapFinding],  # the verdicts, from STEP 2
) -> list[ExperienceBulletClarification]:
    """Attach role identity to each fact gap the review reported.

    Pure code, joined by bullet_id. Only one thing can go wrong here that the
    schema cannot prevent: a finding naming a bullet_id the rewrite never
    produced (a hallucinated id). Whether the gap itself is complete is settled
    by CandidateFactGap being non-optional -- an incomplete gap cannot exist.
    """
    # ---- 3a: index the rewrite records by bullet_id for O(1) lookup --------
    rewrite_by_bullet_id = {
        bullet_rewrite.bullet_id: bullet_rewrite for bullet_rewrite in rewritten_bullets
    }

    clarifications: list[ExperienceBulletClarification] = []

    # ---- 3b: walk every verdict --------------------------------------------
    for finding in findings:
        # 3b-i: null gap == "this bullet is fine as written". Skip it.
        # There is NO separate boolean -- the gap's presence IS the signal.
        if finding.gap is None:
            continue

        # 3b-ii: the ONLY failure the schema can't prevent -- the model echoed
        # back a bullet_id we never gave it. Drop it loudly, don't crash.
        bullet_rewrite = rewrite_by_bullet_id.get(finding.bullet_id)
        if bullet_rewrite is None:
            logger.warning(
                "experience_clarification_finding_dropped",
                reason="unknown_bullet_id",
                bullet_id=finding.bullet_id,
                company=experience.company_name,
            )
            continue

        # 3b-iii: build the persisted question.
        #   **finding.gap.model_dump()  spreads the 4 content fields (category,
        #     missing_fact_summary, why_flagged, question) -- NOT copied by hand,
        #     because ExperienceBulletClarification inherits them from CandidateFactGap.
        #   the rest is the routing identity an answer needs after the pause.
        clarifications.append(
            ExperienceBulletClarification(
                **finding.gap.model_dump(),
                bullet_id=finding.bullet_id,
                company_name=experience.company_name,
                job_title=experience.job_title,
                start_date=experience.start_date.isoformat(),
                bullet=_shipped_bullet_text(bullet_rewrite, shipped_bullets),
            )
        )

    return clarifications


# -----------------------------------------------------------------------------
# _shipped_bullet_text()  --  helper: which text does the candidate actually see?
#   CALLED BY : clarifications_from_findings() (3b-iii), _render_bullet_block()
#   WHY IT'S NOT TRIVIAL : when no truthful rewrite survived, the pipeline ships
#   the ORIGINAL bullets. In that case the rewrite record's rewritten_bullet text
#   never appears in shipped_bullets, so we must fall back to source_bullet.
# -----------------------------------------------------------------------------
def _shipped_bullet_text(
    bullet_rewrite: ExperienceBulletRewrite,
    shipped_bullets: list[str],
) -> str:
    """The text the candidate sees: the rewrite when it shipped, else the source bullet."""
    if bullet_rewrite.rewritten_bullet in shipped_bullets:
        return bullet_rewrite.rewritten_bullet
    return bullet_rewrite.source_bullet


# -----------------------------------------------------------------------------
# _build_fact_gap_review_input()  --  helper: assemble the prompt the model reads
#   CALLED BY : audit_experience_candidate_fact_gaps() (step 2a)
#   CALLS     : _render_bullet_block() once per bullet
#   RETURNS   : one multi-line string -- role context header, then one block per
#               bullet. This is user_content for the LLM call.
# -----------------------------------------------------------------------------
def _build_fact_gap_review_input(
    source_experience: Experience,
    shipped_bullets: list[str],
    rewritten_bullets: list[ExperienceBulletRewrite],
) -> str:
    """Render role evidence and shipped bullets into the semantic-review input.

    Iterates the rewrite records -- the owners of bullet identity -- so every block
    carries its true bullet_id and the exact text that shipped for that bullet.
    """
    # one text block per bullet, numbered from 1 for human readability
    bullet_blocks = [
        _render_bullet_block(block_number, bullet_rewrite, shipped_bullets)
        for block_number, bullet_rewrite in enumerate(rewritten_bullets, start=1)
    ]
    role_skills = (
        ", ".join(source_experience.skills_used)
        if source_experience.skills_used
        else "(none listed)"
    )
    # header (role-level evidence the model may draw on) + the bullet blocks
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


# -----------------------------------------------------------------------------
# _render_bullet_block()  --  helper: format ONE bullet for the prompt
#   CALLED BY : _build_fact_gap_review_input()
#   CALLS     : _shipped_bullet_text()
#   Each block gives the model: the stable id it must echo back, the original
#   text, the shipped text, the ownership level, the evidence the rewriter used,
#   and any question the rewriter itself flagged (WRITER_QUESTION_HINT).
# -----------------------------------------------------------------------------
def _render_bullet_block(
    block_number: int,
    bullet_rewrite: ExperienceBulletRewrite,
    shipped_bullets: list[str],
) -> str:
    """Render one bullet with its identity, shipped text, and supporting evidence."""
    supporting_evidence = (
        "\n".join(
            f"    - {evidence_item}" for evidence_item in bullet_rewrite.supporting_role_evidence
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
