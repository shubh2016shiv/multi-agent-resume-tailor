"""Given one role and one rewrite proposal, decide which version of its bullets ships.

Four outcomes, in the order decide_role_rewrite_outcome tries them:

    the first proposal ships as-is             (nothing wrong with it)
    the repaired proposal ships                (the first had fixable problems)
    the first proposal ships after all         (the repair came back worse)
    the original source bullets ship           (neither proposal was truthful)

_ProposalCheck is how a proposal is scored; truthfulness.py owns the checks that
cannot be waived. See this package's __init__ for how this fits the whole stage.
"""

from dataclasses import dataclass

from src.agents.professional_experience import create_professional_experience_agent
from src.agents.professional_experience.models import (
    ExperienceRewriteProposal,
    OptimizedExperienceSection,
)
from src.core.logger import get_logger
from src.data_models.resume import Experience, Resume
from src.orchestration.crew_task_execution import run_agent_task
from src.orchestration.nodes.experience.findings import findings_from_comments
from src.orchestration.nodes.experience.truthfulness import (
    collect_truthfulness_findings,
    role_with_rewritten_bullets,
)
from src.tools.contracts import ReviewComment, ReviewResult, Severity
from src.tools.engines.resume_diagnostics import audit_experience_rewrite_quality

logger = get_logger(__name__)


@dataclass(frozen=True)
class RoleRewriteDecision:
    """What one role ships: the finished section, and the proposal that produced it.

    The proposal is kept alongside the section because the fact-gap review needs the
    per-bullet rewrite records, not just the final bullet text.
    """

    selected_rewrite_proposal: ExperienceRewriteProposal
    finalized_section: OptimizedExperienceSection


@dataclass(frozen=True)
class _ProposalCheck:
    """What one rewrite proposal scored on both layers, and what that allows.

    Findings are kept in three separate lists because they carry different weight:
    truthfulness_findings blocks shipping outright, repair_required_comments
    (MAJOR/BLOCKER) must be fixed first, and follow_up_comments (MINOR/SUGGESTION)
    may ship as long as the candidate is told about them.
    """

    rewritten_role: Experience
    truthfulness_findings: list[str]
    repair_required_comments: list[ReviewComment]
    follow_up_comments: list[ReviewComment]

    @property
    def is_shippable(self) -> bool:
        """True when this proposal may ship, with or without follow-up notes."""
        return not self.truthfulness_findings and not self.repair_required_comments

    @property
    def is_flawless(self) -> bool:
        """True when this proposal may ship and there is nothing to tell the candidate."""
        return self.is_shippable and not self.follow_up_comments

    @property
    def repair_findings(self) -> list[str]:
        """Every finding the one repair attempt should fix, truth first then quality.

        Follow-up comments are included: the repair is the only chance to improve
        them, even though they would not block shipping on their own.
        """
        return self.truthfulness_findings + findings_from_comments(
            self.repair_required_comments + self.follow_up_comments
        )

    @property
    def blocking_findings(self) -> list[str]:
        """Why this proposal cannot ship, for the note on the preserved source role.

        Truth findings win when present; they are the more fundamental failure.
        """
        return self.truthfulness_findings or findings_from_comments(self.repair_required_comments)


def request_role_rewrite_proposal(
    context: str,
    run_id: str,
) -> ExperienceRewriteProposal:
    """Ask the professional experience agent to rewrite one role's bullets.

    Expects TOON context for a single role.
    Returns an ExperienceRewriteProposal validated by run_agent_task against
    the agent's raw output.
    """
    return run_agent_task(
        agent=create_professional_experience_agent(),
        task_name="optimize_experience_section_task",
        context=context,
        output_model=ExperienceRewriteProposal,
        run_id=run_id,
    )


def _check_proposal(
    rewrite_proposal: ExperienceRewriteProposal,
    source_role_resume: Resume,
    original_experience: Experience,
) -> _ProposalCheck:
    """Score one proposal: the deterministic truth floor, then the semantic review.

    Three steps:
      1. Rebuild the role from the proposal, taking only its bullet text.
      2. Run the truth floor (bullet count, bullet ids, invented numbers).
      3. Run the LLM review for substance -- but only if step 2 passed, since a
         proposal that fails the floor cannot ship however good its prose is.
    """
    rewritten_role = role_with_rewritten_bullets(
        rewrite_proposal,
        original_experience,
    )
    truthfulness_findings = collect_truthfulness_findings(
        source_role_resume,
        rewrite_proposal,
        rewritten_role,
        original_experience,
    )
    # work_experience[0] is the evidence role: the source role with the candidate's
    # clarification answers folded in. Reviewing against the bare original would
    # flag anything sourced from those answers as unsupported.
    quality_review = (
        ReviewResult(comments=[], summary="", score=None)
        if truthfulness_findings
        else audit_experience_rewrite_quality(
            source_role_resume.work_experience[0],
            rewrite_proposal.rewritten_bullets,
        )
    )
    repair_required_comments, follow_up_comments = _split_comments_by_severity(quality_review)
    return _ProposalCheck(
        rewritten_role=rewritten_role,
        truthfulness_findings=truthfulness_findings,
        repair_required_comments=repair_required_comments,
        follow_up_comments=follow_up_comments,
    )


def decide_role_rewrite_outcome(
    rewrite_proposal: ExperienceRewriteProposal,
    context: str,
    source_role_resume: Resume,
    original_experience: Experience,
    run_id: str,
) -> RoleRewriteDecision:
    """Choose the best rewrite to ship, repairing once when the first draft falls short.

    The two lines have different force. Truthfulness is non-negotiable: only a
    version that clears the truth floor may ship, and if none survives one repair
    the source bullets do. Substance is best-effort: thin bullets earn a repair, and
    whatever is still thin afterward is surfaced to the candidate, never invented away.

    Four outcomes, in the order they are tried: the first draft ships as-is; the
    repair ships; the first draft ships anyway because the repair regressed; the
    source bullets ship because neither draft was truthful.
    """
    first = _check_proposal(rewrite_proposal, source_role_resume, original_experience)
    if first.is_flawless:
        return RoleRewriteDecision(
            selected_rewrite_proposal=rewrite_proposal,
            finalized_section=_accepted_section(
                first.rewritten_role,
                "Rewrote bullets with grounded role evidence; recruiter-readable and truthful.",
            ),
        )

    logger.info(
        "experience_rewrite_repair_requested",
        run_id=run_id,
        company=original_experience.company_name,
        truthfulness_findings=len(first.truthfulness_findings),
        quality_findings=len(first.repair_required_comments) + len(first.follow_up_comments),
    )
    repair_context = _build_experience_repair_context(
        context,
        rewrite_proposal,
        first.repair_findings,
    )
    repaired_proposal = request_role_rewrite_proposal(repair_context, run_id=run_id)
    repaired = _check_proposal(repaired_proposal, source_role_resume, original_experience)

    if repaired.is_shippable:
        return RoleRewriteDecision(
            selected_rewrite_proposal=repaired_proposal,
            finalized_section=_accepted_section(
                repaired.rewritten_role,
                "Repaired once with grounded rewrite feedback; truthful."
                + _build_candidate_follow_up_note(repaired.follow_up_comments),
            ),
        )
    if first.is_shippable:
        # The first rewrite was already safe to ship; the repair made it worse, so
        # keep the first rewrite and surface any follow-up gaps it still has.
        return RoleRewriteDecision(
            selected_rewrite_proposal=rewrite_proposal,
            finalized_section=_accepted_section(
                first.rewritten_role,
                "Kept the first grounded rewrite (repair introduced new issues)."
                + _build_candidate_follow_up_note(first.follow_up_comments),
            ),
        )
    logger.warning(
        "experience_rewrite_source_fallback",
        run_id=run_id,
        company=original_experience.company_name,
        findings=repaired.blocking_findings,
    )
    return RoleRewriteDecision(
        selected_rewrite_proposal=repaired_proposal,
        finalized_section=_source_preserved_section(
            original_experience,
            repaired.blocking_findings,
        ),
    )


def _build_experience_repair_context(
    original_context: str,
    rewrite_proposal: ExperienceRewriteProposal,
    findings: list[str],
) -> str:
    """Add the previous proposal and the exact findings for the single repair call."""
    feedback = "\n".join(f"- {finding}" for finding in findings)
    return (
        f"{original_context}\n\n"
        f"PREVIOUS_EXPERIENCE_REWRITE_PROPOSAL_JSON:\n{rewrite_proposal.model_dump_json()}\n\n"
        f"EXPERIENCE_AUDIT_FEEDBACK:\n{feedback}\n\n"
        "Rewrite once more to fix every finding above. Preserve the improvements in "
        "bullets the findings did NOT flag. Fix only the flagged rewrite problems: "
        "unsupported specificity, ownership inflation, brochure tone, vague "
        "accomplishments, or JD keyword decoration. To fix unsupported specificity, "
        "remove or replace the shaky detail with supportable role evidence. To fix "
        "ownership inflation, restate the contribution at its true level in active "
        "phrasing ('Contributed to X', 'Supported Y'). To fix weak bullets, use this "
        "same role's description or skills_used only when the detail genuinely belongs "
        "to that bullet. If a flagged bullet still cannot state a concrete result or "
        "scope without inventing facts, keep it truthful and set its "
        "clarifying_question to exactly the missing piece. Keep the same bullet count "
        "and add no figure the source role does not state. Return only "
        "ExperienceRewriteProposal JSON."
    )


def _split_comments_by_severity(
    quality_review: ReviewResult,
) -> tuple[list[ReviewComment], list[ReviewComment]]:
    """Separate must-fix rewrite findings from follow-up guidance.

    MAJOR/BLOCKER findings indicate unsupported specificity or ownership inflation
    and should not ship unresolved. MINOR/SUGGESTION findings may still ship if a
    truthful rewrite remains imperfect after one repair; those are surfaced as
    follow-up guidance for the candidate.
    """
    repair_required_comments = []
    follow_up_comments = []
    for review_comment in quality_review.comments:
        if review_comment.severity in {Severity.BLOCKER, Severity.MAJOR}:
            repair_required_comments.append(review_comment)
        else:
            follow_up_comments.append(review_comment)
    return repair_required_comments, follow_up_comments


def _accepted_section(revised_role: Experience, note: str) -> OptimizedExperienceSection:
    """Wrap an accepted (truth-verified) role for the downstream merge."""
    return OptimizedExperienceSection(
        optimized_experiences=[revised_role],
        optimization_notes=note,
    )


def _source_preserved_section(
    experience: Experience, reasons: list[str]
) -> OptimizedExperienceSection:
    """Return the untouched source role after no truthful rewrite survived one repair."""
    return OptimizedExperienceSection(
        optimized_experiences=[experience],
        optimization_notes=(
            "Rewrite could not stay truthful after one repair; original bullets "
            "preserved. Findings: " + "; ".join(reasons)
        ),
    )


def _build_candidate_follow_up_note(comments: list[ReviewComment]) -> str:
    """Surface bullets that stayed modest so the candidate can add real detail.

    These are not failures -- they are honest gaps the pipeline will not paper over
    with invented facts. Empty when every shipped bullet reads as a concrete
    accomplishment.
    """
    if not comments:
        return ""
    lines = [f'  - "{comment.quoted_text}": {comment.message}' for comment in comments]
    return (
        "\nBULLETS NEEDING YOUR INPUT (kept truthful but could not be made concrete "
        "without inventing facts -- add a specific result or detail):\n" + "\n".join(lines)
    )
