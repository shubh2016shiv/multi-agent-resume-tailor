"""Stage 5b: deterministic ATS section recovery (no LLM).

A rendered-ATS FAIL means the assembler produced a final_resume with an EMPTY essential
section (experience/education/skills), so the renderer dropped that section's header. The
section's content already exists upstream as canonical typed state, so we restore it in
code -- we do NOT ask the LLM to re-assemble (that would reintroduce the retry-until-timeout
loop Phase 1 killed). One deterministic pass, then re-grade: it either passes or a human
must look. There is no cycle and no attempt counter: a single typed-state restore is
exhaustive; if it does not fix the FAIL, a second identical pass cannot either.

# TODO: investigate WHY the assembler drops essential sections (Phase 2.5). Same lesson as
#       the gap-analysis fix: if the ATS optimizer agent cannot see a section in its lossy
#       context it may omit it. Proposed: pre-compute the sections into typed context so the
#       agent never has to reconstruct them. Deferred: this patch is the safety net; the
#       investigation decides whether it is permanent infrastructure or a temporary guard.
"""

from typing import Any

from src.agents.professional_experience.models import OptimizedExperienceSection
from src.agents.quality_assessment.engines import apply_quality_gate
from src.data_models.evaluation import AtsCheckStatus, ATSMetrics, AtsRenderedOutcome, QualityReport
from src.data_models.resume import OptimizedSkillsSection, Resume
from src.evaluation_rubrics import blend_overall, grade_ats
from src.orchestration.human_review_policy import is_ats_unrecoverable
from src.orchestration.state import ResumeEnhancementPipelineState


def patch_ats_assembly(state: ResumeEnhancementPipelineState) -> dict:
    """Restore essential sections the assembler dropped, then re-grade ATS. No LLM.

    Precondition: entered only when ats_rendered_outcome.status is FAIL (the router
    guarantees this). Refills each empty essential section from canonical upstream typed
    state, re-grades the rebuilt resume, and re-applies the gate.

    Reads: optimized_resume, optimized_experience, optimized_skills, resume, qa_report.
    Writes: optimized_resume (patched final_resume -- overwritten, the pre-patch assembler
            output is not retained), qa_report (re-graded ATS + re-gated), ats_rendered_outcome
            (the re-grade), human_review_required (True if the restore could not fix it).
    """
    assert state["optimized_resume"] is not None, "optimized_resume must be set before ATS patch"
    assert state["optimized_experience"] is not None, "optimized_experience must be set before ATS patch"
    assert state["optimized_skills"] is not None, "optimized_skills must be set before ATS patch"
    assert state["resume"] is not None, "resume must be set before ATS patch"
    assert state["qa_report"] is not None, "qa_report must be set before ATS patch"
    optimized_resume = state["optimized_resume"]
    patched_final = _restore_missing_essential_sections(
        final_resume=optimized_resume.final_resume,
        optimized_experience=state["optimized_experience"],
        optimized_skills=state["optimized_skills"],
        original_resume=state["resume"],
    )
    new_outcome = grade_ats(patched_final)
    qa_report = _regrade_ats_dimension(state["qa_report"], new_outcome)
    return {
        "optimized_resume": optimized_resume.model_copy(update={"final_resume": patched_final}),
        "qa_report": qa_report,
        "ats_rendered_outcome": new_outcome,
        # A restore that still does not PASS means recovery is exhausted (the section was
        # empty upstream too). The escalation policy lives in human_review_policy.
        "human_review_required": is_ats_unrecoverable(new_outcome),
    }


def _restore_missing_essential_sections(
    final_resume: Resume,
    optimized_experience: OptimizedExperienceSection,
    optimized_skills: OptimizedSkillsSection,
    original_resume: Resume,
) -> Resume:
    """Return a copy of final_resume with each EMPTY essential section refilled from its
    canonical upstream source. Non-empty sections, and sections empty upstream too, are
    left untouched (the latter become the genuine, unrecoverable FAIL).

    One strategy throughout: "section empty here AND present upstream -> fill from upstream."
    """
    updates: dict[str, Any] = {}
    if not final_resume.work_experience and optimized_experience.optimized_experiences:
        updates["work_experience"] = optimized_experience.optimized_experiences
    if not final_resume.skills and optimized_skills.optimized_skills:
        updates["skills"] = optimized_skills.optimized_skills
    if not final_resume.education and original_resume.education:
        updates["education"] = original_resume.education
    return final_resume.model_copy(update=updates) if updates else final_resume


def _regrade_ats_dimension(
    qa_report: QualityReport, ats_outcome: AtsRenderedOutcome
) -> QualityReport:
    """Rebuild the ATS dimension and overall score from the re-graded outcome, reusing the
    pre-patch accuracy/relevance, then re-apply the gate and hard-block on a non-PASS status.

    This repeats the small ATS-grounding tail from quality._ground_quality_dimensions. The
    duplication is deliberate: Phase 1 is verified and clean, and extracting a shared helper
    would change its contract and raise a layer question (the gate lives in the agent engines
    package). Local duplication of ~6 lines is the lower-risk choice here.
    """
    grounded_ats = ATSMetrics(
        ats_score=ats_outcome.ats_score,
        keyword_coverage=qa_report.relevance.must_have_skills_coverage,
        formatting_issues=ats_outcome.violations,
        justification=ats_outcome.detail,
    )
    overall = blend_overall(
        qa_report.accuracy.accuracy_score,
        qa_report.relevance.relevance_score,
        ats_outcome.ats_score,
    )
    regraded = qa_report.model_copy(
        update={"ats_optimization": grounded_ats, "overall_quality_score": overall}
    )
    regraded = apply_quality_gate(regraded)
    if ats_outcome.status is not AtsCheckStatus.PASS:
        regraded = regraded.model_copy(update={"passed_quality_threshold": False})
    return regraded
    # TODO: relevance is NOT re-graded after restore; a restored skills section can raise JD
    #       keyword coverage, so the reused relevance score may understate the patched resume.
    #       Proposed: re-run grade_relevance(patched_final, job) inside patch_ats_assembly.
    #       Deferred: Phase 2 honors the stated grade_ats-only re-run; revisit only if a
    #       restored resume sits just under threshold purely on stale relevance.
