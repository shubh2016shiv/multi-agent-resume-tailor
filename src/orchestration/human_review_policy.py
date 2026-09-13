"""When a pipeline run must escalate to human review.

This is the single, documented home for the human-review escalation policy. Before
this module the policy lived as inline boolean expressions scattered across nodes
(quality + ats_patch); a developer asking "when does this system hand off to a human?"
had to find all of them, and one was missed when ats_patch was added later, letting a
successful ATS recovery silently clear an escalation raised for an unrelated reason. Now
the whole policy is stated here, and the nodes call these named predicates.

THE POLICY -- a run escalates to human review (terminal: not rendered, flagged on state
as human_review_required) in exactly three situations:

  1. UNVERIFIABLE -- the resume could not be rendered to .tex, so the ATS check returned
     INCONCLUSIVE. Nothing could be inspected and nothing can be patched (the problem is
     in rendering, not in the data). Discovered at the QA stage.

  2. UNRECOVERABLE -- the ATS check FAILed and the deterministic section restore could
     not fix it (the essential section was empty upstream too), or the re-grade itself
     came back not-PASS. No automated recovery remains. Discovered at the patch stage.

  3. RELEVANCE UNVERIFIABLE -- the target job supplied no structured requirements or ATS
     keywords to score alignment against, so relevance.is_conclusive is False. There is
     nothing to judge quality against, so no score can be trusted. Discovered at the QA
     stage; ats_patch does not re-check it (it re-grades only the ATS dimension), so it
     must survive that node unchanged -- see patch_ats_assembly's OR with the incoming
     state["human_review_required"].

A plain ATS FAIL at the QA stage is NOT escalation: it is recoverable and routes to the
patch node first. Only after recovery is exhausted does a non-PASS become human review.

The same module owns the run's caller-facing verdict: derive_run_disposition() folds
the escalation flag, the quality gate, and any candidate questions into the single
RunDisposition the runner puts on OrchestrationResult.
"""

from src.data_models.evaluation import (
    AtsCheckStatus,
    JobAlignmentEvaluation,
    RenderedStructureEvaluation,
)
from src.data_models.orchestration import RunDisposition


def is_ats_unverifiable(outcome: RenderedStructureEvaluation) -> bool:
    """Return True when the ATS artifact could not be built/inspected (INCONCLUSIVE).

    Used at the QA stage: an unverifiable outcome escalates straight to human review,
    because there is nothing to patch -- the failure is in rendering, not the data.
    """
    return outcome.status is AtsCheckStatus.INCONCLUSIVE


def is_ats_unrecoverable(outcome: RenderedStructureEvaluation) -> bool:
    """Return True when an ATS outcome is anything other than PASS after recovery ran.

    Used at the patch stage, AFTER the deterministic section restore: any remaining
    non-PASS status (FAIL with the section empty upstream too, or an INCONCLUSIVE
    re-grade) means no automated recovery is left, so the run escalates to human review.
    """
    return outcome.status is not AtsCheckStatus.PASS


def is_relevance_unverifiable(relevance: JobAlignmentEvaluation) -> bool:
    """Return True when the target job gave nothing to score alignment against.

    Used at the QA stage: an inconclusive relevance judgment escalates straight to
    human review, because no downstream recovery step re-grades relevance -- it must
    hold from here to the run's final disposition unchanged.
    """
    return not relevance.is_conclusive


def derive_run_disposition(
    human_review_required: bool,
    quality_gate_passed: bool,
    has_candidate_questions: bool,
) -> RunDisposition:
    """Fold a finished run's flags into the one action the caller should take next.

    Precedence: the most blocking condition wins. A human-review escalation trumps
    everything; a failed quality gate trumps candidate questions; candidate
    questions trump a clean render (the resume rendered, but answering the sheet
    and re-running makes it better).
    """
    if human_review_required:
        return RunDisposition.NEEDS_HUMAN_REVIEW
    if not quality_gate_passed:
        return RunDisposition.QUALITY_GATE_FAILED
    if has_candidate_questions:
        return RunDisposition.NEEDS_CANDIDATE_INPUT
    return RunDisposition.RENDERED
