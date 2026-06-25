"""When a pipeline run must escalate to human review.

This is the single, documented home for the human-review escalation policy. Before
this module the policy lived as two inline boolean expressions in two different nodes
(quality + ats_patch); a developer asking "when does this system hand off to a human?"
had to find both. Now the whole policy is stated here, and the nodes call these named
predicates.

THE POLICY -- a run escalates to human review (terminal: not rendered, flagged on state
as human_review_required) in exactly two situations, both about the rendered ATS check:

  1. UNVERIFIABLE -- the resume could not be rendered to .tex, so the ATS check returned
     INCONCLUSIVE. Nothing could be inspected and nothing can be patched (the problem is
     in rendering, not in the data). Discovered at the QA stage.

  2. UNRECOVERABLE -- the ATS check FAILed and the deterministic section restore could
     not fix it (the essential section was empty upstream too), or the re-grade itself
     came back not-PASS. No automated recovery remains. Discovered at the patch stage.

A plain FAIL at the QA stage is NOT escalation: it is recoverable and routes to the
patch node first. Only after recovery is exhausted does a non-PASS become human review.
"""

from src.data_models.evaluation import AtsCheckStatus, RenderedStructureEvaluation


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
