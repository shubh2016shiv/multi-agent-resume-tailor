"""Stage 3 professional summary node."""

import time

from src.agents.professional_summary import create_professional_summary_agent
from src.agents.professional_summary.models import ProfessionalSummary, SummaryDraft
from src.core.logger import get_logger
from src.formatters.professional_summary_formatter import format_professional_summary_context
from src.orchestration.crew_task_execution import run_agent_task
from src.orchestration.state import ResumeEnhancementPipelineState
from src.tools.contracts import Severity
from src.tools.engines.resume_diagnostics.summary_quality import audit_summary_text

logger = get_logger(__name__)


def write_professional_summary(state: ResumeEnhancementPipelineState) -> dict:
    """Generate a professional summary tailored to the job description.

    Reads: resume, job_description, and alignment_strategy from prior stages.
    Writes: professional_summary.
    Returns: partial state with the typed ProfessionalSummary.
    Raises: ValueError if the recommended draft fails the summary quality gate
            (see _enforce_summary_quality_gate) -- a banned phrase, banned opener
            formula, or first-person pronoun is a hard constraint from
            write_professional_summary_task, not a style note, so the draft
            cannot proceed to assembly.
    """
    start_time = time.monotonic()
    logger.info(
        "pipeline_stage_started",
        stage="write_professional_summary",
        run_id=state["run_id"],
    )
    assert state["resume"] is not None, "resume must be set before summary writing"
    assert state["job_description"] is not None, (
        "job_description must be set before summary writing"
    )
    assert state["alignment_strategy"] is not None, (
        "alignment_strategy must be set before summary writing"
    )
    context = format_professional_summary_context(
        resume=state["resume"],
        job_description=state["job_description"],
        strategy=state["alignment_strategy"],
        format_type="toon",
    )
    agent = create_professional_summary_agent()
    professional_summary = run_agent_task(
        agent=agent,
        task_name="write_professional_summary_task",
        context=context,
        output_model=ProfessionalSummary,
        run_id=state["run_id"],
    )
    _enforce_summary_quality_gate(professional_summary)
    result = {"professional_summary": professional_summary}
    duration_ms = round((time.monotonic() - start_time) * 1000)
    logger.info(
        "pipeline_stage_completed",
        stage="write_professional_summary",
        run_id=state["run_id"],
        duration_ms=duration_ms,
    )
    return result


def _select_recommended_draft(summary: ProfessionalSummary) -> SummaryDraft:
    """Return the draft the agent recommended, or the first draft if the name doesn't match.

    Mirrors the fallback in ats_optimization_formatter.choose_summary_text -- duplicated
    locally rather than imported, since that function belongs to the next stage's
    formatter and this selection is small and stage-local (same reasoning ats_patch.py
    gives for its own ~6-line local duplication).
    """
    for draft in summary.drafts:
        if draft.version_name == summary.recommended_version:
            return draft
    return summary.drafts[0]


def _enforce_summary_quality_gate(summary: ProfessionalSummary) -> None:
    """Hard-block the recommended draft on any MAJOR+ summary-quality finding.

    Only MAJOR+ findings block: the audit rubric (summary_quality.md) assigns MAJOR
    solely to write_professional_summary_task's own hard constraints -- banned
    phrases and the banned "[title] with [x] years" opener -- and the mechanical
    first-person check is MAJOR by the same file. MINOR findings (tone, missing
    value proposition) are advisory and do not block.

    No retry loop: matches this pipeline's existing rejection of retry-until-pass
    loops (see ats_patch.py) -- a bad draft fails the run rather than looping the LLM.

    Raises: ValueError naming every blocking finding, so the run stops before a
            draft that violates the task's own hard constraints reaches assembly.
    """
    draft = _select_recommended_draft(summary)
    review = audit_summary_text(draft.content)
    blocking = [
        comment
        for comment in review.comments
        if comment.severity in (Severity.MAJOR, Severity.BLOCKER)
    ]
    if blocking:
        findings = "; ".join(comment.message for comment in blocking)
        raise ValueError(
            f"Professional summary draft '{draft.version_name}' failed the quality "
            f"gate: {findings}"
        )
