"""Deterministic quality gate for generated professional summaries."""

from src.agents.professional_summary.models import ProfessionalSummary
from src.formatters.ats_optimization_formatter import choose_summary_draft
from src.orchestration.exceptions import PipelineQualityGateError
from src.tools.contracts import Severity
from src.tools.engines.resume_diagnostics.summary_quality import audit_summary_text

BLOCKING_SEVERITIES = {Severity.MAJOR, Severity.BLOCKER}
SUMMARY_GATE_USER_ACTION = (
    "The summary is generated from your work-experience achievements, not from your "
    "resume's own summary text -- so this failure means those achievements gave it "
    "too little concrete material. Add specific, measurable outcomes to your "
    "work-experience bullets (numbers, scale, named systems, results) and run the "
    "pipeline again. If the findings above look like style violations rather than "
    "thin evidence, simply re-running once may resolve it."
)


def enforce_summary_quality_gate(summary: ProfessionalSummary) -> None:
    """Raise a quality-gate error when the selected draft has blocking findings."""
    draft = choose_summary_draft(summary)
    review = audit_summary_text(draft.content)
    blocking_findings = [
        comment for comment in review.comments if comment.severity in BLOCKING_SEVERITIES
    ]
    if blocking_findings:
        raise PipelineQualityGateError(
            stage=f"Professional summary (draft '{draft.version_name}')",
            findings=[comment.message for comment in blocking_findings],
            user_action=SUMMARY_GATE_USER_ACTION,
        )


__all__ = ["enforce_summary_quality_gate"]
