"""Stage 3 (parallel with skills and experience): write the professional summary.

What write_professional_summary does, in order, and which module does the work:

    build the writer's context     -> professional_summary_formatter.py
    run the summary-writer agent   -> professional_summary/agent.py,
                                      via crew_task_execution.py
    enforce the quality gate       -> resume_diagnostics/summary_quality.py
                                      (enforce_summary_quality_gate, this file)

The agent returns several drafts and names one as recommended. The gate audits the
draft that will actually ship -- chosen by ats_optimization_formatter.choose_summary_draft,
the same function the assembler uses -- so the gate and the resume can never disagree
about which draft was judged.

Reads from state: resume, job_description, alignment_strategy.
Writes to state: professional_summary.
"""

from src.agents.professional_summary import create_professional_summary_agent
from src.agents.professional_summary.models import ProfessionalSummary
from src.core.logger import get_logger
from src.formatters.ats_optimization_formatter import choose_summary_draft
from src.formatters.professional_summary_formatter import format_professional_summary_context
from src.orchestration.crew_task_execution import run_agent_task
from src.orchestration.exceptions import PipelineQualityGateError
from src.orchestration.nodes._stage import pipeline_stage
from src.orchestration.state import ResumeEnhancementPipelineState, require
from src.tools.contracts import Severity
from src.tools.engines.resume_diagnostics.summary_quality import audit_summary_text

logger = get_logger(__name__)

# A finding at or above this severity blocks the summary from proceeding. The audit
# rubric assigns MAJOR only to the writer task's own hard constraints (banned phrase,
# banned "[title] with [x] years" opener, first-person, out-of-range length), so the
# gate blocks exactly those and lets softer style notes (MINOR) pass.
BLOCKING_SEVERITIES = {Severity.MAJOR, Severity.BLOCKER}


@pipeline_stage("write_professional_summary")
def write_professional_summary(state: ResumeEnhancementPipelineState) -> dict:
    """Generate a professional summary tailored to the job description.

    Raises: PipelineQualityGateError if the recommended draft fails the quality gate
            -- a hard-constraint violation must not reach resume assembly.
    """
    resume = require(state["resume"], "resume")
    job_description = require(state["job_description"], "job_description")
    alignment_strategy = require(state["alignment_strategy"], "alignment_strategy")

    context = format_professional_summary_context(
        resume=resume,
        job_description=job_description,
        strategy=alignment_strategy,
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

    enforce_summary_quality_gate(professional_summary)

    return {"professional_summary": professional_summary}


# What the user can do when the summary gate blocks the run. The advice points at
# the experience section on purpose: the writer never reads the resume's own
# summary text (the formatter drops it -- it over-anchors the writer) and builds
# the summary ONLY from the work-experience achievements. When those achievements
# carry no measurable evidence, there is too little truthful material for a strong
# 80-110 word summary -- and editing the resume's summary section would not help.
SUMMARY_GATE_USER_ACTION = (
    "The summary is generated from your work-experience achievements, not from your "
    "resume's own summary text -- so this failure means those achievements gave it "
    "too little concrete material. Add specific, measurable outcomes to your "
    "work-experience bullets (numbers, scale, named systems, results) and run the "
    "pipeline again. If the findings above look like style violations rather than "
    "thin evidence, simply re-running once may resolve it."
)


def enforce_summary_quality_gate(summary: ProfessionalSummary) -> None:
    """Block the run if the draft that will ship violates a hard constraint.

    There is no retry loop here: this pipeline deliberately avoids retry-until-pass
    loops (see ats_patch.py). A bad draft fails the run so a human sees it, rather
    than the pipeline silently looping the LLM.

    Raises: PipelineQualityGateError naming every blocking finding, which the CLI
            presents to the user as an actionable message instead of a traceback.
    """
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
