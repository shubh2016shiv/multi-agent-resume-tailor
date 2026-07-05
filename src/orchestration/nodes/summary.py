"""Stage 3 of the resume pipeline: write the professional summary.

This node is the entry point and the map for the whole professional-summary
pipeline. Reading write_professional_summary below, top to bottom, shows every
step and which module performs it:

    STEP 1  confirm upstream stages populated the state   (this file)
    STEP 2  build the writer's context                    -> professional_summary_formatter.py
    STEP 3  create the summary-writer agent               -> professional_summary/agent.py
    STEP 4  run the writing task, get a typed result      -> crew_task_execution.py
    STEP 5  enforce the quality gate before handing off   -> resume_diagnostics/summary_quality.py

Reads from state: resume, job_description, alignment_strategy.
Writes to state: professional_summary.
"""

import time

from src.agents.professional_summary import create_professional_summary_agent
from src.agents.professional_summary.models import ProfessionalSummary, SummaryDraft
from src.core.logger import get_logger
from src.formatters.professional_summary_formatter import format_professional_summary_context
from src.orchestration.crew_task_execution import run_agent_task
from src.orchestration.exceptions import PipelineQualityGateError
from src.orchestration.state import ResumeEnhancementPipelineState
from src.tools.contracts import Severity
from src.tools.engines.resume_diagnostics.summary_quality import audit_summary_text

logger = get_logger(__name__)

# A finding at or above this severity blocks the summary from proceeding. The audit
# rubric assigns MAJOR only to the writer task's own hard constraints (banned phrase,
# banned "[title] with [x] years" opener, first-person, out-of-range length), so the
# gate blocks exactly those and lets softer style notes (MINOR) pass.
BLOCKING_SEVERITIES = {Severity.MAJOR, Severity.BLOCKER}


def write_professional_summary(state: ResumeEnhancementPipelineState) -> dict:
    """Generate a professional summary tailored to the job description.

    Raises: ValueError if the recommended draft fails the quality gate (STEP 5) --
            a hard-constraint violation must not reach resume assembly.
    """
    start_time = time.monotonic()
    logger.info(
        "pipeline_stage_started",
        stage="write_professional_summary",
        run_id=state["run_id"],
    )

    ####################################################
    # STEP 1: CONFIRM UPSTREAM STAGES POPULATED STATE#
    ####################################################
    assert state["resume"] is not None, "resume must be set before summary writing"
    assert state["job_description"] is not None, "job_description must be set before summary writing"
    assert state["alignment_strategy"] is not None, "alignment_strategy must be set before summary writing"

    ####################################################
    # STEP 2: BUILD THE CONTEXT THE WRITER READS#
    ####################################################
    context = format_professional_summary_context(
        resume=state["resume"],
        job_description=state["job_description"],
        strategy=state["alignment_strategy"],
        format_type="toon",
    )

    ####################################################
    # STEP 3: CREATE THE SUMMARY-WRITER AGENT#
    ####################################################
    agent = create_professional_summary_agent()

    ####################################################
    # STEP 4: RUN THE WRITING TASK, GET A TYPED RESULT#
    ####################################################
    professional_summary = run_agent_task(
        agent=agent,
        task_name="write_professional_summary_task",
        context=context,
        output_model=ProfessionalSummary,
        run_id=state["run_id"],
    )

    ####################################################
    # STEP 5: ENFORCE THE QUALITY GATE BEFORE HANDOFF#
    ####################################################
    enforce_summary_quality_gate(professional_summary)

    duration_ms = round((time.monotonic() - start_time) * 1000)
    logger.info(
        "pipeline_stage_completed",
        stage="write_professional_summary",
        run_id=state["run_id"],
        duration_ms=duration_ms,
    )
    return {"professional_summary": professional_summary}


def select_recommended_draft(summary: ProfessionalSummary) -> SummaryDraft:
    """Return the draft the agent recommended, falling back to the first draft.

    The fallback matters because the next stage (ats_optimization_formatter.
    choose_summary_text) uses the same rule: if the recommended name does not match
    any draft, the first draft is what actually ships, so that is what we audit.
    """
    ####################################################
    # STEP 1: FIND THE DRAFT MATCHING THE RECOMMENDATION#
    ####################################################
    for draft in summary.drafts:
        if draft.version_name == summary.recommended_version:
            return draft

    ####################################################
    # STEP 2: NO MATCH -> THE FIRST DRAFT IS WHAT SHIPS#
    ####################################################
    return summary.drafts[0]


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
    ####################################################
    # STEP 1: AUDIT THE DRAFT THAT WILL ACTUALLY SHIP#
    ####################################################
    draft = select_recommended_draft(summary)
    review = audit_summary_text(draft.content)

    ####################################################
    # STEP 2: KEEP ONLY THE BLOCKING (MAJOR+) FINDINGS#
    ####################################################
    blocking_findings = [
        comment for comment in review.comments if comment.severity in BLOCKING_SEVERITIES
    ]

    ####################################################
    # STEP 3: FAIL THE RUN IF ANYTHING BLOCKS#
    ####################################################
    if blocking_findings:
        raise PipelineQualityGateError(
            stage=f"Professional summary (draft '{draft.version_name}')",
            findings=[comment.message for comment in blocking_findings],
            user_action=SUMMARY_GATE_USER_ACTION,
        )
