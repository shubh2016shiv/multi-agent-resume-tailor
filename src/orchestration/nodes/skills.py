"""Stage 3 (alternate) of the resume pipeline: optimize the skills section.

This node is the entry point and the map for the whole skill-optimizer
pipeline. Reading optimize_skills below, top to bottom, shows every step and
which module performs it:

    STEP 1  confirm upstream stages populated the state    (this file)
    STEP 2  build the optimizer's context                  -> skills_optimizer_formatter.py
    STEP 3  create the skill-optimizer agent                -> skill_optimizer/agent.py
    STEP 4  run the writing task, get a typed result        -> crew_task_execution.py
    STEP 5  run the code-owned evidence audit on the result -> truthfulness/skills_evidence.py
    STEP 6  if the audit confidently flags an unsupported   -> skills_optimizer_formatter.py
            skill, rewrite once with a scoped correction       (scoped context) + crew_task_execution.py
    STEP 7  re-add any original skill the agent dropped     (this file)

Reads from state: resume, job_description, alignment_strategy.
Writes to state: optimized_skills.
"""

import time

from src.agents.skill_optimizer import create_skill_optimizer_agent
from src.core.logger import get_logger
from src.data_models.resume import OptimizedSkillsSection, Resume
from src.formatters.skills_optimizer_formatter import (
    format_skills_optimizer_context,
    format_skills_rewrite_context,
)
from src.orchestration.crew_task_execution import run_agent_task
from src.orchestration.state import ResumeEnhancementPipelineState
from src.tools.contracts import Confidence, ReviewComment, ReviewResult, Severity
from src.tools.engines.truthfulness.skills_evidence import validate_skills_evidence

logger = get_logger(__name__)

_SERIOUS_SEVERITIES = {Severity.BLOCKER, Severity.MAJOR}


def optimize_skills(state: ResumeEnhancementPipelineState) -> dict:
    """Reorder, categorize, and ATS-optimize the skills section.

    Reads: resume, job_description, and alignment_strategy from prior stages.
    Writes: optimized_skills.
    Returns: partial state with the typed OptimizedSkillsSection.
    """
    start_time = time.monotonic()
    logger.info(
        "pipeline_stage_started",
        stage="optimize_skills",
        run_id=state["run_id"],
    )

    ####################################################
    # STEP 1: CONFIRM UPSTREAM STAGES POPULATED STATE#
    ####################################################
    assert state["resume"] is not None, "resume must be set before skills optimization"
    assert state["job_description"] is not None, (
        "job_description must be set before skills optimization"
    )
    assert state["alignment_strategy"] is not None, (
        "alignment_strategy must be set before skills optimization"
    )
    resume = state["resume"]

    ####################################################
    # STEP 2: BUILD THE CONTEXT THE OPTIMIZER READS#
    ####################################################
    context = format_skills_optimizer_context(
        resume=resume,
        job_description=state["job_description"],
        strategy=state["alignment_strategy"],
        format_type="toon",
    )

    ####################################################
    # STEP 3 & 4: CREATE THE AGENT, RUN THE TASK, GET A TYPED RESULT#
    ####################################################
    optimized_skills = write_skills_section(context, run_id=state["run_id"])

    ####################################################
    # STEP 5: RUN THE CODE-OWNED EVIDENCE AUDIT ON THE RESULT#
    ####################################################
    audit_result = audit_skills_section(resume, optimized_skills)
    needs_rewrite = skills_audit_needs_rewrite(audit_result)
    logger.info(
        "skills_audit_completed",
        findings_count=len(audit_result.comments),
        needs_rewrite=needs_rewrite,
    )

    ####################################################
    # STEP 6: ONE SCOPED REWRITE IF THE AUDIT CONFIDENTLY FLAGGED A SKILL#
    ####################################################
    if needs_rewrite:
        logger.info("skills_rewrite_triggered", findings_count=len(audit_result.comments))
        rewrite_context = build_skills_rewrite_context(optimized_skills, audit_result)
        optimized_skills = write_skills_section(rewrite_context, run_id=state["run_id"])

    ####################################################
    # STEP 7: RE-ADD ANY ORIGINAL SKILL THE AGENT DROPPED#
    ####################################################
    # The candidate's listed skills are facts, and deleting one only loses an ATS keyword
    # match. The LLM is unreliable at preserving a long list verbatim, so completeness is
    # guaranteed here in code, not left to the agent.
    optimized_skills = preserve_original_skills(optimized_skills, resume)

    duration_ms = round((time.monotonic() - start_time) * 1000)
    logger.info(
        "pipeline_stage_completed",
        stage="optimize_skills",
        run_id=state["run_id"],
        duration_ms=duration_ms,
    )
    return {"optimized_skills": optimized_skills}


def write_skills_section(context: str, run_id: str = "unknown") -> OptimizedSkillsSection:
    """Ask the skill optimizer agent to produce a skills section.

    Serves node STEP 3 & 4. Expects TOON context with resume skills, job
    requirements, and strategy. Returns an OptimizedSkillsSection validated by
    CrewAI output_pydantic. Called twice when STEP 6 triggers a rewrite -- once
    with the full context, once with the scoped correction context.
    """
    ####################################################
    # STEP 1: CREATE THE SKILL-OPTIMIZER AGENT#
    ####################################################
    agent = create_skill_optimizer_agent()

    ####################################################
    # STEP 2: RUN THE TASK AND RETURN THE TYPED RESULT#
    ####################################################
    return run_agent_task(
        agent=agent,
        task_name="optimize_skills_section_task",
        context=context,
        output_model=OptimizedSkillsSection,
        run_id=run_id,
    )


def audit_skills_section(
    original_resume: Resume,
    optimized_skills: OptimizedSkillsSection,
) -> ReviewResult:
    """Run the code-owned evidence audit on the optimized skills.

    Serves node STEP 5.
    """
    ####################################################
    # STEP 1: BUILD AN AUDIT RESUME -- OPTIMIZED SKILLS, ORIGINAL EVIDENCE#
    ####################################################
    # The original resume's experience, education, and certifications are the
    # evidence corpus; only the skill list under test changes.
    audit_resume = original_resume.model_copy(update={"skills": optimized_skills.optimized_skills})

    ####################################################
    # STEP 2: RETURN THE PER-SKILL JUDGMENT FINDINGS#
    ####################################################
    return validate_skills_evidence(audit_resume)


def is_confident_unsupported(comment: ReviewComment) -> bool:
    """The single gate for acting on a skill finding: serious AND high confidence.

    HIGH means the skill is concrete and clearly absent from every part of the resume.
    MEDIUM ("likely unsupported") and LOW ("the field may implicitly cover it") are both
    advisory: the candidate listed the skill, so we never delete it on the model's hunch,
    only on its confident judgement. This is what keeps a thin evidence corpus -- e.g. an
    extractor that captured sparse skills_used and empty role descriptions -- from
    stripping skills the candidate truthfully has. The rewrite trigger and the removal
    list share this one predicate so they can never disagree on what counts.
    """
    return comment.severity in _SERIOUS_SEVERITIES and comment.confidence == Confidence.HIGH


def skills_audit_needs_rewrite(audit_result: ReviewResult) -> bool:
    """Return True when a confidently-unsupported skill warrants one rewrite.

    Serves node STEP 5, gates node STEP 6.
    """
    return any(is_confident_unsupported(comment) for comment in audit_result.comments)


def flagged_skill_names(audit_result: ReviewResult) -> list[str]:
    """Names of the skills the audit confidently flagged for removal.

    Serves node STEP 6. quoted_text carries the skill name; message is a
    defensive fallback if a producer leaves it blank.
    """
    return [
        comment.quoted_text or comment.message
        for comment in audit_result.comments
        if is_confident_unsupported(comment)
    ]


def build_skills_rewrite_context(
    section: OptimizedSkillsSection,
    audit_result: ReviewResult,
) -> str:
    """Build the scoped context for exactly one rewrite attempt.

    Serves node STEP 6. Passes only the current skills and the exact names to
    drop -- the evidence judgement is already done, so the rewrite never
    re-sees the job requirements, ats_keywords, or strategy that could tempt
    it to re-infer skills.
    """
    return format_skills_rewrite_context(
        section=section,
        skills_to_remove=flagged_skill_names(audit_result),
    )


def preserve_original_skills(
    optimized_skills: OptimizedSkillsSection,
    original_resume: Resume,
) -> OptimizedSkillsSection:
    """Re-add any original-resume skill the optimizer dropped, appended after its ordering.

    Serves node STEP 7. The candidate's listed skills are truthful facts; the
    optimizer's job is to reorder and categorize them, not delete them.
    Re-adds only skills that were already in the original resume, so it can
    never introduce a fabricated skill. Returns the section unchanged when
    the optimizer kept every original skill.
    """
    ####################################################
    # STEP 1: FIND ORIGINAL SKILLS MISSING FROM THE OPTIMIZED LIST#
    ####################################################
    existing_names = {skill.skill_name.casefold() for skill in optimized_skills.optimized_skills}
    dropped = [
        skill
        for skill in original_resume.skills
        if skill.skill_name.casefold() not in existing_names
    ]

    ####################################################
    # STEP 2: APPEND ANY DROPPED SKILLS BACK ONTO THE SECTION#
    ####################################################
    if not dropped:
        return optimized_skills
    return optimized_skills.model_copy(
        update={"optimized_skills": optimized_skills.optimized_skills + dropped}
    )
