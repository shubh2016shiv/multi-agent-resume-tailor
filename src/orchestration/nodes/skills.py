"""Stage 3 (parallel with summary and experience): optimize the skills section.

What optimize_skills does, in order, and which module does the work:

    build the optimizer's context         -> skills_optimizer_formatter.py
    run the skill-optimizer agent         -> skill_optimizer/agent.py,
                                             via crew_task_execution.py
    audit the result for evidence         -> truthfulness/skills_evidence.py
    rewrite once if the audit is certain  -> skills_optimizer_formatter.py builds a
                                             scoped context naming the skills to drop
    re-add skills the agent dropped          (preserve_original_skills, this file)

The last three steps are code, not agent, because the agent gets two things wrong:
it will claim a skill the resume does not evidence, and it will silently drop skills
when asked to reorder a long list.

Reads from state: resume, job_description, alignment_strategy.
Writes to state: optimized_skills.
"""

from src.agents.skill_optimizer import create_skill_optimizer_agent
from src.core.logger import get_logger
from src.data_models.resume import OptimizedSkillsSection, Resume
from src.formatters.skills_optimizer_formatter import (
    format_skills_optimizer_context,
    format_skills_rewrite_context,
)
from src.orchestration.crew_task_execution import run_agent_task
from src.orchestration.nodes._stage import pipeline_stage
from src.orchestration.state import ResumeEnhancementPipelineState, require
from src.tools.contracts import Confidence, ReviewComment, ReviewResult, Severity
from src.tools.engines.truthfulness.skills_evidence import validate_skills_evidence

logger = get_logger(__name__)

_SERIOUS_SEVERITIES = {Severity.BLOCKER, Severity.MAJOR}


@pipeline_stage("optimize_skills")
def optimize_skills(state: ResumeEnhancementPipelineState) -> dict:
    """Reorder, categorize, and ATS-optimize the skills section.

    Reads: resume, job_description, and alignment_strategy from prior stages.
    Writes: optimized_skills.
    Returns: partial state with the typed OptimizedSkillsSection.
    """
    resume = require(state["resume"], "resume")

    context = format_skills_optimizer_context(
        resume=resume,
        job_description=require(state["job_description"], "job_description"),
        strategy=require(state["alignment_strategy"], "alignment_strategy"),
        format_type="toon",
    )

    optimized_skills = write_skills_section(context, run_id=state["run_id"])

    audit_result = audit_skills_section(resume, optimized_skills)
    needs_rewrite = skills_audit_needs_rewrite(audit_result)
    logger.info(
        "skills_audit_completed",
        findings_count=len(audit_result.comments),
        needs_rewrite=needs_rewrite,
    )

    if needs_rewrite:
        logger.info("skills_rewrite_triggered", findings_count=len(audit_result.comments))
        # The rewrite sees only the current skills and the names to drop -- not the job
        # requirements or strategy, which would tempt it to re-infer the flagged skill.
        # The evidence judgement has already been made by the audit above.
        rewrite_context = format_skills_rewrite_context(
            section=optimized_skills,
            skills_to_remove=flagged_skill_names(audit_result),
        )
        optimized_skills = write_skills_section(rewrite_context, run_id=state["run_id"])

    # Completeness is guaranteed in code, never left to the agent.
    optimized_skills = preserve_original_skills(optimized_skills, resume)

    return {"optimized_skills": optimized_skills}


def write_skills_section(context: str, run_id: str) -> OptimizedSkillsSection:
    """Ask the skill optimizer agent to produce a skills section.

    Expects TOON context carrying the resume skills, job requirements, and strategy.
    Returns an OptimizedSkillsSection, validated by run_agent_task from the agent's raw
    output. Called a second time when the evidence audit triggers a rewrite, with the
    scoped correction context instead of the full one.
    """
    agent = create_skill_optimizer_agent()

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
    """Run the code-owned evidence audit on the optimized skills."""
    # The original resume's experience, education, and certifications are the
    # evidence corpus; only the skill list under test changes.
    audit_resume = original_resume.model_copy(update={"skills": optimized_skills.optimized_skills})

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
    """Return True when a confidently-unsupported skill warrants the one rewrite."""
    return any(is_confident_unsupported(comment) for comment in audit_result.comments)


def flagged_skill_names(audit_result: ReviewResult) -> list[str]:
    """Names of the skills the audit confidently flagged for removal.

    quoted_text carries the skill name; message is a fallback for a producer that
    leaves it blank.
    """
    return [
        comment.quoted_text or comment.message
        for comment in audit_result.comments
        if is_confident_unsupported(comment)
    ]


def preserve_original_skills(
    optimized_skills: OptimizedSkillsSection,
    original_resume: Resume,
) -> OptimizedSkillsSection:
    """Re-add any original-resume skill the optimizer dropped, appended after its ordering.

    The candidate's listed skills are facts, so the optimizer may reorder and categorize
    them but not delete them. Only skills already present in the original resume are
    re-added, so this can never introduce a fabricated one, and the section comes back
    unchanged when nothing was dropped.

    Names are compared casefolded, so a skill the optimizer merely re-cased counts as
    kept. Dropped skills are appended after the optimizer's ordering, preserving its
    ranking of the ones it did keep, and their names are pruned from removed_skills --
    that list must not claim a skill is gone while it sits in the shipped section.
    """
    existing_names = {skill.skill_name.casefold() for skill in optimized_skills.optimized_skills}
    dropped = [
        skill
        for skill in original_resume.skills
        if skill.skill_name.casefold() not in existing_names
    ]

    if not dropped:
        return optimized_skills

    # A re-added skill was never actually removed from the shipped section, so
    # leaving its name in removed_skills would claim a skill is gone when it is
    # right there -- a stale record for whichever caller reads it next.
    reinstated_names = {skill.skill_name.casefold() for skill in dropped}
    surviving_removed_skills = [
        name for name in optimized_skills.removed_skills if name.casefold() not in reinstated_names
    ]
    return optimized_skills.model_copy(
        update={
            "optimized_skills": optimized_skills.optimized_skills + dropped,
            "removed_skills": surviving_removed_skills,
        }
    )
