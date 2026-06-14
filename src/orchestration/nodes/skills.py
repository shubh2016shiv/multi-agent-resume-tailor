"""Stage 3 skills optimization node.

Flow: write -> code-owned evidence audit -> one rewrite if the audit finds
unsupported skill claims with high or medium confidence.
"""

from src.agents.skill_optimizer import create_skill_optimizer_agent
from src.data_models.job import JobDescription
from src.data_models.resume import OptimizedSkillsSection, Resume, Skill
from src.formatters.skills_optimizer_formatter import format_skills_optimizer_context
from src.orchestration.crew_task_execution import run_agent_task
from src.orchestration.state import ResumeEnhancementPipelineState
from src.tools.job_matching import keyword_present_in_text
from src.tools.review_contract.review_models import Confidence, ReviewResult, Severity
from src.tools.shared.resume_rendering import render_resume
from src.tools.truthfulness.skills_evidence_validator import validate_skills_evidence

# Mechanical whole-token matches are certain, so recovered skills carry full confidence.
_RECOVERED_SKILL_CONFIDENCE = 100.0

_SERIOUS_SEVERITIES = {Severity.BLOCKER, Severity.MAJOR}


def optimize_skills(state: ResumeEnhancementPipelineState) -> dict:
    """Reorder, categorize, and ATS-optimize the skills section.

    Reads: resume, job_description, and alignment_strategy from prior stages.
    Writes: optimized_skills.
    Returns: partial state with the typed OptimizedSkillsSection.
    """
    assert state["resume"] is not None, "resume must be set before skills optimization"
    assert state["job_description"] is not None, "job_description must be set before skills optimization"
    assert state["alignment_strategy"] is not None, "alignment_strategy must be set before skills optimization"
    resume = state["resume"]
    context = format_skills_optimizer_context(
        resume=resume,
        job_description=state["job_description"],
        strategy=state["alignment_strategy"],
        format_type="toon",
    )

    optimized_skills = _write_skills_section(context)
    audit_result = _audit_skills_section(resume, optimized_skills)

    if _skills_audit_needs_rewrite(audit_result):
        rewrite_context = _build_skills_rewrite_context(context, optimized_skills, audit_result)
        optimized_skills = _write_skills_section(rewrite_context)

    # Recover JD keywords the resume evidences but the agent dropped from the section
    # (e.g. Docker/Kubernetes present in experience but absent from a thin skills list).
    optimized_skills = _add_evidenced_jd_keywords(optimized_skills, resume, state["job_description"])
    return {"optimized_skills": optimized_skills}


def _write_skills_section(context: str) -> OptimizedSkillsSection:
    """Ask the skill optimizer agent to produce a skills section.

    Expects TOON context with resume skills, job requirements, and strategy.
    Returns an OptimizedSkillsSection validated by CrewAI output_pydantic.
    """
    return run_agent_task(
        agent=create_skill_optimizer_agent(),
        task_name="optimize_skills_section_task",
        context=context,
        output_model=OptimizedSkillsSection,
    )


def _audit_skills_section(
    original_resume: Resume,
    optimized_skills: OptimizedSkillsSection,
) -> ReviewResult:
    """Run the code-owned evidence audit on the optimized skills.

    Builds an audit resume with the optimized skill list against the original
    resume's experience, education, and certifications as the evidence corpus.
    Returns a ReviewResult with per-skill judgment findings.
    """
    audit_resume = original_resume.model_copy(
        update={"skills": optimized_skills.optimized_skills}
    )
    return validate_skills_evidence(audit_resume)


def _skills_audit_needs_rewrite(audit_result: ReviewResult) -> bool:
    """Return True when unsupported skills warrant one rewrite.

    Triggers on BLOCKER or MAJOR findings with HIGH or MEDIUM confidence.
    LOW confidence is advisory — the model is unsure whether the field
    implicitly covers the skill and should not force a rewrite.
    """
    return any(
        comment.severity in _SERIOUS_SEVERITIES and comment.confidence != Confidence.LOW
        for comment in audit_result.comments
    )


def _render_skills_audit_feedback(audit_result: ReviewResult) -> str:
    """Render audit comments into compact plain text for the rewrite context."""
    lines = [audit_result.summary or "Skills audit found unsupported skill claims."]
    for comment in audit_result.comments:
        lines.append(
            f"- {comment.severity.value} ({comment.confidence.value}): {comment.message}"
        )
        lines.append(f"  advice: {comment.advice}")
    return "\n".join(lines)


def _build_skills_rewrite_context(
    original_context: str,
    section: OptimizedSkillsSection,
    audit_result: ReviewResult,
) -> str:
    """Append previous output and audit feedback to the original context.

    Returns context for exactly one rewrite attempt.
    """
    return (
        f"{original_context}\n\n"
        f"PREVIOUS_OPTIMIZED_SKILLS_JSON:\n{section.model_dump_json()}\n\n"
        f"SKILLS_AUDIT_FEEDBACK:\n{_render_skills_audit_feedback(audit_result)}\n\n"
        "Remove every flagged unsupported skill and return corrected OptimizedSkillsSection JSON."
    )


def _add_evidenced_jd_keywords(
    optimized_skills: OptimizedSkillsSection,
    original_resume: Resume,
    job: JobDescription,
) -> OptimizedSkillsSection:
    """Re-insert JD keywords the resume evidences but the skills section dropped.

    For each job.ats_keywords entry not already in optimized_skills and present in the
    rendered original resume (experience, skills_used, original skills), append it as a
    Skill. Mechanical whole-token match means every added skill is truthfully evidenced;
    nothing not already in the resume is ever fabricated. Returns the section unchanged
    when no evidenced keyword is missing.
    """
    existing_names = {skill.skill_name.casefold() for skill in optimized_skills.optimized_skills}
    resume_text = render_resume(original_resume)
    recovered = [
        _build_recovered_skill(keyword)
        for keyword in job.ats_keywords
        if keyword.casefold() not in existing_names
        and keyword_present_in_text(keyword, resume_text)
    ]
    if not recovered:
        return optimized_skills
    return optimized_skills.model_copy(
        update={
            "optimized_skills": optimized_skills.optimized_skills + recovered,
            "added_skills": optimized_skills.added_skills + recovered,
        }
    )


def _build_recovered_skill(keyword: str) -> Skill:
    """Build a Skill for a JD keyword recovered from resume evidence into the section."""
    return Skill(
        skill_name=keyword,
        category=None,
        proficiency_level=None,
        years_of_experience=None,
        justification=(
            "JD keyword present in the original resume (experience or skills); recovered "
            "into the skills section for ATS coverage."
        ),
        evidence=[f"Resume text contains '{keyword}'."],
        confidence_score=_RECOVERED_SKILL_CONFIDENCE,
    )
