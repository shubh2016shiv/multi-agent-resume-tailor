"""Build the context string the professional summary writer reads.

Pipeline position: this is sub-step 2 of the professional-summary pipeline
(see write_professional_summary in src/orchestration/nodes/summary.py, STEP 2).
The node hands three large objects (Resume, JobDescription, AlignmentStrategy)
to this module; this module returns one compact, writer-focused context string.

What this formatter keeps:
- every experience highlight and every achievement in it, unmodified
- the candidate's education
- the job title, role summary, and must-have/should-have requirements
- the Gap Analysis agent's own strategy summary and summary-writing guidance

What this formatter drops (and why):
- contact information -- the writer never needs it
- the candidate's existing summary text -- it over-anchors the writer
- full job text and the raw ATS keyword list -- the job's own requirements
  already carry what the writer needs, in structured form
- the overall fit score, and the experience/skills guidance meant for other
  agents -- not this writer's business

What this formatter does NOT do, and why: it does not rank roles, cap how many
roles or achievements are shown, or maintain a standalone skills/vocabulary
list. Live experiments (see regression_fix/smoke_test/st15, st16) traced the
writer's generic, checklist-style output directly to those mechanisms -- a
capped, ranked "top skills" list and an injected "use this vocabulary"
sentence reliably produced a bare capability-list closer, and the ranking-by-
vocabulary-match cap on experience highlights silently hid a candidate's most
JD-relevant evidence in one real run. Removing a candidate's own facts to
"reduce noise" was the bug, not the fix: the writer's own evidence_used step
(see write_professional_summary_task) already does the picking, and it does
a better job of it when it can see everything.

This formatter also passes strategy.summary_of_strategy and
strategy.professional_summary_guidance through unmodified. AlignmentStrategy
documents a Blackboard Pattern (one writer -- the Gap Analysis agent -- many
readers) specifically so downstream agents read that agent's own analysis
instead of re-deriving it. An earlier version of this formatter discarded the
real guidance and substituted a hardcoded paragraph every run; that duplicated
the Gap Analysis agent's job.
"""

from typing import Any

# JobDescription = the target job; SkillImportance = the must/should/nice tier used
# to keep only the requirements worth a 3-sentence summary.
from src.data_models.job import JobDescription, SkillImportance
from src.data_models.resume import (  # candidate resume + its sub-entries
    Education,
    Experience,
    Resume,
)
from src.data_models.strategy import (
    AlignmentStrategy,  # gap-analysis output; carries summary guidance
)

# Shared rendering: OutputFormat is the "toon"/"markdown" choice; render_context_data
# turns this formatter's filtered payload dict into the final LLM context string.
from src.formatters.llm_context_rendering import OutputFormat, render_context_data


def shape_experience_highlight(experience: Experience) -> dict[str, Any]:
    """Turn one work-experience entry into the shape the summary writer reads."""
    return {
        "job_title": experience.job_title,
        "company_name": experience.company_name,
        "date_range": f"{experience.start_date} - {experience.end_date or 'Present'}",
        "description": experience.description,
        "achievements": list(experience.achievements),
        "skills_used": list(experience.skills_used),
    }


def shape_education_entry(education: Education) -> dict[str, Any]:
    """Turn one education entry into the shape the summary writer reads."""
    return {
        "degree": education.degree,
        "field_of_study": education.field_of_study,
        "institution_name": education.institution_name,
        "graduation_year": education.graduation_year,
    }


def select_resume_context(resume: Resume) -> dict[str, Any]:
    """Keep only the resume fields the summary writer needs, unmodified and uncapped.

    Serves node STEP 2. Every work-experience entry and every achievement in it is
    kept, in the resume's own order (Resume.work_experience is already sorted most-
    to least-recent) -- the writer's own evidence-gathering step decides what to use.
    """
    return {
        "total_years_of_experience": round(resume.total_years_of_experience, 1),
        "experience_highlights": [
            shape_experience_highlight(experience) for experience in resume.work_experience
        ],
        "education": [shape_education_entry(education) for education in resume.education],
    }


def select_job_context(job_description: JobDescription) -> dict[str, Any]:
    """Keep only the job fields the summary writer needs.

    Serves node STEP 2. The raw ATS keyword list and full job text are excluded:
    the structured must-have/should-have requirements already carry what a 3-
    sentence summary can use, in a form the writer can quote precisely.
    """
    ####################################################
    # STEP 1: KEEP MUST-HAVE AND SHOULD-HAVE REQUIREMENTS
    ####################################################
    important = {SkillImportance.MUST_HAVE, SkillImportance.SHOULD_HAVE}
    requirements_for_summary = [
        {
            "requirement": requirement.requirement,
            "importance": requirement.importance.value,
            "years_required": requirement.years_required,
        }
        for requirement in job_description.requirements
        if requirement.importance in important
    ]

    ####################################################
    # STEP 2: RETURN THE WRITER-FOCUSED JOB SLICE
    ####################################################
    return {
        "job_title": job_description.job_title,
        "job_level": job_description.job_level.value,
        "summary": job_description.summary,
        "requirements": requirements_for_summary,
    }


def select_strategy_context(strategy: AlignmentStrategy) -> dict[str, Any]:
    """Keep only the strategy fields the summary writer should read.

    Serves node STEP 2. Both fields are the Gap Analysis agent's own analysis for
    this candidate/job pair, passed through unmodified -- see module docstring on
    why this formatter does not regenerate them. The overall fit score and the
    other agents' guidance fields are intentionally omitted (not this writer's
    business).
    """
    return {
        "summary_of_strategy": strategy.summary_of_strategy,
        "professional_summary_guidance": strategy.professional_summary_guidance,
    }


def build_professional_summary_payload(
    resume: Resume,
    job_description: JobDescription,
    strategy: AlignmentStrategy,
) -> dict[str, Any]:
    """Assemble the three writer-focused slices into one payload (node STEP 2, part 1)."""
    return {
        "candidate_background": select_resume_context(resume),
        "target_role": select_job_context(job_description),
        "summary_strategy": select_strategy_context(strategy),
    }


def format_professional_summary_context(
    resume: Resume,
    job_description: JobDescription,
    strategy: AlignmentStrategy,
    format_type: OutputFormat = "toon",
) -> str:
    """Return the summary writer's context string (node STEP 2, entry point)."""
    ####################################################
    # STEP 1: BUILD THE SMALL PAYLOAD THE WRITER NEEDS
    ####################################################
    payload = build_professional_summary_payload(resume, job_description, strategy)

    ####################################################
    # STEP 2: RENDER THE PAYLOAD INTO THE LLM FORMAT
    ####################################################
    return render_context_data(
        payload,
        format_type=format_type,
        description="Professional Summary Context",
    )
