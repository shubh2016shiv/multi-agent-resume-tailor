"""Build context for the professional summary writer.

Caller:
- `src/orchestration/nodes/summary.py`

Consumer:
- the `write_professional_summary_task`

This formatter keeps:
- the candidate's current summary, strongest experience highlights, top skills,
  and education
- the job title, role summary, high-priority requirements, and ATS keywords
- the strategy guidance that is specific to summary writing

This formatter drops:
- contact information
- full job text
- low-value strategy detail meant for other agents

Toy example:
    If the resume has ten roles and thirty skills, this formatter keeps only
    the summary-relevant highlights and returns one compact context string.
"""

from typing import Any

from src.data_models.job import JobDescription, SkillImportance
from src.data_models.resume import Resume
from src.data_models.strategy import AlignmentStrategy
from src.formatters.llm_context_rendering import OutputFormat, render_context_data

PROFICIENCY_PRIORITY = {
    "expert": 4,
    "advanced": 3,
    "intermediate": 2,
    "beginner": 1,
}


def select_resume_context(resume: Resume) -> dict[str, Any]:
    """Keep only the resume fields the summary writer needs."""
    top_skills = sorted(
        resume.skills,
        key=lambda skill: PROFICIENCY_PRIORITY.get((skill.proficiency_level or "").casefold(), 0),
        reverse=True,
    )

    experience_highlights = []
    for experience in resume.work_experience:
        experience_highlights.append(
            {
                "job_title": experience.job_title,
                "company_name": experience.company_name,
                "date_range": f"{experience.start_date} - {experience.end_date or 'Present'}",
                "description": experience.description,
                "top_achievements": experience.achievements[:3],
                "skills_used": list(experience.skills_used),
            }
        )

    return {
        "current_professional_summary": resume.professional_summary,
        "total_years_of_experience": round(resume.total_years_of_experience, 1),
        "experience_highlights": experience_highlights,
        "top_skills": [
            {
                "skill_name": skill.skill_name,
                "proficiency_level": skill.proficiency_level or "Not Specified",
                "years_of_experience": skill.years_of_experience,
            }
            for skill in top_skills[:15]
        ],
        "education": [
            {
                "degree": education.degree,
                "field_of_study": education.field_of_study,
                "institution_name": education.institution_name,
                "graduation_year": education.graduation_year,
            }
            for education in resume.education
        ],
    }


def select_job_context(job_description: JobDescription) -> dict[str, Any]:
    """Keep only the job fields the summary writer needs."""
    requirements_for_summary = [
        {
            "requirement": requirement.requirement,
            "importance": requirement.importance.value,
            "years_required": requirement.years_required,
        }
        for requirement in job_description.requirements
        if requirement.importance in {SkillImportance.MUST_HAVE, SkillImportance.SHOULD_HAVE}
    ]

    return {
        "job_title": job_description.job_title,
        "job_level": job_description.job_level.value,
        "summary": job_description.summary,
        "requirements": requirements_for_summary,
        "ats_keywords": list(job_description.ats_keywords),
    }


def select_strategy_context(strategy: AlignmentStrategy) -> dict[str, Any]:
    """Keep only the strategy fields the summary writer should read."""
    top_matches = sorted(
        strategy.identified_matches,
        key=lambda match: match.match_score,
        reverse=True,
    )
    must_have_gaps = [
        gap
        for gap in strategy.identified_gaps
        if gap.importance == "must_have"
    ]

    return {
        "overall_fit_score": strategy.overall_fit_score,
        "summary_of_strategy": strategy.summary_of_strategy,
        "professional_summary_guidance": strategy.professional_summary_guidance,
        "keywords_to_integrate": list(strategy.keywords_to_integrate),
        "top_matches": [
            {
                "resume_skill": match.resume_skill,
                "job_requirement": match.job_requirement,
                "match_score": match.match_score,
                "justification": match.justification,
            }
            for match in top_matches[:7]
        ],
        "must_have_gaps": [
            {
                "missing_skill": gap.missing_skill,
                "importance": gap.importance,
                "suggestion": gap.suggestion,
            }
            for gap in must_have_gaps
        ],
    }


def build_professional_summary_payload(
    resume: Resume,
    job_description: JobDescription,
    strategy: AlignmentStrategy,
) -> dict[str, Any]:
    """Build the filtered payload for the summary writer."""
    ####################################################
    # STEP 1: KEEP ONLY THE CANDIDATE BACKGROUND NEEDED FOR THE NARRATIVE#
    ####################################################
    resume_context = select_resume_context(resume)

    ####################################################
    # STEP 2: KEEP ONLY THE JOB SIGNALS THE SUMMARY SHOULD ANSWER#
    ####################################################
    job_context = select_job_context(job_description)

    ####################################################
    # STEP 3: KEEP ONLY THE STRATEGY GUIDANCE MEANT FOR SUMMARY WRITING#
    ####################################################
    strategy_context = select_strategy_context(strategy)

    return {
        "candidate_background": resume_context,
        "target_role": job_context,
        "summary_strategy": strategy_context,
    }


def format_professional_summary_context(
    resume: Resume,
    job_description: JobDescription,
    strategy: AlignmentStrategy,
    format_type: OutputFormat = "toon",
) -> str:
    """Return the professional summary writer's context string."""
    ####################################################
    # STEP 1: BUILD THE SMALL DATA PAYLOAD THE SUMMARY WRITER ACTUALLY NEEDS#
    ####################################################
    payload = build_professional_summary_payload(resume, job_description, strategy)

    ####################################################
    # STEP 2: RENDER THAT PAYLOAD INTO THE REQUESTED OUTPUT FORMAT#
    ####################################################
    return render_context_data(
        payload,
        format_type=format_type,
        description="Professional Summary Context",
    )
