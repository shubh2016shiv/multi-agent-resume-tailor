"""Build context for the professional summary writer.

Caller:
- `src/orchestration/nodes/summary.py`

Consumer:
- the `write_professional_summary_task`

This formatter keeps:
- the candidate's strongest experience highlights, top skills, and education
- the job title, role summary, and high-priority requirements
- only supported strategy priorities for summary writing

This formatter drops:
- contact information
- the candidate's existing summary text, which over-anchors the writer
- full job text and the raw ATS keyword list (mostly unsupported gap terms
  that invite keyword-stuffing; the strategy's supported vocabulary replaces it)
- the overall fit score (a scalar verdict the writer has no use for)
- low-value strategy detail meant for other agents

Toy example:
    If the resume has ten roles and thirty skills, this formatter keeps only
    the summary-relevant highlights and returns one compact context string.
"""

from typing import Any

from src.data_models.job import JobDescription, SkillImportance
from src.data_models.resume import Resume
from src.data_models.strategy import AlignmentStrategy, SkillMatch
from src.formatters.llm_context_rendering import OutputFormat, render_context_data

PROFICIENCY_PRIORITY = {
    "expert": 4,
    "advanced": 3,
    "intermediate": 2,
    "beginner": 1,
}

MAX_SUMMARY_PRIORITY_TERMS = 8
MAX_SUMMARY_EXPERIENCE_HIGHLIGHTS = 3


def _score_experience_relevance(experience: Any, supported_terms: list[str]) -> int:
    """Score one role by how many supported role terms it already evidences."""
    experience_text = " ".join(
        [
            experience.job_title,
            experience.description,
            *experience.achievements,
            *experience.skills_used,
        ]
    ).casefold()
    return sum(1 for term in supported_terms if term.casefold() in experience_text)


def select_resume_context(resume: Resume, supported_terms: list[str]) -> dict[str, Any]:
    """Keep only the resume fields the summary writer needs."""
    top_skills = sorted(
        resume.skills,
        key=lambda skill: PROFICIENCY_PRIORITY.get((skill.proficiency_level or "").casefold(), 0),
        reverse=True,
    )

    ranked_experiences = resume.work_experience
    if supported_terms:
        ranked_experiences = [
            experience
            for _, experience in sorted(
                enumerate(resume.work_experience),
                key=lambda item: (
                    -_score_experience_relevance(item[1], supported_terms),
                    item[0],
                ),
            )
        ]

    experience_highlights = []
    for experience in ranked_experiences[:MAX_SUMMARY_EXPERIENCE_HIGHLIGHTS]:
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


def _supported_role_vocabulary(top_matches: list[SkillMatch]) -> list[str]:
    """Return deduplicated supported role vocabulary from real matches only."""
    ordered_terms: list[str] = []
    seen: set[str] = set()

    for match in top_matches:
        for term in (match.job_requirement, match.resume_skill):
            clean = term.strip()
            if not clean:
                continue
            key = clean.casefold()
            if key in seen:
                continue
            seen.add(key)
            ordered_terms.append(clean)
            if len(ordered_terms) >= MAX_SUMMARY_PRIORITY_TERMS:
                return ordered_terms

    return ordered_terms


def _focused_summary_guidance(
    strategy: AlignmentStrategy,
    supported_terms: list[str],
) -> str:
    """Focus summary guidance on supported priorities when real matches exist."""
    if not supported_terms:
        return strategy.professional_summary_guidance

    # Achievements outrank vocabulary. An earlier version of this guidance named only
    # the vocabulary list as the explicit priority, and live runs showed the writer
    # checklist-completing those terms while dropping every quantified achievement
    # from the highlights -- the summary read like a category description.
    return (
        "The summary's one job is to make a recruiter want to read the rest of the "
        "resume. Open with the candidate's defining professional identity aimed at "
        "this role. Prove it with the one or two strongest, most role-relevant "
        "achievements from the experience highlights: when a highlight states a "
        "measurable outcome, carry that figure into the summary exactly as written "
        "-- concrete results outrank category words. When the evidence has no "
        "numbers, use its most concrete specifics instead. "
        f"Use this supported role vocabulary where it is the truthful wording: "
        f"{', '.join(supported_terms)}. "
        "Do not mirror the original resume summary, and do not spend summary space "
        "on unsupported gaps."
    )


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

    # The raw ATS keyword list is deliberately NOT passed to the summary writer:
    # most of its entries are unsupported gap terms (the skills/ATS stages own
    # those), and handing the writer a keyword list invites stuffing over
    # evidence. The strategy context already carries the evidence-verified
    # supported_role_vocabulary subset.
    return {
        "job_title": job_description.job_title,
        "job_level": job_description.job_level.value,
        "summary": job_description.summary,
        "requirements": requirements_for_summary,
    }


def select_strategy_context(strategy: AlignmentStrategy) -> dict[str, Any]:
    """Keep only the strategy fields the summary writer should read."""
    top_matches = sorted(
        strategy.identified_matches,
        key=lambda match: match.match_score,
        reverse=True,
    )
    supported_terms = _supported_role_vocabulary(top_matches)

    # overall_fit_score is deliberately omitted: the writer has no use for a
    # scalar fit verdict, and a low score can only bias the writing toward
    # hedged, apologetic phrasing about the candidate.
    return {
        "summary_of_strategy": strategy.summary_of_strategy,
        "professional_summary_guidance": _focused_summary_guidance(
            strategy, supported_terms
        ),
        "supported_role_vocabulary": supported_terms,
        "top_matches": [
            {
                "resume_skill": match.resume_skill,
                "job_requirement": match.job_requirement,
                "match_score": match.match_score,
                "justification": match.justification,
            }
            for match in top_matches[:7]
        ],
    }


def build_professional_summary_payload(
    resume: Resume,
    job_description: JobDescription,
    strategy: AlignmentStrategy,
) -> dict[str, Any]:
    """Build the filtered payload for the summary writer."""
    ####################################################
    # STEP 1: KEEP ONLY THE STRATEGY GUIDANCE MEANT FOR SUMMARY WRITING#
    ####################################################
    strategy_context = select_strategy_context(strategy)

    ####################################################
    # STEP 2: KEEP ONLY THE JOB SIGNALS THE SUMMARY SHOULD ANSWER#
    ####################################################
    job_context = select_job_context(job_description)

    ####################################################
    # STEP 3: KEEP ONLY THE CANDIDATE BACKGROUND NEEDED FOR THE NARRATIVE#
    ####################################################
    resume_context = select_resume_context(
        resume,
        strategy_context["supported_role_vocabulary"],
    )

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
