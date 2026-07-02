"""Build the context string the professional summary writer reads.

Pipeline position: this is sub-step 2 of the professional-summary pipeline
(see write_professional_summary in src/orchestration/nodes/summary.py, STEP 2).
The node hands three large objects (Resume, JobDescription, AlignmentStrategy)
to this module; this module returns one compact, writer-focused context string.

What this formatter keeps:
- the candidate's strongest experience highlights, top skills, and education
- the job title, role summary, and high-priority requirements
- only the supported strategy priorities relevant to summary writing

What this formatter drops (and why):
- contact information -- the writer never needs it
- the candidate's existing summary text -- it over-anchors the writer
- full job text and the raw ATS keyword list -- mostly unsupported gap terms
  that invite keyword-stuffing; the strategy's supported vocabulary replaces it
- the overall fit score -- a scalar verdict the writer has no use for, and a low
  score only biases the writing toward hedged, apologetic phrasing
- low-value strategy detail meant for other agents

Why the caps below exist: this module's entire job is noise reduction -- turn a
full resume/JD/strategy into only the few facts a summary can actually use. Every
MAX_* constant is a deliberate cap serving that job. The specific figures are
editorial (how much a 3-4 sentence, 80-110 word summary can draw on before the
context is just noise), tuned against live runs, not derived from a formula.
"""

from typing import Any

from src.data_models.job import JobDescription, SkillImportance
from src.data_models.resume import Resume
from src.data_models.strategy import AlignmentStrategy, SkillMatch
from src.formatters.llm_context_rendering import OutputFormat, render_context_data

# Skill proficiency ranked so the strongest-declared skills surface first.
PROFICIENCY_PRIORITY = {
    "expert": 4,
    "advanced": 3,
    "intermediate": 2,
    "beginner": 1,
}

# Noise-reduction caps (see module docstring). Each bounds one list the writer reads.
MAX_SUPPORTED_VOCABULARY_TERMS = 8   # role terms the summary may lean on
MAX_EXPERIENCE_HIGHLIGHTS = 3        # roles carried into the narrative
MAX_ACHIEVEMENTS_PER_ROLE = 3        # bullets kept per carried role
MAX_SKILLS = 15                      # top skills by proficiency
MAX_STRATEGY_MATCHES = 7             # resume<->JD matches shown as evidence


def count_supported_terms_in_experience(experience: Any, supported_terms: list[str]) -> int:
    """Count how many supported role terms one role already evidences.

    Used to rank roles by relevance before the top MAX_EXPERIENCE_HIGHLIGHTS are kept.
    """
    ####################################################
    # STEP 1: FLATTEN THE ROLE INTO ONE SEARCHABLE TEXT#
    ####################################################
    experience_text = " ".join(
        [
            experience.job_title,
            experience.description,
            *experience.achievements,
            *experience.skills_used,
        ]
    ).casefold()

    ####################################################
    # STEP 2: COUNT SUPPORTED TERMS PRESENT IN THAT TEXT#
    ####################################################
    return sum(1 for term in supported_terms if term.casefold() in experience_text)


def rank_roles_by_relevance(resume: Resume, supported_terms: list[str]) -> list:
    """Return the resume's roles ordered most-relevant first.

    "Relevance" is simply how many supported role terms a role mentions. With no
    supported terms to rank by, the resume's own order is kept. sorted() is stable,
    so roles that tie on relevance also keep their original order.
    """
    ####################################################
    # STEP 1: NOTHING TO RANK BY -> KEEP RESUME ORDER#
    ####################################################
    if not supported_terms:
        return resume.work_experience

    ####################################################
    # STEP 2: SORT BY RELEVANCE COUNT, HIGHEST FIRST#
    ####################################################
    return sorted(
        resume.work_experience,
        key=lambda role: count_supported_terms_in_experience(role, supported_terms),
        reverse=True,
    )


def shape_experience_highlight(role: Any) -> dict[str, Any]:
    """Turn one role into the compact highlight shape the writer reads."""
    return {
        "job_title": role.job_title,
        "company_name": role.company_name,
        "date_range": f"{role.start_date} - {role.end_date or 'Present'}",
        "description": role.description,
        "top_achievements": role.achievements[:MAX_ACHIEVEMENTS_PER_ROLE],
        "skills_used": list(role.skills_used),
    }


def select_resume_context(resume: Resume, supported_terms: list[str]) -> dict[str, Any]:
    """Keep only the resume fields the summary writer needs, ranked for usefulness.

    Serves node STEP 2. Ranks skills by proficiency and roles by relevance, then
    caps each list to its MAX_* bound.
    """
    ####################################################
    # STEP 1: RANK SKILLS BY DECLARED PROFICIENCY#
    ####################################################
    skills_by_proficiency = sorted(
        resume.skills,
        key=lambda skill: PROFICIENCY_PRIORITY.get((skill.proficiency_level or "").casefold(), 0),
        reverse=True,
    )

    ####################################################
    # STEP 2: RANK ROLES, THEN KEEP THE TOP FEW#
    ####################################################
    ranked_roles = rank_roles_by_relevance(resume, supported_terms)
    top_roles = ranked_roles[:MAX_EXPERIENCE_HIGHLIGHTS]
    experience_highlights = [shape_experience_highlight(role) for role in top_roles]

    ####################################################
    # STEP 3: RETURN THE WRITER-FOCUSED RESUME SLICE#
    ####################################################
    return {
        "total_years_of_experience": round(resume.total_years_of_experience, 1),
        "experience_highlights": experience_highlights,
        "top_skills": [
            {
                "skill_name": skill.skill_name,
                "proficiency_level": skill.proficiency_level or "Not Specified",
                "years_of_experience": skill.years_of_experience,
            }
            for skill in skills_by_proficiency[:MAX_SKILLS]
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


def collect_supported_vocabulary(matches: list[SkillMatch]) -> list[str]:
    """Return de-duplicated role vocabulary drawn only from real resume<->JD matches.

    Each match names a JD requirement and the resume skill that satisfies it; both
    are worth keeping. We walk the matches in order, keep each new term once, and
    stop at MAX_SUPPORTED_VOCABULARY_TERMS.
    """
    supported_terms: list[str] = []
    already_added: set[str] = set()

    for match in matches:
        for term in (match.job_requirement, match.resume_skill):
            ####################################################
            # STEP 1: SKIP BLANKS AND TERMS ALREADY ADDED#
            ####################################################
            clean_term = term.strip()
            if not clean_term:
                continue
            if clean_term.casefold() in already_added:
                continue

            ####################################################
            # STEP 2: KEEP THIS NEW TERM#
            ####################################################
            supported_terms.append(clean_term)
            already_added.add(clean_term.casefold())

            ####################################################
            # STEP 3: STOP ONCE THE CAP IS REACHED#
            ####################################################
            if len(supported_terms) >= MAX_SUPPORTED_VOCABULARY_TERMS:
                return supported_terms

    return supported_terms


def build_summary_guidance(strategy: AlignmentStrategy, supported_terms: list[str]) -> str:
    """Write the single guidance paragraph that tells the writer what to prioritize.

    Achievements outrank vocabulary. An earlier version named only the vocabulary
    list as the explicit priority, and live runs showed the writer checklist-
    completing those terms while dropping every quantified achievement from the
    highlights -- the summary read like a category description. When there are no
    supported matches to focus on, fall back to the strategy's own guidance.
    """
    ####################################################
    # STEP 1: FALL BACK WHEN THERE ARE NO SUPPORTED TERMS#
    ####################################################
    if not supported_terms:
        return strategy.professional_summary_guidance

    ####################################################
    # STEP 2: LEAD WITH ACHIEVEMENTS, VOCABULARY SECOND#
    ####################################################
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
    """Keep only the job fields the summary writer needs.

    Serves node STEP 2. The raw ATS keyword list is deliberately excluded: most of
    its entries are unsupported gap terms owned by the skills/ATS stages, and handing
    the writer a keyword list invites stuffing over evidence. The strategy context
    already carries the evidence-verified supported vocabulary.
    """
    ####################################################
    # STEP 1: KEEP MUST-HAVE AND SHOULD-HAVE REQUIREMENTS#
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
    # STEP 2: RETURN THE WRITER-FOCUSED JOB SLICE#
    ####################################################
    return {
        "job_title": job_description.job_title,
        "job_level": job_description.job_level.value,
        "summary": job_description.summary,
        "requirements": requirements_for_summary,
    }


def select_strategy_context(strategy: AlignmentStrategy) -> dict[str, Any]:
    """Keep only the strategy fields the summary writer should read.

    Serves node STEP 2. Ranks matches by score, derives the supported vocabulary,
    and writes the guidance paragraph. The overall fit score is intentionally omitted
    (see module docstring).
    """
    ####################################################
    # STEP 1: RANK MATCHES BY SCORE, STRONGEST FIRST#
    ####################################################
    matches_by_score = sorted(
        strategy.identified_matches,
        key=lambda match: match.match_score,
        reverse=True,
    )

    ####################################################
    # STEP 2: DERIVE THE SUPPORTED ROLE VOCABULARY#
    ####################################################
    supported_terms = collect_supported_vocabulary(matches_by_score)

    ####################################################
    # STEP 3: RETURN THE WRITER-FOCUSED STRATEGY SLICE#
    ####################################################
    return {
        "summary_of_strategy": strategy.summary_of_strategy,
        "professional_summary_guidance": build_summary_guidance(strategy, supported_terms),
        "supported_role_vocabulary": supported_terms,
        "top_matches": [
            {
                "resume_skill": match.resume_skill,
                "job_requirement": match.job_requirement,
                "match_score": match.match_score,
                "justification": match.justification,
            }
            for match in matches_by_score[:MAX_STRATEGY_MATCHES]
        ],
    }


def build_professional_summary_payload(
    resume: Resume,
    job_description: JobDescription,
    strategy: AlignmentStrategy,
) -> dict[str, Any]:
    """Assemble the three writer-focused slices into one payload (node STEP 2, part 1)."""
    ####################################################
    # STEP 1: STRATEGY SLICE (RUN FIRST -- IT YIELDS#
    #         THE SUPPORTED VOCABULARY THE RESUME#
    #         SLICE RANKS AGAINST)#
    ####################################################
    strategy_context = select_strategy_context(strategy)

    ####################################################
    # STEP 2: JOB SLICE#
    ####################################################
    job_context = select_job_context(job_description)

    ####################################################
    # STEP 3: RESUME SLICE, RANKED AGAINST THE VOCABULARY#
    ####################################################
    resume_context = select_resume_context(
        resume,
        strategy_context["supported_role_vocabulary"],
    )

    ####################################################
    # STEP 4: COMBINE INTO THE FINAL PAYLOAD#
    ####################################################
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
    """Return the summary writer's context string (node STEP 2, entry point)."""
    ####################################################
    # STEP 1: BUILD THE SMALL PAYLOAD THE WRITER NEEDS#
    ####################################################
    payload = build_professional_summary_payload(resume, job_description, strategy)

    ####################################################
    # STEP 2: RENDER THE PAYLOAD INTO THE LLM FORMAT#
    ####################################################
    return render_context_data(
        payload,
        format_type=format_type,
        description="Professional Summary Context",
    )
