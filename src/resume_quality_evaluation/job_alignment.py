"""Evaluate how well the tailored resume aligns with the target job."""

from src.data_models.evaluation import JobAlignmentEvaluation
from src.data_models.job import JobDescription, JobRequirement, SkillImportance
from src.data_models.resume import Resume, Skill
from src.resume_quality_evaluation.skill_similarity_match import (
    is_required_skill_evidenced,
    match_term_for_skill,
)
from src.tools.engines.document_rendering.resume_text_renderer import render_resume
from src.tools.engines.job_matching import keyword_present_in_text

_IMPORTANCE_WEIGHTS = {
    SkillImportance.MUST_HAVE: 3,
    SkillImportance.SHOULD_HAVE: 2,
    SkillImportance.NICE_TO_HAVE: 1,
}


def evaluate_job_alignment(revised: Resume, job: JobDescription) -> JobAlignmentEvaluation:
    """Evaluate structured requirements, falling back to ATS keywords when needed.

    Expects a tailored resume and its target job. Each structured requirement is
    matched against the resume's skills by embedding similarity. Returns an
    inconclusive zero score only when the job has neither structured requirements
    nor ATS keywords.
    """
    resume_text = render_resume(revised)
    ats_coverage, missed_keywords = _calculate_term_coverage(job.ats_keywords, resume_text)
    if job.requirements:
        return _evaluate_requirements(job.requirements, revised.skills, ats_coverage)
    if job.ats_keywords:
        return JobAlignmentEvaluation(
            relevance_score=ats_coverage,
            must_have_skills_coverage=0.0,
            ats_keyword_coverage=ats_coverage,
            missed_requirements=missed_keywords,
            justification="No structured requirements; used ATS keyword coverage as fallback.",
        )
    return JobAlignmentEvaluation(
        relevance_score=0.0,
        must_have_skills_coverage=0.0,
        ats_keyword_coverage=0.0,
        missed_requirements=[],
        is_conclusive=False,
        justification="Target job has no structured requirements or ATS keywords to evaluate.",
    )


def _evaluate_requirements(
    requirements: list[JobRequirement],
    resume_skills: list[Skill],
    ats_keyword_coverage: float,
) -> JobAlignmentEvaluation:
    """Return importance-weighted alignment from per-requirement similarity matches."""
    ####################################################
    # STEP 1: BUILD THE RESUME SKILL CANDIDATE SET ONCE#
    ####################################################
    # Every requirement is matched against the same canonicalized skill set, so we
    # build it once. Falls back to the raw skill name when extraction set none.
    candidate_skills = [match_term_for_skill(skill) for skill in resume_skills]

    ####################################################
    # STEP 2: MATCH EACH REQUIREMENT AGAINST THE SKILL SET#
    ####################################################
    # is_required_skill_evidenced owns the similarity threshold; a requirement either
    # matches a resume skill or is a real gap (no fractional term-group credit).
    matches = [
        (requirement, is_required_skill_evidenced(_requirement_term_for(requirement), candidate_skills))
        for requirement in requirements
    ]

    ####################################################
    # STEP 3: WEIGHT COVERAGE BY REQUIREMENT IMPORTANCE#
    ####################################################
    # Must-have gaps cost the most; relevance is the matched share of total weight.
    total_weight = sum(_IMPORTANCE_WEIGHTS[requirement.importance] for requirement, _ in matches)
    matched_weight = sum(
        _IMPORTANCE_WEIGHTS[requirement.importance] for requirement, matched in matches if matched
    )
    must_have_results = [
        matched for requirement, matched in matches if requirement.importance is SkillImportance.MUST_HAVE
    ]
    must_have_coverage = (
        _calculate_percentage(sum(must_have_results), len(must_have_results))
        if must_have_results
        else 0.0
    )
    return JobAlignmentEvaluation(
        relevance_score=_calculate_percentage(matched_weight, total_weight),
        must_have_skills_coverage=must_have_coverage,
        ats_keyword_coverage=ats_keyword_coverage,
        missed_requirements=[requirement.requirement for requirement, matched in matches if not matched],
        is_conclusive=True,
        justification="Importance-weighted similarity match of requirements to resume skills.",
    )


def _requirement_term_for(requirement: JobRequirement) -> str:
    """Return the requirement's canonicalized form for matching, or its raw text if absent."""
    return requirement.canonicalized_requirement or requirement.requirement


def _calculate_term_coverage(terms: list[str], resume_text: str) -> tuple[float, list[str]]:
    """Return percentage coverage and missing terms for literal ATS keywords.

    ATS keyword matching stays literal (whole-token) because real ATS systems match
    keywords literally; this is intentionally not the embedding path.
    """
    missed = [term for term in terms if not keyword_present_in_text(term, resume_text)]
    return _calculate_percentage(len(terms) - len(missed), len(terms)), missed


def _calculate_percentage(matched_count: float, total_count: float) -> float:
    """Return a one-decimal percentage, using 100 for an empty applicable set."""
    return 100.0 if total_count == 0 else round(100.0 * matched_count / total_count, 1)
