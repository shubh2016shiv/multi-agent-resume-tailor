"""Deterministic relevance grading: how well does the optimized resume cover the JD?

Grades the relevance dimension from mechanical keyword coverage of the optimized
resume against the job's ATS keywords -- the same whole-token matching used across
the pipeline -- rather than from an LLM's impression. Coverage is the fraction of JD
keywords present in the optimized resume text.
"""

from src.data_models.evaluation import RelevanceMetrics
from src.data_models.job import JobDescription
from src.data_models.resume import Resume
from src.tools.job_matching import keyword_present_in_text
from src.tools.shared.resume_rendering import render_resume


def grade_relevance(revised: Resume, job: JobDescription) -> RelevanceMetrics:
    """Grade the relevance dimension from mechanical JD keyword coverage.

    Expects the revised resume and the job description.
    Returns RelevanceMetrics where relevance_score and must_have_skills_coverage are
    the percentage of job.ats_keywords present in the optimized resume text, and
    missed_requirements lists the absent JD keywords. Returns 100 coverage when the
    job carries no ATS keywords (nothing to miss).
    """
    keywords = job.ats_keywords
    if not keywords:
        return RelevanceMetrics(
            relevance_score=100.0,
            must_have_skills_coverage=100.0,
            missed_requirements=[],
            justification="Job description carries no ATS keywords to match.",
        )
    resume_text = render_resume(revised)
    missed = [keyword for keyword in keywords if not keyword_present_in_text(keyword, resume_text)]
    coverage = round(100.0 * (len(keywords) - len(missed)) / len(keywords), 1)
    return RelevanceMetrics(
        relevance_score=coverage,
        must_have_skills_coverage=coverage,
        missed_requirements=missed,
        justification=(
            f"Mechanical keyword coverage: {len(keywords) - len(missed)} of {len(keywords)} "
            f"JD keywords present in the optimized resume."
        ),
    )
