"""Compose the two job-matching engines into one resume-vs-job report.

This is the single computation the gap-analysis stage runs in code rather than via
an LLM tool: it asks the requirements engine how well the resume evidences each job
requirement (judgment), and the keyword engine which JD keywords literally appear
(mechanical), then returns both as one ReviewResult. The orchestration node persists
it as a typed artifact and the formatter renders it into the agent's context, so the
agent reads pre-computed facts instead of reconstructing the resume itself.
"""

from src.data_models.job import JobDescription
from src.data_models.resume import Resume
from src.tools.job_matching.keyword_coverage_analyzer import analyze_keyword_coverage
from src.tools.job_matching.requirements_matcher import match_requirements
from src.tools.review_contract.review_models import ReviewResult
from src.tools.shared.resume_rendering import render_resume


def match_resume_to_job(resume: Resume, job: JobDescription) -> ReviewResult:
    """Combine requirement-evidence matching and keyword coverage into one report.

    Expects a validated Resume and JobDescription. Returns a ReviewResult whose
    comments hold the per-requirement findings plus the missing-keyword findings,
    and whose score is the must-have requirement coverage -- the headline metric,
    carried up from the requirements engine (the lead engine in this composite).
    """
    requirement_result = match_requirements(resume, job)
    keyword_result = analyze_keyword_coverage(render_resume(resume), job.ats_keywords)
    return ReviewResult(
        comments=requirement_result.comments + keyword_result.comments,
        summary="; ".join(s for s in (requirement_result.summary, keyword_result.summary) if s),
        score=requirement_result.score,
    )
