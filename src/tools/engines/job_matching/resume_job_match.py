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
from src.tools.contracts import ReviewResult
from src.tools.engines.document_rendering.resume_text_renderer import render_resume
from src.tools.engines.job_matching.keyword_coverage import analyze_keyword_coverage
from src.tools.engines.job_matching.requirement_matching import match_requirements


def match_resume_to_job(resume: Resume, job: JobDescription) -> ReviewResult:
    """Combine requirement-evidence matching and keyword coverage into one report.

    Expects a validated Resume and JobDescription. Returns a ReviewResult whose
    comments hold the per-requirement findings plus the missing-keyword findings,
    and whose score is the must-have requirement coverage -- the headline metric,
    carried up from the requirements engine (the lead engine in this composite).
    """
    ####################################################
    # STEP 1: RUN THE REQUIREMENT-EVIDENCE MATCHING ENGINE#
    ####################################################
    # This is the semantic side: does the resume actually show evidence
    # for the job's stated requirements?
    requirement_result = match_requirements(resume, job)

    ####################################################
    # STEP 2: RUN THE MECHANICAL KEYWORD COVERAGE CHECK#
    ####################################################
    # This is the literal ATS side: which job keywords appear in the resume text?
    keyword_result = analyze_keyword_coverage(render_resume(resume), job.ats_keywords)

    ####################################################
    # STEP 3: MERGE BOTH RESULTS INTO ONE COMPOSITE REPORT#
    ####################################################
    # We keep the requirement score as the headline score because semantic
    # requirement fit matters more than raw keyword repetition.
    return ReviewResult(
        comments=requirement_result.comments + keyword_result.comments,
        summary="; ".join(s for s in (requirement_result.summary, keyword_result.summary) if s),
        score=requirement_result.score,
    )
