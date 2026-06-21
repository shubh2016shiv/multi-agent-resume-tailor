"""
Requirements matching: how much of THIS job does the resume actually evidence?

Judgment engine, one LLM call. This is the core Mode B (job-tailoring) question and,
per issues.md, the single riskiest engine in the system: semantic equivalence is
exactly where models confidently hallucinate ("SQL covers NoSQL" -- it does not).
The mechanical keyword side stays in keyword_coverage_analyzer; this engine owns the
semantic side, and its whole design is built around the model's known failure mode.

The mitigation is confidence gating, not pretended precision. The rubric forces the
model to claim a match only when the resume CLEARLY evidences a requirement, to prefer
PARTIAL over a confident MATCH when unsure, and to set confidence honestly (~80%
ceiling acknowledged). Downstream, only high-confidence findings should gate the Gap
Analysis agent; medium/low are advisory.

It takes a JobDescription as input (produced by the Job Analyzer agent), so it is
buildable and testable on its own, ahead of any job_requirement_extractor wiring.
"""

from src.core.prompt_catalog import load_tool_prompt
from src.data_models.job import JobDescription
from src.data_models.resume import Resume
from src.tools.contracts import ReviewResult
from src.tools.engines.document_rendering.resume_text_renderer import render_resume
from src.tools.llm_gateway import request_review

ENGINE_ID = "requirements_matcher"

REQUIREMENTS_RUBRIC = load_tool_prompt("job_matching/requirements_matcher.md")


def match_requirements(resume: Resume, job: JobDescription) -> ReviewResult:
    """Classify each job requirement as matched / partial / gap against the resume's evidence.

    Args:
        resume: The candidate's resume, used as the evidence corpus.
        job: The target job; its structured requirements (with importance and years) are matched.

    Returns:
        A ReviewResult with one comment per partial or gap (matched requirements get none),
        severity set from each requirement's importance, and score = fraction of must-have
        requirements matched. An empty result (no LLM call) means the job had no requirements.
    """
    # TODO: score is model-computed and can be inflated while a contradicting gap is missed
    #       (the ~80% semantic ceiling). Proposed: callers sanity-check score against the emitted
    #       comments -- e.g. score == 1.0 with a BLOCKER comment present is incoherent and should be
    #       distrusted. Deferred because: this is a caller concern, not engine logic.
    ####################################################
    # STEP 1: EXIT EARLY IF THE JOB HAS NO STRUCTURED REQUIREMENTS#
    ####################################################
    if not job.requirements:
        return ReviewResult(comments=[], summary="Job has no structured requirements to match")

    ####################################################
    # STEP 2: BUILD ONE PROMPT PAYLOAD WITH JOB REQUIREMENTS AND RESUME EVIDENCE#
    ####################################################
    # The model should judge against explicit evidence, not a raw resume blob
    # with no structure, so we format both sides clearly.
    payload = (
        f"JOB REQUIREMENTS:\n{_format_requirements(job)}\n\n"
        f"RESUME EVIDENCE:\n{render_resume(resume)}"
    )

    ####################################################
    # STEP 3: SEND THE PAYLOAD TO THE REVIEW GATEWAY#
    ####################################################
    return request_review(ENGINE_ID, REQUIREMENTS_RUBRIC, payload)


def _format_requirements(job: JobDescription) -> str:
    """List each requirement with its importance tag and years_required, when stated."""
    ####################################################
    # STEP 1: RENDER EACH REQUIREMENT AS A NUMBERED, TAGGED LINE#
    ####################################################
    # The tags give the model the importance and years context
    # right beside the requirement text.
    lines = []
    for index, requirement in enumerate(job.requirements, start=1):
        tags = requirement.importance.value
        if requirement.years_required is not None:
            tags += f", {requirement.years_required}+ yrs"
        lines.append(f"{index}. {requirement.requirement} [{tags}]")

    ####################################################
    # STEP 2: JOIN THE LINES INTO ONE READABLE REQUIREMENT LIST#
    ####################################################
    return "\n".join(lines)
