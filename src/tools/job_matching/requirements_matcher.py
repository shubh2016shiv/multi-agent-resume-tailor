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

from src.data_models.job import JobDescription
from src.data_models.resume import Resume
from src.tools.llm_gateway import request_review
from src.tools.review_contract.review_models import ReviewResult
from src.tools.shared.resume_rendering import render_resume

ENGINE_ID = "requirements_matcher"

REQUIREMENTS_RUBRIC = """You compare a candidate's resume against a job's requirements and report
how well the resume EVIDENCES each requirement. Each requirement is tagged with its importance
([must_have], [should_have], [nice_to_have]) and, when stated, a minimum years figure.

For each requirement, decide one of:
- MATCHED: the resume clearly evidences it. Emit NO comment.
- PARTIAL: the resume covers it only partly, or with adjacent (not equivalent) evidence.
- GAP: the resume does not evidence it at all.

Emit one comment per PARTIAL or GAP (never for a MATCHED requirement), with:
- severity: for a GAP, take it from the importance tag -- "must_have" -> "blocker",
  "should_have" -> "major", "nice_to_have" -> "minor". For a PARTIAL (any importance) -> "suggestion".
- quoted_text: the requirement text exactly as given.
- message: matched / partial / gap and why, referencing the evidence (or its absence).
- advice: the specific evidence the candidate should add or strengthen.
- location: the section where that evidence belongs -- "skills" or "experience".

Reliability rules (critical -- this judgment is the easiest in the system to get wrong):
- Claim MATCHED only when the resume CLEARLY evidences the requirement. Do NOT treat adjacent
  technologies as equivalent: Flask experience PARTIALLY covers a FastAPI requirement, it does not
  fully match it; "SQL" does NOT cover a "NoSQL" requirement.
- When a years figure is given, weigh it: a "5+ years" requirement supported by one short role is
  at most PARTIAL, not MATCHED.
- When unsure whether the evidence truly satisfies a requirement, set confidence to "low" and prefer
  PARTIAL over MATCHED. Never assert a confident match you cannot ground in the resume text.
- confidence: "high" only for clear-cut matches or clear-cut gaps; "medium" for likely calls; "low"
  when the call rests on a semantic-equivalence judgment you cannot be sure of.

Finally, set score to the fraction of "must_have" requirements you judged MATCHED (0.0 to 1.0), and
write a one-line summary. Do not invent requirements or evidence.
"""


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
    if not job.requirements:
        return ReviewResult(comments=[], summary="Job has no structured requirements to match")
    payload = (
        f"JOB REQUIREMENTS:\n{_format_requirements(job)}\n\n"
        f"RESUME EVIDENCE:\n{render_resume(resume)}"
    )
    return request_review(ENGINE_ID, REQUIREMENTS_RUBRIC, payload)


def _format_requirements(job: JobDescription) -> str:
    """List each requirement with its importance tag and years_required, when stated."""
    lines = []
    for index, requirement in enumerate(job.requirements, start=1):
        tags = requirement.importance.value
        if requirement.years_required is not None:
            tags += f", {requirement.years_required}+ yrs"
        lines.append(f"{index}. {requirement.requirement} [{tags}]")
    return "\n".join(lines)
