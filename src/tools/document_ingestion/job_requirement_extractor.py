"""
Job requirement extraction: turn job-description Markdown into a structured JobDescription.

This is an orchestrator pipeline stage, not an agent tool: it makes one bounded,
schema-constrained LLM call to produce a typed JobDescription. It mirrors
resume_section_extractor (Markdown -> Resume) on the job-description side, and runs
ONLY in Mode B (a job description was supplied). Its output is what
requirements_matcher and keyword_coverage_analyzer consume.

Unlike the resume path there is no redaction precondition: a job posting is public
text (company, role, requirements), not candidate PII.
"""

from src.data_models.job import JobDescription
from src.tools.llm_gateway import request_structured_output

JOB_EXTRACTION_PROMPT = """You extract a job posting into a structured JobDescription object.

Rules:
- Use only information present in the job description. Do not invent requirements, keywords,
  seniority, or a company the text does not state.
- job_level: classify the role's seniority as one of entry, junior, mid-level, senior, lead,
  principal, manager, director, executive. Use "unspecified" when the text gives no clear signal.
- requirements: break the posting into individual skills/qualifications. For each, set importance:
  "must_have" for non-negotiables (often phrased "required", "must have"), "should_have" for
  strongly preferred ones, and "nice_to_have" for bonuses ("a plus", "preferred"). Set
  years_required only when the text states a minimum number of years for that specific skill;
  otherwise leave it null.
- ats_keywords: the high-value terms (technologies, tools, certifications, role-specific nouns) a
  resume should mirror to pass automated screening. Draw them from the posting, not a generic list.
- summary: a 1-3 sentence overview of the role.
- full_text: copy the complete original job-description text verbatim.
- If an optional field is absent, leave it null or an empty list. Do not fabricate values.
"""


def extract_job_requirements(job_markdown: str) -> JobDescription:
    """Extract a structured JobDescription from job-description Markdown.

    Mode B only: runs when the user supplied a job description. The result feeds
    requirements_matcher and keyword_coverage_analyzer.

    Args:
        job_markdown: The job posting as text or Markdown (converted upstream).

    Returns:
        A validated JobDescription. full_text holds the original posting verbatim;
        requirements carry per-item importance and any stated years_required.

    Raises:
        RuntimeError: If the model cannot produce a schema-valid JobDescription.
    """
    # TODO: Flag valid-but-empty extraction (no requirements / no ats_keywords). The schema
    #       accepts it silently. Proposed: a follow-up check that surfaces a "couldn't parse
    #       this JD" signal to the user. Deferred because: need to see real failures first.
    return request_structured_output(JobDescription, JOB_EXTRACTION_PROMPT, job_markdown)
