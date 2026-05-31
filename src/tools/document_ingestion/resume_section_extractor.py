"""
Resume extraction: turn privacy-redacted resume Markdown into a structured Resume.

This is an orchestrator pipeline stage, not an agent tool: it makes one bounded,
schema-constrained LLM call to produce a typed Resume. It is invoked directly by
the orchestrator (like convert and redact), so the extraction LLM call is paid
for exactly once, with no agent reasoning layered on top.
"""

from src.data_models.resume import Resume
from src.tools.llm_gateway import request_structured_output

# Prompt duplicated from tasks.yaml (extract_resume_content_task) for now.
# TODO: Remove the prompt from tasks.yaml once the Resume Extractor agent is
#       migrated to call this engine instead of doing extraction inline.
RESUME_EXTRACTION_PROMPT = """You extract a candidate's resume into a structured Resume object.

Rules:
- Use only information present in the resume text. Do not invent or infer data that is not there.
- The text is privacy-redacted: tokens like [PERSON_1], [EMAIL_ADDRESS_1], [PHONE_NUMBER_1] are
  placeholders. Copy them verbatim into the matching fields (full_name, email, phone_number).
- Dates: convert to ISO format (YYYY-MM-DD). When only a month and year are given, use the first
  day of the month. For an ongoing role, set is_current_position to true and leave end_date null.
- For each work experience, capture the role's bullet points as achievements and the free-text
  role summary as description.
- Education graduation_year is the year the qualification was (or will be) completed.
- If an optional field is absent, leave it null or an empty list. Do not fabricate values.
"""


def extract_resume(redacted_markdown: str) -> Resume:
    """Extract a structured Resume from privacy-redacted resume Markdown.

    Args:
        redacted_markdown: Resume text with PII already masked by redact_pii.
            Precondition: PII must be redacted before this is called.

    Returns:
        A validated Resume. Personal fields hold redaction placeholders
        (e.g. full_name == "[PERSON_1]") for the orchestrator to rehydrate.

    Raises:
        RuntimeError: If the model cannot produce a schema-valid Resume.
    """
    # TODO: Tolerate unparseable/absent dates. A resume with no dates fails the
    #       required Experience.start_date and raises. Proposed: a date-tolerant
    #       fallback. Deferred: need to see real failures first.
    # TODO: Flag valid-but-empty extraction (e.g. no work_experience). The schema
    #       accepts it silently. Proposed: a follow-up check. Deferred until seen.
    return request_structured_output(Resume, RESUME_EXTRACTION_PROMPT, redacted_markdown)
