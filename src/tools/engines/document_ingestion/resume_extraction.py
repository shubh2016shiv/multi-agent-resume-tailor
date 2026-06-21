"""
Resume extraction: turn privacy-redacted resume Markdown into a structured Resume.

This is an orchestrator pipeline stage, not an agent tool: it makes one bounded,
schema-constrained LLM call to produce a typed Resume. It is invoked directly by
the orchestrator (like convert and redact), so the extraction LLM call is paid
for exactly once, with no agent reasoning layered on top.
"""

import re

from src.core.prompt_catalog import load_tool_prompt
from src.data_models.resume import Experience, Resume
from src.tools.llm_gateway import request_structured_output

RESUME_EXTRACTION_PROMPT = load_tool_prompt("document_ingestion/resume_extraction.md")


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
    ####################################################
    # STEP 1: ASK THE MODEL TO TURN REDACTED TEXT INTO A TYPED RESUME#
    ####################################################
    # The prompt and schema do the heavy lifting here. This stage expects
    # the incoming text to already have PII masked.
    resume = request_structured_output(Resume, RESUME_EXTRACTION_PROMPT, redacted_markdown)

    ####################################################
    # STEP 2: ADD DETERMINISTIC EXPERIENCE IDS OWNED BY OUR CODE#
    ####################################################
    # We do this after extraction so downstream steps can refer to each
    # experience entry with a stable identifier.
    return assign_experience_ids(resume)


def assign_experience_ids(resume: Resume) -> Resume:
    """Assign deterministic IDs to resume experience entries.

    Expects a validated Resume, with or without existing experience IDs.
    Returns a Resume copy whose work_experience entries have code-owned IDs.
    """
    ####################################################
    # STEP 1: REBUILD EACH EXPERIENCE ENTRY WITH A STABLE ID#
    ####################################################
    # The ID comes from resume order plus durable role fields so it is
    # readable and repeatable within a run.
    experiences = [
        experience.model_copy(update={"experience_id": build_experience_id(index, experience)})
        for index, experience in enumerate(resume.work_experience, start=1)
    ]

    ####################################################
    # STEP 2: RETURN A NEW RESUME OBJECT WITH THE UPDATED EXPERIENCES#
    ####################################################
    return resume.model_copy(update={"work_experience": experiences})


def build_experience_id(index: int, experience: Experience) -> str:
    """Build a human-readable ID from resume order and stable role fields.

    Expects an Experience with company, title, and start_date populated.
    Returns an ID suitable for per-run role correlation.
    """
    ####################################################
    # STEP 1: NORMALIZE THE COMPANY AND TITLE INTO SAFE ID PARTS#
    ####################################################
    company_slug = slugify_experience_id_part(experience.company_name)
    title_slug = slugify_experience_id_part(experience.job_title)

    ####################################################
    # STEP 2: ADD THE START DATE SO SIMILAR ROLES STAY DISTINCT#
    ####################################################
    start_date = experience.start_date.isoformat().replace("-", "_")

    ####################################################
    # STEP 3: BUILD ONE READABLE ID STRING#
    ####################################################
    return f"exp_{index:03d}_{company_slug}_{title_slug}_{start_date}"


def slugify_experience_id_part(text: str) -> str:
    """Normalize one text field for inclusion in an experience ID.

    Expects a company or title string.
    Returns lowercase ASCII words joined by underscores, or 'unknown'.
    """
    ####################################################
    # STEP 1: REMOVE NON-ALPHANUMERIC CHARACTERS AND NORMALIZE SEPARATORS#
    ####################################################
    # This keeps the ID readable and safe to use in logs, maps, and comparisons.
    normalized_text = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")

    ####################################################
    # STEP 2: FALL BACK TO A SAFE PLACEHOLDER IF NOTHING REMAINS#
    ####################################################
    return normalized_text or "unknown"
