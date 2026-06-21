"""
Rewrite-drift detection: did the rewrite stay truthful to the original resume?

Pure judgment engine, no mechanical half. An automated optimizer rewrites bullets
and summaries to sound stronger, and in doing so it can invent a number, add a
skill, inflate a contribution, or quietly drop an achievement. This engine diffs
the ORIGINAL against the REVISED resume and flags that drift in three kinds:
invented claims, exaggerations, and losses.

It is mode-independent: it guards ANY rewrite, job-tailored or not, because an
optimizer can overshoot whether or not a job description is involved.

Honest about its ceiling (issues.md): the model is diffing work a model produced,
so it reliably catches obvious fabrications -- a new credential, a number that was
not in the original -- but not subtle reframings the optimizer already rationalised.
It is a safety net, not a lie detector. Confidence gating reflects this: only
high-confidence drift (a concrete, verifiable change) should block; the rest stays
advisory.
"""

from src.core.prompt_catalog import load_tool_prompt
from src.data_models.resume import Resume
from src.tools.contracts import ReviewResult
from src.tools.engines.document_rendering.resume_text_renderer import render_resume
from src.tools.llm_gateway import request_review

ENGINE_ID = "rewrite_drift_detector"

REWRITE_DRIFT_RUBRIC = load_tool_prompt("truthfulness/rewrite_drift.md")


def detect_rewrite_drift(original: Resume, revised: Resume) -> ReviewResult:
    """Flag where a rewritten resume drifts from the original: inventions, exaggerations, losses.

    Args:
        original: The source-of-truth resume, before optimization.
        revised: The rewritten resume to check against the original.

    Returns:
        A ReviewResult of judgment comments with honest confidence. An empty result
        means the revision is faithful -- or identical, in which case no LLM call is made.
    """
    ####################################################
    # STEP 1: RENDER BOTH RESUMES INTO THE SAME PLAIN-TEXT VIEW#
    ####################################################
    # The model should compare like with like, so both versions go through
    # the same renderer before we build the diff payload.
    original_text = render_resume(original)
    revised_text = render_resume(revised)

    ####################################################
    # STEP 2: SKIP THE LLM CALL IF NOTHING CHANGED AT ALL#
    ####################################################
    if original_text == revised_text:
        return ReviewResult(comments=[], summary="Revision is identical to the original")

    ####################################################
    # STEP 3: BUILD A SIDE-BY-SIDE PAYLOAD FOR THE REVIEW GATEWAY#
    ####################################################
    payload = f"ORIGINAL RESUME:\n{original_text}\n\nREVISED RESUME:\n{revised_text}"

    ####################################################
    # STEP 4: ASK THE REVIEW GATEWAY TO JUDGE THE DRIFT#
    ####################################################
    return request_review(ENGINE_ID, REWRITE_DRIFT_RUBRIC, payload)
