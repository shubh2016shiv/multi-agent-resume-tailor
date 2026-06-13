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

from src.data_models.resume import Resume
from src.tools.llm_gateway import load_tool_prompt, request_review
from src.tools.review_contract.review_models import ReviewResult
from src.tools.shared.resume_rendering import render_resume

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
    original_text = render_resume(original)
    revised_text = render_resume(revised)
    if original_text == revised_text:
        return ReviewResult(comments=[], summary="Revision is identical to the original")
    payload = f"ORIGINAL RESUME:\n{original_text}\n\nREVISED RESUME:\n{revised_text}"
    return request_review(ENGINE_ID, REWRITE_DRIFT_RUBRIC, payload)
