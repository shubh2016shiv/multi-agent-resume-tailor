"""
Review-specific wrapper over the structured-output harness.

Judgment review engines call request_review with a rubric prompt and get back a
ReviewResult whose comments are stamped with the calling engine's id (the model
cannot know which engine asked, so the harness sets it).
"""

from src.tools.review_contract.review_models import ReviewResult

from .structured_llm import request_structured_output


def request_review(engine_id: str, rubric_prompt: str, resume_text: str) -> ReviewResult:
    """Ask the model to review resume_text against a rubric, as a ReviewResult.

    Args:
        engine_id: Identifier of the calling engine; stamped on every comment.
        rubric_prompt: System prompt defining what the model should look for.
        resume_text: The resume content to review.

    Returns:
        A ReviewResult whose comments are all tagged with engine_id.

    Raises:
        RuntimeError: If the model returns malformed output twice in a row.
    """
    result = request_structured_output(ReviewResult, rubric_prompt, resume_text)
    return _stamp_engine_id(result, engine_id)


def _stamp_engine_id(result: ReviewResult, engine_id: str) -> ReviewResult:
    """Tag every comment with the calling engine's id (the model cannot know it)."""
    for comment in result.comments:
        comment.engine_id = engine_id
    return result
