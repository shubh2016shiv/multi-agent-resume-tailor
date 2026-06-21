"""
Review-specific wrapper over the structured-output harness.

Judgment review engines call request_review with a rubric prompt and get back a
ReviewResult whose comments are stamped with the calling engine's id (the model
cannot know which engine asked, so the harness sets it).
"""

from src.tools.contracts import ReviewResult

from .structured_output import request_structured_output


def request_review(engine_id: str, rubric_prompt: str, review_input: str) -> ReviewResult:
    """Ask the model to review review_input against a rubric, as a ReviewResult.

    Args:
        engine_id: Identifier of the calling engine; stamped on every comment.
        rubric_prompt: System prompt defining what the model should look for.
        review_input: The text or payload to review.

    Returns:
        A ReviewResult whose comments are all tagged with engine_id.

    Raises:
        RuntimeError: If the model returns malformed output twice in a row.
    """
    ####################################################
    # STEP 1: REQUEST A STRUCTURED REVIEW RESULT FROM THE LLM#
    ####################################################
    result = request_structured_output(ReviewResult, rubric_prompt, review_input)

    ####################################################
    # STEP 2: STAMP EVERY COMMENT WITH THE CALLING ENGINE ID#
    ####################################################
    # The model cannot reliably know which engine invoked it, so the
    # gateway sets that boundary metadata after validation.
    for comment in result.comments:
        comment.engine_id = engine_id
    return result
