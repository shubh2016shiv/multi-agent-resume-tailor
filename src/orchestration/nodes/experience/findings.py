"""Turn structured review comments into the plain finding strings this stage passes around.

Both checks produce ReviewComments, and both need them as flat text: the truth floor
to report an invented figure, and the repair prompt to tell the writer what to fix.
Lives in its own module because truthfulness.py and rewrite.py both need it, and
rewrite.py already imports truthfulness.py -- putting it in either would mean a cycle.
"""

from src.tools.contracts import ReviewComment


def findings_from_comments(comments: list[ReviewComment]) -> list[str]:
    """Flatten each comment into "what is wrong. what to do about it"."""
    return [f"{comment.message}. {comment.advice}" for comment in comments]
