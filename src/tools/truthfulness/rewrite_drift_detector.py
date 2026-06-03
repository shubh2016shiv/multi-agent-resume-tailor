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
from src.tools.llm_gateway import request_review
from src.tools.review_contract.review_models import ReviewResult

from src.tools.shared.resume_rendering import render_resume

ENGINE_ID = "rewrite_drift_detector"

REWRITE_DRIFT_RUBRIC = """You compare an ORIGINAL resume with a REVISED version of the same
resume and flag where the rewrite stopped being truthful to the original.

You are given two labelled versions of one person's resume. The revision was produced by an
automated optimizer, which can overstate or invent things while trying to make the resume
stronger. Judge only against the ORIGINAL -- it is the source of truth. Do NOT reward the
revision for sounding more impressive.

Flag three kinds of drift:

1. Invented claims (most serious): content in the REVISED resume with no basis in the ORIGINAL
   -- a new metric or number, a skill, a role, a degree, or a certification the original never
   supports. A fabricated credential or invented number is a blocker.

2. Exaggeration: the REVISED resume overstates what the ORIGINAL said -- e.g. "helped on three
   projects" rewritten as "led three cross-functional initiatives", or a vague contribution
   rewritten with a specific number the original did not contain.

3. Loss: important content present in the ORIGINAL that the REVISED resume dropped -- a key
   achievement, a whole role, or a quantified result. This is a quality regression, not a lie,
   so it is lower severity.

Return one comment per real drift, with:
- severity: "blocker" for a fabricated number, credential, role, or degree; "major" for other
  invented claims and clear exaggerations; "minor" for dropped (lost) content
- confidence: "high" when the change is concrete and verifiable against the original (a number
  or named entity plainly new or plainly removed); "medium" or "low" when the drift is a
  matter of subjective phrasing you cannot be sure overstates the source
- message: what drifted, naming both versions ("original said X, revision says Y")
- quoted_text: the exact REVISED text (for a loss, the exact ORIGINAL text that was dropped)
- advice: restore the claim to what the original supports, or re-add the dropped content
- location: the section the change is in (summary, experience, skills, education, ...)

Only compare the two versions given. Do not invent drift. If the revision stays faithful to
the original, return no comments.
"""


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
