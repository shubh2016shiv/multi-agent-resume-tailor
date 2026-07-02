"""
ST-08: Live smoke test — professional summary quality gate (real LLM, no mocks)

Root cause being tested: write_professional_summary_task declares hard constraints
(a banned-phrase list, a banned "[title] with [x] years of experience" opener) that
the audit_summary tool could already detect, but nothing in the pipeline enforced
its findings -- the tool call was optional for the agent, and its only sink
(writing_notes) is discarded before assembly (see ats_optimization_formatter.
choose_summary_text, which forwards only draft.content). A draft could violate the
task's own hard constraints and still ship.

The fix has two parts, both exercised here with REAL LLM calls (no mocked
ReviewResult, no synthetic findings):

1. src/config/tool_prompts/resume_diagnostics/summary_quality.md -- recalibrated so
   the two hard-constraint categories (generic boilerplate / banned opener formula)
   carry MAJOR severity; the two subjective categories (missing value proposition,
   brochure tone) stay MINOR.
2. src/orchestration/nodes/summary.enforce_summary_quality_gate -- calls the real
   audit_summary_text() on the recommended draft and raises ValueError on any
   MAJOR+ finding, blocking the draft before it reaches assembly.

Case A: the literal regression text, run through the CURRENT (fixed) rubric via the
        real gate function -- must raise.
Case B: the SAME regression text, run through the OLD rubric (all severities hard-
        coded to "minor", reconstructed verbatim below since it predates this fix)
        vs. the CURRENT rubric loaded from disk -- both are real LLM calls to the
        same auditor, only the rubric text differs. Demonstrates the severity
        actually flips from minor to major.
Case C: a clean, evidence-based summary with no banned phrases or banned opener --
        must NOT raise, so the gate isn't a blunt instrument that blocks everything.

Run:
    cd /home/shubham_singh/Projects/resume_tailor
    uv run python -m regression_fix.smoke_test.st08_summary_quality_gate_live
"""

import sys

sys.path.insert(0, "/home/shubham_singh/Projects/resume_tailor")

from src.agents.professional_summary.models import ProfessionalSummary, SummaryDraft
from src.core.prompt_catalog import load_tool_prompt
from src.orchestration.nodes.summary import enforce_summary_quality_gate
from src.tools.llm_gateway import request_review

SEPARATOR = "=" * 60

# The exact text from the reported regression: opens with the banned
# "[title] with [x] years of experience" formula and uses "Proven track record".
REGRESSION_SUMMARY_TEXT = (
    "Senior Machine Learning Engineer with 9+ years spanning AI/ML production "
    "deployments and software engineering. Proven track record owning production "
    "ML systems across the full lifecycle, from data pipelines to model serving. "
    "Experienced in MLOps using MLflow, Docker, and serverless inference on AWS Lambda."
)

# A clean draft: evidence-based thesis opener, no banned phrases, no formulaic
# "[title] with [x] years" opener, and inside the 80-110 word range (the length
# floor is MAJOR severity, so an under-length draft now correctly fails the gate).
CLEAN_SUMMARY_TEXT = (
    "Machine learning engineer trusted to take production AI systems from "
    "prototype to on-call ownership. Built and operated retrieval-augmented "
    "generation pipelines backed by vector search for a healthcare document "
    "workflow, cutting p99 response latency from 20 seconds to under 5 under "
    "concurrent load. Re-architected a monolithic validation script generator "
    "into a multi-agent pipeline, reducing critical failures by 95 percent. "
    "Brings MLOps discipline -- MLflow experiment tracking, containerized "
    "serving, and serverless inference on AWS Lambda -- so deployed models stay "
    "observable, reproducible, and debuggable long after launch."
)

# The rubric exactly as it existed before this fix: 4 categories, but every
# category hardcoded to severity "minor" regardless of whether it maps to one of
# write_professional_summary_task's hard constraints. Reconstructed verbatim from
# the pre-fix file (this codebase does not keep an old-version file around per the
# "no dead config left in production files" lesson from a prior review, so the
# comparison text lives here, in the diagnostic that needs it).
OLD_RUBRIC_ALL_MINOR = """You review a resume's professional summary for judgment-based prose quality.

Your job is to catch the patterns that make a summary sound generic, brochure-like,
or obviously AI-written even when it is factually safe.

Flag a real issue when you see any of these:

1. Generic boilerplate
   The summary says almost nothing specific about the candidate.
   Examples: "results-oriented professional", "proven track record", "dynamic professional",
   "passionate engineer", "highly motivated", "detail-oriented".

2. Weak or formulaic opener
   The summary opens with a stock role-and-years formula instead of a thesis.
   Example pattern: "[job title] with [x] years of experience..."

3. Missing value proposition
   The summary does not make clear what the candidate is actually trusted to do,
   what kind of problems they solve, or what domain-specific strength they bring.

4. Brochure tone
   The wording sounds promotional, padded, or banner-copy-like instead of terse,
   specific, and senior.

Read the summary as a recruiter would. Prefer high-signal, specific writing over
keyword performance. The summary does not need a fixed keyword count; it needs a
clear thesis, believable specificity, and credible professional tone.

Return one review comment per real issue, with:
- severity: "minor"
- confidence: "medium"
- message: what is weak
- quoted_text: the exact phrase, or the whole summary if the problem is global
- advice: how to make it more thesis-led, specific, and credible
- location: section "summary"

If the summary is specific, thesis-led, and conveys clear value in a credible tone,
return no comments.
Do not comment on length or pronouns; those are checked separately.
"""


def _summary_with_draft(content: str) -> ProfessionalSummary:
    draft = SummaryDraft(
        version_name="Test Draft",
        strategy_used="N/A -- smoke test fixture.",
        evidence_used="N/A -- smoke test fixture.",
        content=content,
        score=80,
    )
    return ProfessionalSummary(drafts=[draft], recommended_version="Test Draft")


def main() -> int:
    print(SEPARATOR)
    print("ST-08: Professional summary quality gate (live LLM, no mocks)")
    print(SEPARATOR)

    failures = []

    # --- Case A: the real gate, on the real regression text, current rubric ---
    print("\n[Case A] Regression text through the REAL gate (current, fixed rubric)")
    print(f"  text: {REGRESSION_SUMMARY_TEXT[:80]}...")
    try:
        enforce_summary_quality_gate(_summary_with_draft(REGRESSION_SUMMARY_TEXT))
        failures.append("Case A: gate did not raise on the regression text")
        print("  [FAIL] gate did not raise -- the reported bug would still ship")
    except ValueError as error:
        print(f"  [PASS] gate raised: {error}")

    # --- Case B: same text, same auditor, old rubric vs. current rubric ---
    print("\n[Case B] Same regression text, same LLM call -- old rubric vs. current rubric")
    current_rubric = load_tool_prompt("resume_diagnostics/summary_quality.md")

    old_result = request_review(
        "summary_quality_auditor", OLD_RUBRIC_ALL_MINOR, REGRESSION_SUMMARY_TEXT
    )
    new_result = request_review(
        "summary_quality_auditor", current_rubric, REGRESSION_SUMMARY_TEXT
    )

    print("  BEFORE (old rubric):")
    for comment in old_result.comments:
        print(f"    [{comment.severity.value.upper()}] {comment.message}")
    old_has_major = any(c.severity.value == "major" for c in old_result.comments)

    print("  AFTER (current rubric):")
    for comment in new_result.comments:
        print(f"    [{comment.severity.value.upper()}] {comment.message}")
    new_has_major = any(c.severity.value == "major" for c in new_result.comments)

    if old_has_major:
        failures.append("Case B: old rubric already produced a MAJOR finding (unexpected)")
        print("  [FAIL] old rubric unexpectedly produced a MAJOR finding")
    elif not new_has_major:
        failures.append("Case B: current rubric did not produce a MAJOR finding")
        print("  [FAIL] current rubric did not escalate any finding to MAJOR")
    else:
        print("  [PASS] severity flipped from minor-only to MAJOR after the rubric fix")

    # --- Case C: a clean summary must not be blocked ---
    print("\n[Case C] Clean, evidence-based summary through the REAL gate")
    print(f"  text: {CLEAN_SUMMARY_TEXT[:80]}...")
    try:
        enforce_summary_quality_gate(_summary_with_draft(CLEAN_SUMMARY_TEXT))
        print("  [PASS] gate did not raise -- a clean draft is not blocked")
    except ValueError as error:
        failures.append(f"Case C: gate wrongly raised on a clean draft: {error}")
        print(f"  [FAIL] gate raised on a clean draft: {error}")

    print(f"\n{SEPARATOR}")
    if failures:
        print(f"RESULT: {len(failures)} FAILURE(S)")
        for failure in failures:
            print(f"  - {failure}")
    else:
        print("RESULT: PASS -- the regression is caught, the old rubric could not")
        print("catch it, and a clean draft is not blocked.")
    print(SEPARATOR)
    return len(failures)


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
