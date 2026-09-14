"""Stage 3 (parallel with summary and skills): rewrite the work-experience bullets.

Start here. This package is four modules because one file of 700 lines hid how the
parts fit together:

    node.py           the two graph nodes, and the fan-out over roles
    rewrite.py        given one role, decide which version of its bullets ships
    truthfulness.py   the deterministic checks a rewrite must pass to ship at all

WHAT THIS STAGE DOES
    One LLM call per role rewrites that role's bullets 1:1 into specific,
    recruiter-readable accomplishments, using only that role's own evidence. Code
    then decides whether the rewrite may ship, because an LLM asked to make bullets
    impressive will otherwise invent the impressive part.

THE TWO CHECKS, WHICH HAVE DIFFERENT FORCE
    Truthfulness is non-negotiable (truthfulness.py, code only). A rewrite must keep
    the source bullet count, keep bullet identity and order, and introduce no number
    the source role does not state. Only a rewrite that clears this may ship; if
    neither the first attempt nor the repair clears it, the original bullets ship.

    Substance is best-effort (audit_experience_rewrite_quality, an LLM review called
    from rewrite.py). It flags unsupported specificity, ownership inflation, vague
    accomplishments, brochure tone, and JD-keyword decoration. MAJOR findings must be
    fixed before shipping; MINOR ones may ship and are surfaced to the candidate.

HOW A RUN FLOWS THROUGH THESE FILES
    node.optimize_experience                    fan out over roles, merge back
      node._run_single_experience_optimization  one role, end to end
        rewrite.request_role_rewrite_proposal   the LLM call
        rewrite._check_proposal                 score it on both checks
          truthfulness.collect_truthfulness_findings
        rewrite.decide_role_rewrite_outcome     which version ships; repair once
        build_bullet_clarifications             -> src/hitl/professional_experience/

    node.await_candidate_clarifications         a separate graph node: the pause
                                                boundary, reached after all of Stage 3

AT MOST ONE REPAIR
    A failing proposal earns exactly one more LLM attempt, never a loop. If the repair
    comes back worse than the first attempt, the first attempt ships. This is the same
    no-retry-loops rule the whole pipeline follows.

WHAT THE LLM CANNOT CHANGE
    Only bullet text. Role metadata (company, title, dates, description, skills_used)
    is always rebuilt from the source object, whatever the proposal contains.

THE HUMAN-IN-THE-LOOP PART
    Bullets that stayed truthful but thin produce a question for the candidate, which
    pauses the run until they answer. The review that decides this, and the
    pause/resume file handling, live in src/hitl/professional_experience/; this package
    only calls into it and owns the pause boundary before ATS assembly.
"""

from src.orchestration.nodes.experience.node import (
    await_candidate_clarifications,
    optimize_experience,
)

__all__ = ["await_candidate_clarifications", "optimize_experience"]
