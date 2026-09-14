"""Apply candidate answers to professional-experience bullet rewrites.

WHAT THIS MODULE IS
-------------------
The RESUME-SIDE glue. After a paused run comes back to life with the candidate's
answers loaded into pipeline state, these two functions get those answers to the
right role and fold them in as evidence so the next rewrite pass can use them.

WHERE IT SITS IN THE LIFECYCLE
------------------------------
    resume_paused_run()                              [orchestration/runner.py]
      loads answered clarifications into state["clarification_answers"]
      |
      v
    optimize_experience() runs AGAIN                 [orchestration/nodes/experience/node.py]
      for each role:
        answers_for_role(role, all_answers)          <- THIS MODULE: filter
        experience_with_candidate_answers(role, ...)  <- THIS MODULE: fold in
      then the rewriter + reviewers see the augmented role and treat the
      candidate's facts as source truth, not as invention to be flagged.

NEITHER FUNCTION HAS SIDE EFFECTS. Both return new objects; the input Experience
is never mutated (Pydantic model_copy).
"""

from src.data_models.resume import Experience
from src.hitl.professional_experience.models import (
    ExperienceBulletClarification,
    build_experience_bullet_id,
)


# =============================================================================
# answers_for_role()  --  ROUTING: which of these answers belong to this role?
# -----------------------------------------------------------------------------
# A resumed run carries EVERY answered clarification for the whole candidate in
# state["clarification_answers"] -- possibly spanning several jobs. Each role's
# rewrite must see only its own. This function is that filter.
#
# CALLED BY : _resume_with_candidate_answers_as_source()  [experience/node.py]
#             _optimize_experience_entries()              [experience/node.py]
# CALLS     : build_experience_bullet_id()  [models.py] -- the stable id scheme
# RETURNS   : the subset of clarification_answers that (a) has a real answer and
#             (b) belongs to THIS experience.
# =============================================================================
def answers_for_role(
    experience: Experience,
    clarification_answers: list[ExperienceBulletClarification],  # ALL answers, all roles
) -> list[ExperienceBulletClarification]:
    """Return answered clarifications whose bullet_id belongs to this exact role."""
    # ---- STEP 1: compute the set of bullet_ids this role could have produced.
    # build_experience_bullet_id(experience, i) is deterministic -- the SAME
    # scheme the trigger used when it created the questions -- so ids computed
    # now match ids stored on the sheet days ago, as long as the resume's roles
    # and bullet counts have not changed.
    bullet_ids_for_role = {
        build_experience_bullet_id(experience, bullet_index)
        for bullet_index, _ in enumerate(experience.achievements)
    }

    # ---- STEP 2: keep answers that are (a) actually answered and (b) ours.
    # A blank answer is dropped here so a resume never wastes an LLM pass on it.
    return [
        clarification
        for clarification in clarification_answers
        if clarification.answer.strip() and clarification.bullet_id in bullet_ids_for_role
    ]


# =============================================================================
# experience_with_candidate_answers()  --  FOLD-IN: answers become role evidence
# -----------------------------------------------------------------------------
# Appends the candidate's answers to the role's `description` field, tagged as
# first-class facts. Downstream, the truthfulness checker reads the description,
# so a metric the candidate supplied ("cut p95 latency 40%") is now treated as
# source evidence instead of being flagged as an unsupported invention.
#
# CALLED BY : _resume_with_candidate_answers_as_source()  [experience/node.py]
#             _run_single_experience_optimization()       [experience/node.py]
# CALLS     : nothing (pure string assembly + Experience.model_copy)
# RETURNS   : a NEW Experience with an augmented description. Input unchanged.
# =============================================================================
def experience_with_candidate_answers(
    experience: Experience,
    answers: list[ExperienceBulletClarification],  # already filtered to this role
) -> Experience:
    """Add bullet-level candidate answers to the role evidence for the next rewrite."""
    # ---- STEP 1: nothing to fold in -> hand back the role untouched --------
    if not answers:
        return experience

    # ---- STEP 2: one line per answer, carrying the bullet it belongs to ----
    answer_lines = "\n".join(
        f'- BULLET_ID {answer.bullet_id} | About "{answer.bullet}": {answer.answer}'
        for answer in answers
    )

    # ---- STEP 3: splice those lines onto the end of the role description ----
    # The "first-class facts" wording matters: it tells the rewriter and the
    # reviewers to trust these as if they were on the original resume.
    augmented_description = (
        f"{experience.description}\n\n"
        f"Candidate-provided clarifications (their own words, first-class facts):\n"
        f"{answer_lines}"
    )

    # ---- STEP 4: return a copy -- never mutate the caller's Experience -----
    return experience.model_copy(update={"description": augmented_description})
