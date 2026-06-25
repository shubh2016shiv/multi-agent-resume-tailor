"""Similarity-based skill-evidence matching for evaluation.

Replaces the former literal + curated-synonym-table matching with embedding
similarity. The gateway returns the score; this module owns the threshold -- the
gate policy that decides which score counts as evidence. Inputs are expected to
be the canonicalized skill forms produced upstream at extraction.
"""

from src.data_models.resume import Skill
from src.tools.llm_gateway import max_similarity

# The minimum embedding cosine similarity at which a required skill counts as
# evidenced. This is a product gate policy, not a number the model chooses.
# Set to 0.50 from a real-data smoke: genuine same-concept variants score 0.54-0.84
# (e.g. "Deep Neural Networks" vs "Deep Learning" = 0.76), while DIFFERENT skills in
# the SAME domain sit at a noise floor of ~0.27-0.45 (e.g. "Kubernetes" vs "Machine
# Learning" = 0.30). A lower cut (0.30) false-matched same-domain skills.
# TODO: Calibrate against a labeled gold set whose DIFF pairs are WITHIN-domain, not
#       just cross-domain (cross-domain is too easy and inflated the earlier estimate).
#       Proposed: lock the value from a >=30 SAME / >=30 within-domain-DIFF set.
#       Deferred because: only a small smoke exists so far.
SIMILARITY_MATCH_THRESHOLD = 0.50


def is_required_skill_evidenced(required_skill: str, candidate_skills: list[str]) -> bool:
    """Return whether a required skill is evidenced by any candidate skill.

    Compares the canonicalized required skill against the candidate skills by
    embedding similarity; True when the best match meets SIMILARITY_MATCH_THRESHOLD.
    Expects canonicalized terms. Returns False when there are no candidate skills.
    """
    ####################################################
    # STEP 1: SCORE THE BEST CANDIDATE MATCH VIA EMBEDDINGS#
    ####################################################
    # We ask the gateway for the single highest similarity between the required
    # skill and any candidate. The gateway owns the model call; we own the cutoff.
    best_match_score = max_similarity(required_skill, candidate_skills)

    ####################################################
    # STEP 2: APPLY THE EVALUATION THRESHOLD (THE GATE POLICY)#
    ####################################################
    # A match must clear the documented threshold. The number is a product
    # decision, calibrated separately, so the model never controls the gate.
    return best_match_score >= SIMILARITY_MATCH_THRESHOLD


def match_term_for_skill(skill: Skill) -> str:
    """Return the skill's canonicalized form for matching, or its raw name if absent.

    Shared by the truthfulness and job-alignment evaluators so both feed the same
    text to the similarity matcher. Embeddings tolerate the raw name when extraction
    set no canonicalized form.
    """
    return skill.canonicalized_skill or skill.skill_name
