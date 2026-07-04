"""Apply candidate answers to professional-experience bullet rewrites."""

from src.data_models.resume import Experience
from src.hitl.professional_experience.models import (
    ExperienceBulletClarification,
    build_experience_bullet_id,
)


def answers_for_role(
    experience: Experience,
    clarification_answers: list[ExperienceBulletClarification],
) -> list[ExperienceBulletClarification]:
    """Return answered clarifications whose bullet_id belongs to this exact role."""
    bullet_ids_for_role = {
        build_experience_bullet_id(experience, bullet_index)
        for bullet_index, _ in enumerate(experience.achievements)
    }
    return [
        clarification
        for clarification in clarification_answers
        if clarification.answer.strip() and clarification.bullet_id in bullet_ids_for_role
    ]


def experience_with_candidate_answers(
    experience: Experience,
    answers: list[ExperienceBulletClarification],
) -> Experience:
    """Add bullet-level candidate answers to the role evidence for the next rewrite."""
    if not answers:
        return experience
    answer_lines = "\n".join(
        (
            f'- BULLET_ID {answer.bullet_id} | About "{answer.bullet}": {answer.answer}'
        )
        for answer in answers
    )
    augmented_description = (
        f"{experience.description}\n\n"
        f"Candidate-provided clarifications (their own words, first-class facts):\n"
        f"{answer_lines}"
    )
    return experience.model_copy(update={"description": augmented_description})
