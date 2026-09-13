"""Contracts for the code-owned guarantees in the skills optimization node.

Two of this node's steps exist because the LLM cannot be trusted with them, so
they are the ones worth pinning before the node is simplified:

  * STEP 7 re-adds any original skill the optimizer dropped, and must keep
    removed_skills honest about what actually shipped.
  * The removal gate acts only on findings that are both serious AND
    high-confidence, so a thin evidence corpus cannot strip truthful skills.
"""

from src.data_models.resume import OptimizedSkillsSection, Resume, Skill
from src.orchestration.nodes.skills import (
    flagged_skill_names,
    is_confident_unsupported,
    preserve_original_skills,
    skills_audit_needs_rewrite,
)
from src.tools.contracts import (
    Confidence,
    Location,
    ReviewComment,
    ReviewResult,
    Section,
    Severity,
)


def _skill(name: str) -> Skill:
    return Skill(
        skill_name=name,
        canonicalized_skill=None,
        category=None,
        proficiency_level=None,
        years_of_experience=None,
        justification=None,
        confidence_score=None,
    )


def _resume_with_skills(skills: list[Skill]) -> Resume:
    """preserve_original_skills reads only resume.skills; the rest is filler."""
    return Resume(
        full_name="Jane Doe",
        email="jane.doe@example.com",
        phone_number=None,
        location=None,
        website_or_portfolio=None,
        professional_summary="",
        work_experience=[],
        education=[],
        skills=skills,
        certifications=[],
        languages=[],
    )


def _section(
    skill_names: list[str],
    *,
    removed_skills: list[str] | None = None,
) -> OptimizedSkillsSection:
    return OptimizedSkillsSection(
        optimized_skills=[_skill(name) for name in skill_names],
        removed_skills=removed_skills or [],
    )


def _comment(
    skill_name: str,
    *,
    severity: Severity = Severity.MAJOR,
    confidence: Confidence = Confidence.HIGH,
) -> ReviewComment:
    return ReviewComment(
        engine_id="skills_evidence",
        message=f"{skill_name} is unsupported",
        quoted_text=skill_name,
        location=Location(
            section=Section.SKILLS,
            bullet_index=None,
            item_id=None,
            character_span=None,
        ),
        severity=severity,
        confidence=confidence,
        advice="Remove it or add evidence",
        proposed_rewrite=None,
    )


# --- the removal gate --------------------------------------------------------


def test_only_serious_high_confidence_findings_are_acted_on() -> None:
    """The candidate listed the skill, so a hunch is never enough to remove it."""
    assert is_confident_unsupported(_comment("Rust")) is True
    assert is_confident_unsupported(_comment("Rust", severity=Severity.BLOCKER)) is True
    assert is_confident_unsupported(_comment("Rust", severity=Severity.MINOR)) is False
    assert is_confident_unsupported(_comment("Rust", confidence=Confidence.MEDIUM)) is False
    assert is_confident_unsupported(_comment("Rust", confidence=Confidence.LOW)) is False


def test_a_rewrite_is_triggered_only_by_a_confident_finding() -> None:
    advisory_only = ReviewResult(
        comments=[_comment("Rust", confidence=Confidence.MEDIUM)], summary="", score=None
    )
    confident = ReviewResult(comments=[_comment("Rust")], summary="", score=None)

    assert skills_audit_needs_rewrite(ReviewResult(comments=[], summary="", score=None)) is False
    assert skills_audit_needs_rewrite(advisory_only) is False
    assert skills_audit_needs_rewrite(confident) is True


def test_only_confidently_flagged_skill_names_are_listed_for_removal() -> None:
    audit = ReviewResult(
        comments=[
            _comment("Rust"),
            _comment("Kubernetes", confidence=Confidence.LOW),
            _comment("Haskell", severity=Severity.SUGGESTION),
        ],
        summary="",
        score=None,
    )

    assert flagged_skill_names(audit) == ["Rust"]


# --- STEP 7: original skills are facts --------------------------------------


def test_dropped_original_skills_are_re_added_after_the_optimizer_ordering() -> None:
    """The optimizer may reorder and categorize, but never delete a listed skill."""
    original = _resume_with_skills([_skill("Python"), _skill("SQL"), _skill("Docker")])
    optimized = _section(["Python", "SQL"])

    preserved = preserve_original_skills(optimized, original)

    names = [skill.skill_name for skill in preserved.optimized_skills]
    assert names == ["Python", "SQL", "Docker"]


def test_a_re_added_skill_is_removed_from_the_removed_skills_record() -> None:
    """removed_skills must not claim a skill is gone when it is right there."""
    original = _resume_with_skills([_skill("Python"), _skill("Docker")])
    optimized = _section(["Python"], removed_skills=["Docker"])

    preserved = preserve_original_skills(optimized, original)

    assert [skill.skill_name for skill in preserved.optimized_skills] == ["Python", "Docker"]
    assert preserved.removed_skills == []


def test_skill_matching_ignores_case() -> None:
    """A optimizer that re-cased a skill has not dropped it."""
    original = _resume_with_skills([_skill("Python"), _skill("SQL")])
    optimized = _section(["python", "sql"])

    preserved = preserve_original_skills(optimized, original)

    assert [skill.skill_name for skill in preserved.optimized_skills] == ["python", "sql"]


def test_a_section_that_kept_every_skill_is_returned_unchanged() -> None:
    original = _resume_with_skills([_skill("Python")])
    optimized = _section(["Python"])

    assert preserve_original_skills(optimized, original) is optimized
