"""
Bullet structure auditing: are experience bullets balanced by recency and length?

Mechanical engine, no LLM. It flags two domain-neutral craft problems: the most
recent (most important) role carrying too few bullets while older roles carry too
many, and individual bullets long enough to read as paragraphs.
"""

from src.data_models.resume import Experience, Resume
from src.tools.review_contract.review_models import (
    Confidence,
    Location,
    ReviewComment,
    ReviewResult,
    Section,
    Severity,
)

ENGINE_ID = "bullet_structure_auditor"

# TODO: Tune the count thresholds against real resumes.
#       Proposed: calibrate on a labelled sample of strong/weak resumes.
#       Deferred: these are unmeasured guesses; the 35-word cap below is grounded
#       (TOOLING_PLAN: "a bullet over ~35 words is a paragraph").
MIN_BULLETS_RECENT_ROLE = 3
MAX_BULLETS_PER_ROLE = 8
MAX_BULLET_WORDS = 35


def audit_bullet_structure(resume: Resume) -> ReviewResult:
    """Flag recency-imbalanced bullet counts and over-long bullets across roles.

    Args:
        resume: The resume to audit; only work_experience is read.

    Returns:
        A ReviewResult. An empty comment list means the bullet structure looks
        balanced. All comments anchor to Section.EXPERIENCE.
    """
    return audit_bullet_structure_for_experiences(resume.work_experience)


def audit_bullet_structure_for_experiences(experiences: list[Experience]) -> ReviewResult:
    """Flag bullet count and length issues across experience entries.

    Args:
        experiences: Experience entries to audit; no other resume fields are read.

    Returns:
        A ReviewResult. An empty comment list means the bullet structure looks
        balanced. All comments anchor to Section.EXPERIENCE.
    """
    if not experiences:
        return ReviewResult(comments=[], summary="No work experience to audit")
    comments: list[ReviewComment] = []
    comments.extend(_check_bullet_counts(experiences))
    comments.extend(_check_bullet_lengths(experiences))
    summary = (
        "Bullet structure looks balanced"
        if not comments
        else f"{len(comments)} bullet structure issue(s)"
    )
    return ReviewResult(comments=comments, summary=summary)


def _check_bullet_counts(experiences: list[Experience]) -> list[ReviewComment]:
    """Flag the most-recent role with too few bullets, or any role with too many."""
    findings = []
    most_recent_role = max(experiences, key=lambda role: role.start_date)
    if len(most_recent_role.achievements) < MIN_BULLETS_RECENT_ROLE:
        findings.append(
            _make_finding(
                message=f"Most recent role has only {len(most_recent_role.achievements)} bullet(s)",
                quoted_text=f"{most_recent_role.job_title} at {most_recent_role.company_name}",
                severity=Severity.MINOR,
                advice=f"Expand your most recent role to at least {MIN_BULLETS_RECENT_ROLE} achievement bullets.",
            )
        )
    for role in experiences:
        if len(role.achievements) > MAX_BULLETS_PER_ROLE:
            findings.append(
                _make_finding(
                    message=f"Role has {len(role.achievements)} bullets, more than {MAX_BULLETS_PER_ROLE}",
                    quoted_text=f"{role.job_title} at {role.company_name}",
                    severity=Severity.SUGGESTION,
                    advice=f"Trim to your strongest {MAX_BULLETS_PER_ROLE} bullets; merge or cut the weakest.",
                )
            )
    return findings


def _check_bullet_lengths(experiences: list[Experience]) -> list[ReviewComment]:
    """Flag each achievement bullet longer than MAX_BULLET_WORDS (a paragraph, not a bullet)."""
    findings = []
    for role in experiences:
        for bullet_index, achievement in enumerate(role.achievements):
            word_count = len(achievement.split())
            if word_count > MAX_BULLET_WORDS:
                findings.append(
                    _make_finding(
                        message=f"Bullet is {word_count} words, over the {MAX_BULLET_WORDS}-word guideline",
                        quoted_text=achievement,
                        severity=Severity.MINOR,
                        advice="Tighten into a single achievement; split or cut filler.",
                        bullet_index=bullet_index,
                    )
                )
    return findings


def _make_finding(
    message: str,
    quoted_text: str,
    severity: Severity,
    advice: str,
    bullet_index: int | None = None,
) -> ReviewComment:
    """Build an EXPERIENCE-section comment for this engine (mechanical, so HIGH confidence).

    bullet_index, when set, is the bullet's position within its role; the role
    itself is named in message/quoted_text since Location has no entry index.
    """
    return ReviewComment(
        engine_id=ENGINE_ID,
        message=message,
        quoted_text=quoted_text,
        location=Location(section=Section.EXPERIENCE, bullet_index=bullet_index),
        severity=severity,
        confidence=Confidence.HIGH,
        advice=advice,
    )
