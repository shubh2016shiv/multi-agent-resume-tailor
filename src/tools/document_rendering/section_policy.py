"""
Resume layout policy -- domain-neutral ergonomics, not professional content judgment.

Where each section goes depends on the candidate's stage: an experienced hire leads
with Experience; a new graduate floats Education and Skills up to offset thin
experience; a student/intern leads with Education. These are presentation
conventions (TOOLING_PLAN section 1/9 explicitly permits freezing ergonomics like
this), NOT field-specific content knowledge -- the renderer only ORDERS already-
decided content, it never rewrites it.
"""

from enum import Enum

from src.data_models.resume import Resume, Skill

# Skills with no category land in one bucket, rendered last.
_UNCATEGORIZED = "Skills"

# Total years of experience at/above which a resume is laid out as EXPERIENCED.
# TODO: calibrate on real resumes — Proposed: tune against a labelled sample —
#       Deferred because: this is an unmeasured starting point.
EXPERIENCED_MIN_YEARS = 2.5


class RenderProfile(str, Enum):
    """Which layout a resume gets, by career stage."""

    EXPERIENCED = "experienced"
    ENTRY = "entry"
    STUDENT = "student"


class ResumeSection(str, Enum):
    """A renderable section (ordering is decided per profile)."""

    SUMMARY = "summary"
    SKILLS = "skills"
    EXPERIENCE = "experience"
    EDUCATION = "education"
    CERTIFICATIONS = "certifications"
    LANGUAGES = "languages"


_SECTION_ORDERS: dict[RenderProfile, list[ResumeSection]] = {
    RenderProfile.EXPERIENCED: [
        ResumeSection.SUMMARY,
        ResumeSection.SKILLS,
        ResumeSection.EXPERIENCE,
        ResumeSection.EDUCATION,
        ResumeSection.CERTIFICATIONS,
        ResumeSection.LANGUAGES,
    ],
    RenderProfile.ENTRY: [
        ResumeSection.SUMMARY,
        ResumeSection.EDUCATION,
        ResumeSection.SKILLS,
        ResumeSection.EXPERIENCE,
        ResumeSection.CERTIFICATIONS,
        ResumeSection.LANGUAGES,
    ],
    RenderProfile.STUDENT: [
        ResumeSection.EDUCATION,
        ResumeSection.SKILLS,
        ResumeSection.EXPERIENCE,
        ResumeSection.CERTIFICATIONS,
        ResumeSection.LANGUAGES,
    ],
}


def infer_profile(resume: Resume) -> RenderProfile:
    """Pick a layout profile from the resume's total experience.

    No work experience -> STUDENT (lead with education); below the experienced
    threshold -> ENTRY; at or above it -> EXPERIENCED. Callers may override by
    passing a profile explicitly to the renderer.
    """
    if not resume.work_experience:
        return RenderProfile.STUDENT
    if resume.total_years_of_experience >= EXPERIENCED_MIN_YEARS:
        return RenderProfile.EXPERIENCED
    return RenderProfile.ENTRY


def section_order(profile: RenderProfile) -> list[ResumeSection]:
    """Return the section order for a profile (the caller skips empty sections)."""
    return list(_SECTION_ORDERS[profile])


def group_skills(skills: list[Skill]) -> list[tuple[str, list[str]]]:
    """Group skill names by category, preserving the resume's given order.

    Category order follows first appearance in the resume (the upstream skills
    optimizer already orders them by relevance, so the renderer does not re-rank
    them -- that would be content judgment, not ergonomics). Uncategorized skills
    collect into one "Skills" bucket placed last. Returns (category, names) pairs.
    """
    grouped: dict[str, list[str]] = {}
    for skill in skills:
        grouped.setdefault(skill.category or _UNCATEGORIZED, []).append(skill.skill_name)
    categorized = [(cat, names) for cat, names in grouped.items() if cat != _UNCATEGORIZED]
    if _UNCATEGORIZED in grouped:
        categorized.append((_UNCATEGORIZED, grouped[_UNCATEGORIZED]))
    return categorized
