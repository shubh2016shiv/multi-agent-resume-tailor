"""
Skills-evidence validation: is every listed skill actually backed by the resume?

Pure judgment engine, no mechanical half. Whether a skill is "evidenced" is not a
string match -- and string matching fails in both directions. A skill can be
backed without being named: "Kubernetes" is supported by a bullet about "container
orchestration", and a degree or certification can support a skill the experience
section never spells out. A skill can also be named without being backed:
"Familiar with Kubernetes through reading" contains the word but is worse than
unbacked, yet a substring search would pass it. Only reading the context settles
either case, so the detection IS the judgment.

Scope is deliberately broad (per issues.md): evidence may live anywhere in the
resume -- summary, experience, education, or certifications -- not only in work
experience. Two guardrails keep false positives down: self-evident skills (e.g.
"Microsoft Word") are never flagged, and when the field may implicitly cover a
skill the model sets confidence to "low" rather than flagging it confidently. The
truthfulness gating policy then blocks only high-confidence findings.
"""

from src.data_models.resume import Experience, Resume
from src.tools.llm_gateway import load_tool_prompt, request_review
from src.tools.review_contract.review_models import ReviewResult

ENGINE_ID = "skills_evidence_validator"

SKILLS_EVIDENCE_RUBRIC = load_tool_prompt("truthfulness/skills_evidence.md")


def validate_skills_evidence(resume: Resume) -> ReviewResult:
    """Flag listed skills that no part of the resume supports, judged for the field.

    Evidence is read broadly: summary, experience (descriptions, achievements, and
    skills_used), education, and certifications all count as support.

    Args:
        resume: The resume to check; its skills are cross-referenced against the
            rest of the resume as the evidence corpus.

    Returns:
        A ReviewResult of judgment comments with honest confidence. An empty result
        (no LLM call) means the candidate listed no skills to verify.
    """
    # TODO: "self-evident" is role-dependent — "Microsoft Word" is noise for an
    #       embedded engineer but a real skill to evidence for a technical writer.
    #       Proposed: rely on the model's field inference; ambiguous cases set
    #       confidence=LOW so gating treats them as advisory, not blocking.
    #       Deferred because: no measured false-negative rate yet to justify more.
    if not resume.skills:
        return ReviewResult(comments=[], summary="No skills listed to verify")
    return request_review(ENGINE_ID, SKILLS_EVIDENCE_RUBRIC, _build_payload(resume))


def _build_payload(resume: Resume) -> str:
    """Lay out the skills-to-verify list and the evidence corpus as one prompt body."""
    skills = "\n".join(
        f"{index}. {name}" for index, name in enumerate(resume.list_of_skill_names, start=1)
    )
    return f"SKILLS TO VERIFY:\n{skills}\n\nEVIDENCE FROM THE RESUME:\n{_format_evidence(resume)}"


def _format_evidence(resume: Resume) -> str:
    """Gather summary, experience, education, and certifications as the evidence corpus.

    Deliberately NOT shared with src.tools.shared.resume_rendering.render_resume,
    despite the surface similarity. This corpus must diverge for correctness:
    it OMITS the skills section (skills are what this engine verifies -- including
    them would let a skill evidence itself) and it INCLUDES per-role skills_used
    (a backing signal render_experience drops). The ~6 identical lines
    (summary/education/certs) are not worth a parameterised shared renderer.
    """
    parts = [f"[Summary]\n{resume.professional_summary}"]
    if resume.work_experience:
        parts.append(f"[Experience]\n{_format_experience(resume.work_experience)}")
    if resume.education:
        education = "\n".join(
            f"- {entry.degree} in {entry.field_of_study}, {entry.institution_name}"
            for entry in resume.education
        )
        parts.append(f"[Education]\n{education}")
    if resume.certifications:
        certifications = "\n".join(f"- {item}" for item in resume.certifications)
        parts.append(f"[Certifications]\n{certifications}")
    return "\n\n".join(parts)


def _format_experience(experiences: list[Experience]) -> str:
    """Render each role's description, achievements, and skills_used as evidence lines."""
    blocks = []
    for role in experiences:
        lines = [f"- {role.job_title} at {role.company_name}: {role.description}"]
        lines.extend(f"  - {achievement}" for achievement in role.achievements)
        if role.skills_used:
            lines.append(f"  (skills used: {', '.join(role.skills_used)})")
        blocks.append("\n".join(lines))
    return "\n".join(blocks)
