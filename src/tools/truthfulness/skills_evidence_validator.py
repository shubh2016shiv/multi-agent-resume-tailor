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
from src.tools.llm_gateway import request_review
from src.tools.review_contract.review_models import ReviewResult

ENGINE_ID = "skills_evidence_validator"

SKILLS_EVIDENCE_RUBRIC = """You verify that each skill a candidate lists is actually backed
by evidence somewhere in their resume, judged for the candidate's apparent field.

You are given two parts:
1. SKILLS TO VERIFY -- the skills the candidate claims.
2. EVIDENCE FROM THE RESUME -- their summary, experience, education, and certifications.

For each listed skill, decide whether the evidence supports it. Evidence can appear
ANYWHERE in the resume: a project or achievement that uses the skill (even without naming
it), a degree or field of study, a certification, or the summary. A skill does not need its
own sentence -- demonstrated use is enough.

Flag a skill ONLY when nothing in the resume supports it. An unbacked skill exposes the
candidate in interviews and trips ATS keyword-stuffing filters.

Do NOT flag self-evident, universally-assumed skills such as "Microsoft Word", "Email", or
"Internet" -- nobody needs evidence for those, and flagging them is noise.

When you are unsure whether the field implicitly covers a skill, set confidence to "low"
instead of flagging it confidently.

Return one comment per unsupported skill, with:
- severity: "major" (an unbacked skill is a credibility risk)
- confidence: "high" when the skill is concrete and clearly absent from all evidence;
  "medium" when it is likely unsupported; "low" when the field may implicitly cover it
- message: which skill lacks supporting evidence
- quoted_text: the skill name exactly as listed
- advice: either add experience that demonstrates the skill, or remove it
- location: section "skills"

Do not invent evidence. If every listed skill is supported, return no comments.
"""


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
