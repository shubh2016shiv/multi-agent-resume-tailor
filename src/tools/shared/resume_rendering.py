"""
Shared resume-to-text rendering.

Why it lives in shared/: this generic resume->text renderer started local to the
truthfulness engines (rewrite_drift_detector + claim_inflation_detector). Once
job_matching/requirements_matcher became a third consumer in a different package,
keeping it under truthfulness/ would have forced job_matching to import from
truthfulness -- wrong-way coupling. Generic rendering belongs in a neutral home;
domain-specific rendering (e.g. skills_evidence_validator's evidence corpus, which
deliberately excludes the skills section) stays in its owning package.
"""

from src.data_models.resume import Experience, Resume


def render_resume(resume: Resume) -> str:
    """Render the claim-bearing sections (summary, experience, skills, education, certs) as text."""
    parts = [f"[Summary]\n{resume.professional_summary}"]
    if resume.work_experience:
        parts.append(f"[Experience]\n{render_experience(resume.work_experience)}")
    if resume.skills:
        parts.append(f"[Skills]\n{', '.join(resume.list_of_skill_names)}")
    if resume.education:
        education = "\n".join(
            f"- {entry.degree} in {entry.field_of_study}, {entry.institution_name} "
            f"({entry.graduation_year})"
            for entry in resume.education
        )
        parts.append(f"[Education]\n{education}")
    if resume.certifications:
        certifications = "\n".join(f"- {item}" for item in resume.certifications)
        parts.append(f"[Certifications]\n{certifications}")
    return "\n\n".join(parts)


def render_experience(experiences: list[Experience]) -> str:
    """Render each role's title, company, description, and achievement bullets."""
    blocks = []
    for role in experiences:
        lines = [f"- {role.job_title} at {role.company_name}: {role.description}"]
        lines.extend(f"  - {achievement}" for achievement in role.achievements)
        blocks.append("\n".join(lines))
    return "\n".join(blocks)
