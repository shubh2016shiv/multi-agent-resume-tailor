"""Resume-to-text rendering used by checks that need plain claim-bearing text."""

from src.data_models.resume import Experience, Resume


def render_resume(resume: Resume) -> str:
    """Render the claim-bearing sections (summary, experience, skills, education, certs) as text."""
    ####################################################
    # STEP 1: START WITH THE SECTIONS MOST REVIEW ENGINES NEED#
    ####################################################
    # This renderer is for checks that need plain, claim-bearing text,
    # not polished formatting, so we keep the structure simple and explicit.
    parts = [f"[Summary]\n{resume.professional_summary}"]

    ####################################################
    # STEP 2: ADD EACH NON-EMPTY SECTION IN A PLAIN TEXT FORM#
    ####################################################
    # We skip empty sections so downstream checks only see information
    # the resume actually contains.
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

    ####################################################
    # STEP 3: JOIN THE SECTIONS INTO ONE READABLE TEXT BLOCK#
    ####################################################
    return "\n\n".join(parts)


def render_experience(experiences: list[Experience]) -> str:
    """Render each role's title, company, description, and achievement bullets."""
    ####################################################
    # STEP 1: TURN EACH ROLE INTO A SMALL PLAIN-TEXT BLOCK#
    ####################################################
    # Each block starts with the role identity, then lists the concrete
    # achievements underneath so review engines can inspect the claims.
    blocks = []
    for role in experiences:
        lines = [f"- {role.job_title} at {role.company_name}: {role.description}"]
        lines.extend(f"  - {achievement}" for achievement in role.achievements)
        blocks.append("\n".join(lines))

    ####################################################
    # STEP 2: JOIN ALL ROLE BLOCKS INTO ONE EXPERIENCE SECTION#
    ####################################################
    return "\n".join(blocks)
