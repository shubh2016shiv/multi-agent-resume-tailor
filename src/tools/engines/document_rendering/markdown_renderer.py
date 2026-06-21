"""Resume -> Markdown: the guaranteed, zero-dependency output format.

Pure Python, no toolchain -- this is what every user gets regardless of OS, and the
fallback when the PDF toolchain is absent. It reuses the same profile-based section
order and skill grouping as the LaTeX/PDF renderer (section_policy), so the Markdown
and the PDF present the candidate identically; only the markup differs.
"""

from src.data_models.resume import Education, Experience, Resume
from src.tools.engines.document_rendering.section_layout import (
    ResumeSection,
    experience_date_range,
    group_skills,
    infer_profile,
    section_order,
)


def build_resume_markdown(resume: Resume) -> str:
    """Render the resume to a complete Markdown document string.

    Section order follows the same career-stage profile as the PDF. Empty sections are
    omitted. Returns Markdown ending in a trailing newline.
    """
    ####################################################
    # STEP 1: MAP EACH SECTION TYPE TO ITS MARKDOWN RENDERER#
    ####################################################
    # This keeps the main flow simple: decide order once, then call the
    # matching renderer for each section.
    renderers = {
        ResumeSection.SUMMARY: _render_summary,
        ResumeSection.SKILLS: _render_skills,
        ResumeSection.EXPERIENCE: _render_experience,
        ResumeSection.EDUCATION: _render_education,
        ResumeSection.CERTIFICATIONS: _render_certifications,
        ResumeSection.LANGUAGES: _render_languages,
    }

    ####################################################
    # STEP 2: START WITH THE HEADER, THEN FOLLOW THE PROFILE-BASED ORDER#
    ####################################################
    blocks = [_render_header(resume)]
    for section in section_order(infer_profile(resume)):
        block = renderers[section](resume)
        if block:
            blocks.append(block)

    ####################################################
    # STEP 3: JOIN THE NON-EMPTY BLOCKS INTO ONE MARKDOWN DOCUMENT#
    ####################################################
    return "\n\n".join(blocks) + "\n"


def _section(title: str, body: str) -> str:
    """A '## Title' heading, a blank line, then the body."""
    ####################################################
    # STEP 1: WRAP RAW BODY TEXT IN A CONSISTENT SECTION SHAPE#
    ####################################################
    return f"## {title}\n\n{body}"


def _render_header(resume: Resume) -> str:
    """The name as an H1 plus a single contact line (email, phone, location, site)."""
    ####################################################
    # STEP 1: BUILD ONE COMPACT CONTACT LINE FROM THE FIELDS THAT EXIST#
    ####################################################
    contact = [resume.email]
    if resume.phone_number:
        contact.append(resume.phone_number)
    if resume.location:
        contact.append(resume.location)
    if resume.website_or_portfolio:
        contact.append(resume.website_or_portfolio)

    ####################################################
    # STEP 2: RETURN THE NAME PLUS CONTACT DETAILS AS THE DOCUMENT HEADER#
    ####################################################
    return f"# {resume.full_name}\n\n{' | '.join(contact)}"


def _render_summary(resume: Resume) -> str:
    if not resume.professional_summary:
        return ""
    return _section("Summary", resume.professional_summary)


def _render_skills(resume: Resume) -> str:
    if not resume.skills:
        return ""
    ####################################################
    # STEP 1: RENDER EACH SKILL CATEGORY AS ONE BULLET LINE#
    ####################################################
    lines = [
        f"- **{category}:** {', '.join(names)}" for category, names in group_skills(resume.skills)
    ]
    return _section("Skills", "\n".join(lines))


def _render_experience(resume: Resume) -> str:
    if not resume.work_experience:
        return ""
    ####################################################
    # STEP 1: RENDER EACH ROLE, THEN SEPARATE ROLES WITH A BLANK LINE#
    ####################################################
    return _section(
        "Experience", "\n\n".join(_render_role(role) for role in resume.work_experience)
    )


def _render_role(role: Experience) -> str:
    """One role: bold title + company, an italic location/date meta line, achievement bullets."""
    ####################################################
    # STEP 1: BUILD THE ROLE HEADING AND META LINE#
    ####################################################
    meta = " | ".join(part for part in (role.location, experience_date_range(role)) if part)
    lines = [f"**{role.job_title}**, {role.company_name}"]
    if meta:
        lines.append(f"_{meta}_")

    ####################################################
    # STEP 2: LIST EACH ACHIEVEMENT AS A BULLET#
    ####################################################
    lines.extend(f"- {achievement}" for achievement in role.achievements)
    return "\n".join(lines)


def _render_education(resume: Resume) -> str:
    if not resume.education:
        return ""
    return _section(
        "Education", "\n\n".join(_render_education_entry(entry) for entry in resume.education)
    )


def _render_education_entry(education: Education) -> str:
    ####################################################
    # STEP 1: BUILD THE DEGREE LINE, INCLUDING FIELD OF STUDY WHEN PRESENT#
    ####################################################
    degree = education.degree
    if education.field_of_study:
        degree += f" in {education.field_of_study}"

    ####################################################
    # STEP 2: RETURN THE SCHOOL HEADING PLUS DEGREE DETAILS#
    ####################################################
    return f"**{education.institution_name}** ({education.graduation_year})\n{degree}"


def _render_certifications(resume: Resume) -> str:
    if not resume.certifications:
        return ""
    return _section("Certifications", "\n".join(f"- {item}" for item in resume.certifications))


def _render_languages(resume: Resume) -> str:
    if not resume.languages:
        return ""
    return _section("Languages", ", ".join(resume.languages))
