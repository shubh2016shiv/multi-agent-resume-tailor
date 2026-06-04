"""
Resume rendering: a structured Resume -> a polished, ATS-safe PDF.

The pipeline's final, file-producing engine. Mechanical, no LLM. It lays out
already-decided content per domain-neutral rules (section_policy) into a proven
ATS LaTeX template, then compiles it via the rendering sidecar. It returns a file
Path, not a ReviewResult -- a renderer produces an artifact, it does not review.

Two entry points keep the toolchain dependency at the edge:
- build_resume_tex: pure, deterministic, no toolchain -- fully testable offline.
- render_resume_document: build_resume_tex + sidecar compile -> PDF Path.

Input precondition: the Resume is the FINAL, PII-rehydrated resume. The orchestrator
rehydrates redaction placeholders before rendering (TOOLING_PLAN section 10).
"""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from src.data_models.resume import Education, Experience, Resume
from src.tools.document_rendering.latex_escape import escape_latex, escape_latex_url
from src.tools.document_rendering.section_policy import (
    RenderProfile,
    ResumeSection,
    group_skills,
    infer_profile,
    section_order,
)
from src.tools.document_rendering.sidecar import compile_tex_to_pdf

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_TEMPLATE_NAME = "resume.tex.j2"


def build_resume_tex(resume: Resume, profile: RenderProfile | None = None) -> str:
    """Render the Resume into a complete LaTeX document string (no compilation).

    Args:
        resume: The final, PII-rehydrated resume to render.
        profile: Layout profile; inferred from experience when None.

    Returns:
        A complete LaTeX document as a string, ready for the sidecar to compile.
    """
    chosen = profile or infer_profile(resume)
    template = _build_environment().get_template(_TEMPLATE_NAME)
    return template.render(**_build_context(resume, chosen))


def render_resume_document(
    resume: Resume, output_path: Path, profile: RenderProfile | None = None
) -> Path:
    """Render the Resume to a PDF at output_path and return it.

    Args:
        resume: The final, PII-rehydrated resume.
        output_path: Destination .pdf path.
        profile: Layout profile; inferred when None.

    Returns:
        output_path, pointing at the produced PDF.

    Raises:
        RuntimeError: If the sidecar's toolchain is unavailable or compilation fails.
    """
    return compile_tex_to_pdf(build_resume_tex(resume, profile), output_path)


def _build_environment() -> Environment:
    """A Jinja2 env with LaTeX-safe delimiters and the escaping filters registered.

    LaTeX uses { } and % heavily, which collide with Jinja's defaults, so variables
    are << >>, blocks <% %>, comments <# #>. e_tex/e_url make escaping un-forgettable.
    """
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        variable_start_string="<<",
        variable_end_string=">>",
        block_start_string="<%",
        block_end_string="%>",
        comment_start_string="<#",
        comment_end_string="#>",
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=False,
    )
    env.filters["e_tex"] = escape_latex
    env.filters["e_url"] = escape_latex_url
    return env


def _build_context(resume: Resume, profile: RenderProfile) -> dict:
    """Assemble the template context: header, ordered non-empty sections, and data."""
    ordered = [s.value for s in section_order(profile) if _has_section(resume, s)]
    return {
        "name": resume.full_name,
        "headline": _headline(resume),
        "email": resume.email,
        "phone": resume.phone_number or "",
        "location": resume.location or "",
        "website": resume.website_or_portfolio or "",
        "summary": resume.professional_summary,
        "skill_groups": [{"category": c, "skills": n} for c, n in group_skills(resume.skills)],
        "experiences": [_experience_context(e) for e in resume.work_experience],
        "education": [_education_context(e) for e in resume.education],
        "certifications": resume.certifications,
        "languages": resume.languages,
        "sections": ordered,
    }


def _has_section(resume: Resume, section: ResumeSection) -> bool:
    """Whether the resume has any data for a section (empty sections are dropped)."""
    return bool(
        {
            ResumeSection.SUMMARY: resume.professional_summary,
            ResumeSection.SKILLS: resume.skills,
            ResumeSection.EXPERIENCE: resume.work_experience,
            ResumeSection.EDUCATION: resume.education,
            ResumeSection.CERTIFICATIONS: resume.certifications,
            ResumeSection.LANGUAGES: resume.languages,
        }[section]
    )


def _headline(resume: Resume) -> str:
    """A one-line headline: the most recent role's title, or empty when no roles."""
    if not resume.work_experience:
        return ""
    return max(resume.work_experience, key=lambda role: role.start_date).job_title


def _experience_context(experience: Experience) -> dict:
    """Flatten an Experience into template fields, with a formatted date range."""
    return {
        "company": experience.company_name,
        "job_title": experience.job_title,
        "location": experience.location or "",
        "date_range": _date_range(experience),
        "achievements": experience.achievements,
    }


def _date_range(experience: Experience) -> str:
    """Format 'Mon YYYY -- Mon YYYY', or '-- Present' for the current role."""
    start = experience.start_date.strftime("%b %Y")
    if experience.is_current_position or experience.end_date is None:
        return f"{start} -- Present"
    return f"{start} -- {experience.end_date.strftime('%b %Y')}"


def _education_context(education: Education) -> dict:
    """Flatten an Education entry into template fields."""
    return {
        "institution": education.institution_name,
        "degree": education.degree,
        "field": education.field_of_study,
        "year": education.graduation_year,
    }
