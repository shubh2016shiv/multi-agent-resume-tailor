"""Data models for resume extraction and parsing."""

import re

from pydantic import BaseModel, Field, field_validator

DATE_PATTERN = re.compile(r"^\d{4}(-(0[1-9]|1[0-2]))?$")


def _check_date(value: str | None) -> str | None:
    """Accept None, 'YYYY' or 'YYYY-MM'; raise ValueError for anything else."""
    if value is not None and not DATE_PATTERN.match(value):
        raise ValueError("date must be 'YYYY-MM' or 'YYYY'")
    return value


class ExperienceEntry(BaseModel):
    """One job in the work history."""

    job_title: str = Field(description="Job title, copied exactly as written in the resume.")
    company_name: str = Field(description="Company name, copied exactly as written.")
    start_date: str = Field(description="Start date as 'YYYY-MM', or 'YYYY' if no month is given.")
    end_date: str | None = Field(
        default=None,
        description="End date as 'YYYY-MM' or 'YYYY'. Null if this is the current job.",
    )
    bullets: list[str] = Field(
        default_factory=list,
        description="Achievement/responsibility bullets, copied word for word. Do not rewrite them.",
    )

    @field_validator("start_date", "end_date")
    @classmethod
    def validate_date(cls, value: str | None) -> str | None:
        """Reject dates that are not 'YYYY-MM' or 'YYYY'."""
        return _check_date(value)


class EducationEntry(BaseModel):
    """One education entry."""

    institution: str = Field(description="School or university name, copied exactly as written.")
    degree: str | None = Field(default=None, description="Degree, e.g. 'B.Tech'. Null if absent.")
    field_of_study: str | None = Field(default=None, description="Major or field. Null if absent.")
    graduation_date: str | None = Field(
        default=None, description="Graduation date as 'YYYY-MM' or 'YYYY'. Null if absent."
    )

    @field_validator("graduation_date")
    @classmethod
    def validate_date(cls, value: str | None) -> str | None:
        """Reject dates that are not 'YYYY-MM' or 'YYYY'."""
        return _check_date(value)


class ParsedResume(BaseModel):
    """A resume as structured data. Every value must come from the source text."""

    full_name: str = Field(description="Candidate's full name as written.")
    email: str | None = Field(default=None, description="Email address. Null if absent.")
    phone: str | None = Field(default=None, description="Phone number. Null if absent.")
    location: str | None = Field(default=None, description="City/region. Null if absent.")
    summary: str | None = Field(
        default=None, description="Professional summary text. Null if absent."
    )
    experience: list[ExperienceEntry] = Field(
        default_factory=list, description="Work history, in the order written in the resume."
    )
    education: list[EducationEntry] = Field(default_factory=list, description="Education entries.")
    skills: list[str] = Field(
        default_factory=list, description="Skills listed in the resume. Do not add inferred skills."
    )
