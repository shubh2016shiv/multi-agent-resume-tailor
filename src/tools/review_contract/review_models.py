"""
The universal return contract for every tool in src/tools/.

Every tool, mechanical or judgment, returns a ReviewResult: a list of
ReviewComments, each carrying the quoted line, a structured location, a
severity, a confidence, and advice. This is the single shape the orchestrator
and QA layer read, so findings can be weighed instead of blindly trusted.
"""

from enum import Enum

from pydantic import BaseModel


class Severity(str, Enum):
    """How serious a finding is, from blocking to optional polish."""

    BLOCKER = "blocker"
    MAJOR = "major"
    MINOR = "minor"
    SUGGESTION = "suggestion"


class Confidence(str, Enum):
    """How certain the producing tool is. Mechanical tools are always high."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Section(str, Enum):
    """The resume section a comment points at."""

    SUMMARY = "summary"
    EXPERIENCE = "experience"
    SKILLS = "skills"
    EDUCATION = "education"
    CERTIFICATIONS = "certifications"
    OTHER = "other"


class Location(BaseModel):
    """A structured anchor for a comment, used for reliable UI highlighting.

    Preferred over quoted_text for anchoring: a raw quote breaks if the model
    misquotes by a character, but section + index or character span do not.
    """

    section: Section
    bullet_index: int | None = None
    character_span: list[int] | None = None  # [start, end] offsets; list, not tuple,
    # because OpenAI structured output rejects fixed-length-tuple (prefixItems) schemas.
    # TODO: Validate that character_span has exactly 2 elements and end > start.
    #       Proposed: a field validator raising ValueError on invalid spans.
    #       Deferred: no producer emits spans yet; add when the first one does.


class ReviewComment(BaseModel):
    """One finding from one engine: what was noticed, where, and what to change.

    engine_id is stamped by the harness, not the model, so every comment traces
    back to the exact engine (and, via observability, its prompt and token cost).
    """

    engine_id: str
    message: str
    quoted_text: str
    location: Location
    severity: Severity
    confidence: Confidence
    advice: str
    proposed_rewrite: str | None = None


class ReviewResult(BaseModel):
    """The full output of one tool: its comments plus an optional verdict."""

    comments: list[ReviewComment]
    summary: str = ""
    score: float | None = None
    # TODO: Constrain score to 0.0-1.0.
    #       Proposed: Field(ge=0.0, le=1.0) once a tool actually sets a score.
    #       Deferred: no tool emits a score yet.

