"""Contracts for professional-experience bullet clarification HITL.

WHAT THIS MODULE IS
-------------------
The vocabulary of the whole feature. Every other file in the package produces or
consumes one of these shapes. Nothing here has behaviour beyond validation and
two computed properties.

THE SHAPES, AND HOW THEY FLOW THROUGH ONE PAUSE
----------------------------------------------

  1. LLM emits, per bullet:        ExperienceBulletFactGapFinding
                                     .gap : CandidateFactGap | None
                                             (null  -> ship the bullet as-is)
                                             (value -> pause and ask)
                                        |
  2. pure code joins + stamps:     ExperienceBulletClarification   (INHERITS CandidateFactGap)
     (clarifications_from_findings)   the persisted question. answer="" for now.
                                        |
  3. written to the sheet, pipeline exits ................. days pass .........
                                        |
  4. candidate answers; each answer   ClarificationAnswerRecord  -> answers_audit.jsonl
     also appends an audit line       (immutable; the sheet entry is the mutable copy)
                                        |
  5. resume reads the sheet back:    ExperienceBulletClarification  with answer + answered_at + answered_by

  ALONGSIDE, describing the paused run itself:
     ExperienceClarificationPausedRunManifest   run_id, paths, paused_at, expires_at
       -> its .is_expired property is what the runner checks before resuming

WHY ExperienceBulletClarification INHERITS CandidateFactGap
----------------------------------------------------------
The four "content" fields (category, missing fact, why, question) are the SAME
fields the LLM phrased and the candidate answers. Inheritance means no code
copies them across the join, so they cannot be renamed on one side only or drift
out of sync. Optionality lives at ONE nesting point
(ExperienceBulletFactGapFinding.gap), never on the individual fields -- so
"a category and a question exist exactly when input is needed" is guaranteed by
the schema, not re-checked in code.
"""

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field

from src.data_models.resume import Experience


# =============================================================================
# ExperienceBulletMissingFactCategory  --  a closed set of "kinds of missing fact"
# -----------------------------------------------------------------------------
# SET BY  : the LLM, on each gap it reports
# READ BY : nothing branches on it -- it is labelling/analytics, so you can later
#           ask "how many pauses were about scale vs result?"
# =============================================================================
class ExperienceBulletMissingFactCategory(str, Enum):
    """Candidate-owned fact types that can justify pausing the experience flow."""

    ARTIFACT = "artifact"  # What the candidate built, changed, shipped, or operated.
    RESULT = "result"  # What improved because of the work, with or without a metric.
    USER_SCOPE = "user_scope"  # Who or what the work served: users, teams, systems, clients.
    SCALE = "scale"  # Size/context such as data volume, request load, team size, or rollout scope.


# =============================================================================
# CandidateFactGap  --  what one bullet is missing + the question that asks for it
# -----------------------------------------------------------------------------
# CONSTRUCTED BY : the LLM (as part of an ExperienceBulletFactGapFinding)
# INHERITED BY   : ExperienceBulletClarification  (the persisted question)
# THE KEY PROPERTY : all four fields are REQUIRED. A "half a gap" -- category but
#   no question, say -- is unrepresentable, so the join code has no partial-gap
#   branch to defend against and nothing gets silently dropped.
# =============================================================================
class CandidateFactGap(BaseModel):
    """What one bullet is missing, plus the question that asks the candidate for it.

    Every field is required on purpose: a gap that cannot name its category or
    phrase its question is not a gap. Optionality lives at the *nesting* point
    (ExperienceBulletFactGapFinding.gap), never on these fields -- so "a category
    and a question exist exactly when candidate input is needed" is guaranteed by
    the schema rather than re-checked, and silently dropped, in code.
    """

    gap_category: ExperienceBulletMissingFactCategory = Field(
        ...,
        description="The main kind of candidate-owned fact still missing from this bullet.",
    )
    missing_fact_summary: str = Field(
        ...,
        description="Plain summary of the exact fact missing from the shipped bullet.",
    )
    why_flagged: str = Field(
        ...,
        description="Why this fact must come from the candidate instead of being inferred.",
    )
    question: str = Field(
        ...,
        description=(
            "One direct, concise, professional candidate-facing question asking for "
            "exactly the missing fact."
        ),
    )


# =============================================================================
# ExperienceBulletFactGapFinding  --  the LLM's verdict on ONE bullet
# -----------------------------------------------------------------------------
# CONSTRUCTED BY : request_structured_output(), one per shipped bullet, inside
#                  audit_experience_candidate_fact_gaps()  [clarifications.py]
# READ BY        : clarifications_from_findings()  [clarifications.py]
# THE TRIGGER    : `gap is None` -> this bullet ships unchanged.
#                  `gap is not None` -> the pipeline will pause. There is NO
#                  separate boolean; the gap's presence IS the entire signal.
# =============================================================================
class ExperienceBulletFactGapFinding(BaseModel):
    """The LLM's verdict on one shipped bullet.

    A null gap means the bullet may ship as written. A present gap is the entire
    trigger for pausing the pipeline -- there is no separate boolean that could
    disagree with it.
    """

    bullet_id: str = Field(
        ...,
        description="Stable ID of the bullet being judged, copied exactly from the input.",
    )
    # Unread by code: the text shown to the candidate is resolved from the rewrite
    # record instead. Kept because making the model restate what it is judging
    # grounds the verdict; drop it only with a quality comparison in hand.
    current_bullet: str = Field(
        ...,
        description="The exact bullet text that would ship if no candidate input is collected.",
    )
    gap: CandidateFactGap | None = Field(
        default=None,
        description=(
            "The missing fact and its question when this bullet needs candidate input; "
            "null when the bullet is good enough to ship as written."
        ),
    )


# =============================================================================
# ExperienceBulletFactGapReview  --  the wrapper request_structured_output returns
# -----------------------------------------------------------------------------
# CONSTRUCTED BY : request_structured_output() -- structured output needs an
#                  object at the top level, not a bare list, so this exists.
# READ BY        : build_bullet_clarifications() uses `.findings`
# =============================================================================
class ExperienceBulletFactGapReview(BaseModel):
    """One structured-output call's verdict on every shipped bullet of a role."""

    findings: list[ExperienceBulletFactGapFinding] = Field(
        default_factory=list,
        description="One fact-gap decision per shipped experience bullet.",
    )


# =============================================================================
# ExperienceBulletClarification  --  THE persisted question/answer contract
# -----------------------------------------------------------------------------
# The one object that crosses the pause. SAME class on both sides:
#   * built with answer="" by clarifications_from_findings()  [clarifications.py]
#   * written to the sheet by write_clarification_sheet()      [persistence.py]
#   * read back with answer filled by read_answered_clarifications()  [persistence.py]
#   * routed to its role by answers_for_role()                 [answers.py]
#   * carried through pipeline state as experience_clarifications
#     and clarification_answers                                [orchestration/state.py]
#
# INHERITS the 4 content fields from CandidateFactGap -- see module docstring.
# ADDS the routing identity (company/title/start_date/bullet_id) an answer needs
# to find home, plus the answer and its provenance.
# =============================================================================
class ExperienceBulletClarification(CandidateFactGap):
    """One question to the candidate about one experience bullet.

    The same object travels both directions across the pause: built with an empty
    answer when the pipeline stops, read back with the answer filled in when it
    resumes. One shape for both means there is nothing to reconcile -- the class
    of bug that breaks most human-in-the-loop systems.
    """

    bullet_id: str = Field(
        ...,
        description="Stable ID of the exact bullet this question belongs to; the routing key.",
    )
    company_name: str = Field(
        ...,
        description="Company of the role this bullet belongs to.",
    )
    job_title: str = Field(
        ...,
        description="Job title of the role this bullet belongs to.",
    )
    start_date: str = Field(
        default="",
        description=(
            "ISO start date of the role: the tiebreaker when the candidate held the "
            "same title at the same company twice."
        ),
    )
    bullet: str = Field(
        ...,
        description="The bullet text as it shipped -- shown so the candidate knows what to elaborate on.",
    )
    answer: str = Field(
        default="",
        description="The candidate's answer in their own words; empty until they fill it in.",
    )
    answered_at: datetime | None = Field(
        default=None,
        description="When the answer was recorded (UTC). None while unanswered.",
    )
    answered_by: str | None = Field(
        default=None,
        description=(
            "Who supplied the answer. Not authenticated today -- this records the "
            "surface that submitted it, and is the seam real authorization plugs into."
        ),
    )

    @property
    def is_answered(self) -> bool:
        """Whether this question carries a usable answer.

        USED BY : read_answered_clarifications() [persistence.py] and the
        'at least one answer' guards. Whitespace does not count.
        """
        return bool(self.answer.strip())


# =============================================================================
# ClarificationAnswerRecord  --  one immutable audit line
# -----------------------------------------------------------------------------
# CONSTRUCTED BY : record_clarification_answers()  [persistence.py], one per
#                  freshly-answered question
# WRITTEN BY     : append_answer_records()  -> answers_audit.jsonl (append-only)
# READ BY        : nothing in code today -- it exists to be read by a human or a
#                  compliance process. Correcting an answer leaves BOTH lines.
# =============================================================================
class ClarificationAnswerRecord(BaseModel):
    """One immutable audit entry: who answered what, and when."""

    bullet_id: str = Field(..., description="The bullet this answer belongs to.")
    answer: str = Field(..., description="The answer exactly as supplied.")
    answered_by: str = Field(..., description="The surface or identity that supplied it.")
    answered_at: datetime = Field(..., description="When it was recorded (UTC).")


# =============================================================================
# ExperienceClarificationPausedRunManifest  --  identity + lifetime of a pause
# -----------------------------------------------------------------------------
# CONSTRUCTED BY : runner._finalize_pipeline_output()  [orchestration/runner.py]
#                  at the moment the graph interrupts
# WRITTEN BY     : save_paused_run_state()  -> paused_run_manifest.json
# READ BACK BY   : load_paused_run()  [persistence.py] on every resume attempt
# THE POINT      : when the graph pauses the process may exit entirely. A resume
#                  can arrive days later from a process that has never seen this
#                  run. This file is how that process learns which LangGraph
#                  thread to continue (run_id) and whether it still may (is_expired).
# =============================================================================
class ExperienceClarificationPausedRunManifest(BaseModel):
    """Identity and lifetime of one paused run.

    File *names* deliberately live in PausedRunLayout, not here: they never vary
    per run, so storing them per run only created two copies of one fact.
    """

    run_id: str = Field(..., description="Pipeline run id, and the LangGraph thread id on resume.")
    resume_path: str = Field(..., description="Original resume path, carried across the pause.")
    jd_path: str = Field(
        ..., description="Original job-description path, carried across the pause."
    )
    paused_at: datetime = Field(..., description="When the run paused for candidate input (UTC).")
    expires_at: datetime = Field(
        ...,
        description=(
            "When this paused run stops being resumable (UTC). Without a deadline a "
            "paused run waits forever, which is the standard failure mode of "
            "human-in-the-loop systems."
        ),
    )

    @property
    def is_expired(self) -> bool:
        """Whether the candidate's window to answer has closed.

        CHECKED BY : resume_paused_run()  [orchestration/runner.py], before it
        opens a checkpoint or compiles a graph. Computed from the clock, never
        stored: a stored status can drift out of sync with reality, a computed
        one cannot.
        """
        return datetime.now(UTC) >= self.expires_at


# =============================================================================
# build_experience_bullet_id()  --  the stable id scheme every bullet_id uses
# -----------------------------------------------------------------------------
# CALLED BY : _collect_rewrite_truthfulness_findings()  [orchestration/nodes/experience.py]
#             answers_for_role()                        [answers.py]
#             (and, indirectly, the rewriter agent, which is told to echo these ids)
# WHY IT MATTERS : this id is written onto the sheet and must still match days
#   later when a resume recomputes it. A bare list index would silently point at
#   a different bullet if the resume's roles or bullet counts changed in between.
#   Deriving from company + title + start_date + position survives that.
# =============================================================================
def build_experience_bullet_id(experience: Experience, bullet_index: int) -> str:
    """Return a stable id for one bullet inside one professional-experience role."""
    # Prefer an explicit code-assigned id if the pipeline set one; otherwise
    # derive a deterministic one from fields that do not shift between runs.
    role_id = experience.experience_id or (
        f"{experience.company_name.strip().lower()}::{experience.job_title.strip().lower()}::"
        f"{experience.start_date.isoformat()}"
    )
    return f"{role_id}::bullet::{bullet_index}"
