"""
ResumeEnhancementPipelineState -- the graph's shared state, and how to design one.

WHAT "STATE" ACTUALLY IS
    In LangGraph, state is not an object you create -- it is a SCHEMA: a type that
    describes the shape of the data flowing through the graph. Here that schema is
    the class below, a TypedDict. At runtime a state is nothing more than a plain
    Python dict; TypedDict adds no runtime behavior at all. All it gives you is a
    fixed set of key names and a type for each one, checked by your editor and by
    pyright -- so a typo like state["jobdescription"] is caught before the code runs,
    instead of a real dict silently returning None from state.get("jobdescription").

HOW A VALUE ACTUALLY MOVES THROUGH THE GRAPH
    graph.invoke(initial_state) starts with one dict matching this schema (built by
    new_pipeline_state, below). Every node function receives the WHOLE current state
    and returns a dict holding ONLY the keys it changed -- never the state back in
    full. LangGraph takes that partial dict and merges it into the master state
    before running the next node. The merge rule used everywhere in this file is
    overwrite: a key in the returned dict replaces that key's old value outright.
    LangGraph can also merge a field a different way (a "reducer" -- e.g. append
    to a list instead of replacing it, common for a running chat-message history),
    but nothing here needs to accumulate, so no field in this class uses one.

WHY EVERY FIELD A NODE PRODUCES IS TYPED "SomeType | None"
    A node may only read a field once the node that produces it has actually run,
    and at any point mid-graph most fields have not been produced yet -- especially
    here, since Stage 1 and Stage 3 each run several nodes in parallel. This file's
    convention is: type every field a node produces as `X | None`, default it to
    None in new_pipeline_state(), and read it through require() (below) rather than
    directly -- so "this hasn't run yet" fails clearly at the read, not as a
    confusing crash a few lines further down. graph.py's edges are what guarantee a
    field is actually populated by the time a downstream node reads it; require()
    is only the safety net for when that wiring turns out to be wrong.

    This is a convention, not something LangGraph requires. Other projects handle
    the same problem differently -- e.g. a separate input schema and output schema
    instead of unioning None into every field, or a reducer with a default so a
    field is never literally None. This file's choice is the simplest one for a
    pipeline that is mostly linear, with a handful of parallel stages.

HOW TO DECIDE WHAT BELONGS IN STATE (this generalizes to any LangGraph project)
    Put a field in state only if it is PRODUCED by one node and READ by a different
    node, or by one of graph.py's routing functions. That is the entire test --
    state exists to carry a value across a node boundary, nothing else.

      - A value used only inside one node's own function body is a local variable
        in that function, full stop -- it never touches state. The raw markdown
        text extracted from a job-description PDF, for example, lives and dies
        inside analyze_job(); only the structured JobDescription it produces
        (job_description, below) is what other nodes actually need.
      - Configuration that stays fixed all run (which LLM to call, output paths,
        feature flags) is not state -- state is for what changes stage to stage.
        Nodes read that from get_config() instead (see src/core/settings). The
        exception is the true run inputs below (resume_path, jd_path, run_id):
        those are fixed too, but enough different nodes need them that threading
        them through state is simpler than threading them through every function.
      - If two fields would always be computable from one another, keep only one --
        prefer a field a downstream node can use as-is over one it must re-derive.
      - Keep the dict flat, and group fields by the stage that produces them (as
        below). Flat matters mechanically, not just for readability: LangGraph's
        default merge only looks at top-level keys, so if a field's value is itself
        a nested object, any update to it replaces that whole object -- there is no
        merging inside it.

Fields below are grouped by the stage that produces them; graph.py's edges are the
source of truth for which stage actually runs before which. A None value means the
producing node has not yet run.
"""

from typing import TypedDict

from src.agents.ats_optimizer.models import AtsOptimizedResume
from src.agents.professional_experience.models import OptimizedExperienceSection
from src.agents.professional_summary.models import ProfessionalSummary
from src.data_models.evaluation import RenderedStructureEvaluation, ResumeQualityReport
from src.data_models.job import JobDescription
from src.data_models.rendering import RenderedResumeArtifacts
from src.data_models.resume import OptimizedSkillsSection, Resume
from src.data_models.strategy import AlignmentStrategy
from src.hitl.professional_experience.models import (
    ExperienceBulletClarification,
)
from src.tools.contracts import ReviewResult


class ResumeEnhancementPipelineState(TypedDict):
    """All data in flight between pipeline nodes. See the module docstring above
    for what state is, how it moves through the graph, and how these fields were
    chosen."""

    # --- inputs (set by the runner before graph.invoke()) ---
    run_id: str  # identifies this pipeline run
    resume_path: str
    jd_path: str
    # Candidate's answered clarifications from a previous run's sheet (HITL loop).
    # None/empty on a first run; answers are folded into role evidence by the
    # experience node so candidate-provided facts and figures become usable.
    clarification_answers: list[ExperienceBulletClarification] | None

    # --- Stage 1: parallel ingestion ---
    resume: Resume | None
    job_description: JobDescription | None

    # --- Stage 2: sequential gap analysis ---
    requirement_match_report: ReviewResult | None  # code-computed; fed into the agent's context
    alignment_strategy: AlignmentStrategy | None

    # --- Stage 3: parallel content generation ---
    professional_summary: ProfessionalSummary | None
    optimized_experience: OptimizedExperienceSection | None
    optimized_skills: OptimizedSkillsSection | None
    # Questions for the candidate about bullets that shipped truthful but thin
    # (HITL loop). The runner writes these to an editable clarification sheet.
    experience_clarifications: list[ExperienceBulletClarification] | None

    # --- Stage 4: sequential ATS assembly ---
    optimized_resume: AtsOptimizedResume | None

    # --- Stage 5: sequential quality assurance ---
    quality_report: ResumeQualityReport | None
    # Code-owned rendered-ATS verdict (authoritative over the agent's self-cert).
    # A non-PASS status here forces quality_report.passes_quality_gate to False.
    rendered_structure_evaluation: RenderedStructureEvaluation | None

    # --- Stage 5b: conditional ATS section recovery (only when the ATS check FAILed) ---
    # Terminal disposition: True when the ATS failure is unrecoverable (the essential
    # section is empty upstream too) or the check was INCONCLUSIVE (nothing could be
    # built to inspect). The resume is not rendered; a human must review it.
    human_review_required: bool

    # --- Stage 6: conditional render (only when the QA gate passes) ---
    # Markdown always; PDF best-effort. None until the render node runs.
    rendered_artifacts: RenderedResumeArtifacts | None


def new_pipeline_state(
    run_id: str,
    resume_path: str,
    jd_path: str,
) -> ResumeEnhancementPipelineState:
    """Build the state a fresh run starts from: the three inputs, everything else empty.

    Every key must be present even when its value is None, so a new field added to
    the TypedDict must be added here too.

    Two fields start with a value rather than None, because they are read before any
    node could have produced them: human_review_required (False) is read by the final
    disposition, and clarification_answers (empty list) is read by the HITL router.
    """
    return {
        "run_id": run_id,
        "resume_path": resume_path,
        "jd_path": jd_path,
        "clarification_answers": [],
        "resume": None,
        "job_description": None,
        "requirement_match_report": None,
        "alignment_strategy": None,
        "professional_summary": None,
        "optimized_experience": None,
        "experience_clarifications": None,
        "optimized_skills": None,
        "optimized_resume": None,
        "quality_report": None,
        "rendered_structure_evaluation": None,
        "human_review_required": False,
        "rendered_artifacts": None,
    }


def require[T](value: T | None, field: str) -> T:
    """Check one state field is not None before a node uses it, or crash with a clear reason.

    Every field in ResumeEnhancementPipelineState is typed "SomeType | None" (see the
    class above), because it really is None until its producing node has run. A node
    reading a field it needs must always be sure that node already ran. This function
    is that check, written once instead of copy-pasted in every node:

        resume = require(state["resume"], "resume")
        # is short for:
        #   resume = state["resume"]
        #   if resume is None:
        #       raise RuntimeError(...)

    If this ever fires, it means the GRAPH is wired wrong -- some node ran before the
    node that was supposed to fill this field first. It does NOT mean the candidate's
    resume or job description was bad; that is a different, expected kind of failure,
    handled elsewhere (see exceptions.py). That is why this raises instead of quietly
    returning a default, and why the CLI lets this crash reach the user as a real
    traceback instead of a friendly message: it is a bug to fix, not an outcome to
    handle gracefully.

    The "[T]" in the function signature is only for your editor: it tells the type
    checker "whatever type goes in comes back out, just with None ruled out", so
    `resume.full_name` autocompletes correctly right after this call. It changes
    nothing about how the code runs.
    """
    if value is None:
        raise RuntimeError(
            f"Pipeline state '{field}' is None; the node that produces it has not run. "
            "This is a graph wiring bug, not a bad input."
        )
    return value
