"""The classes a resumed run is allowed to rebuild from its checkpoint file.

Read checkpointing.py first; it explains why this list exists. In short: resuming a
run means rebuilding Pydantic objects from a file that names their classes, and
LangGraph will only import a class that appears here, so a tampered checkpoint cannot
make it instantiate anything else (CVE-2026-28277).

WHEN YOU HAVE TO EDIT THIS FILE
    You added a Pydantic model or Enum that is now reachable from
    ResumeEnhancementPipelineState -- either as a field, or nested inside one.
    Add its (module, class) pair below. Miss one and a resumed run fails on load.

    You do not have to work out that list by hand.
    tests/unit/orchestration/test_checkpoint_allowlist.py asks LangGraph's own schema
    walker what the compiled graph can reach, and fails naming whatever is missing.
    An extra entry that is no longer reachable is harmless, so the test only complains
    in one direction.

    Note the shape: each entry is (module path, class name) -- a class, not a module,
    despite LangGraph calling its parameter allowed_msgpack_modules.

Entries are grouped by the state field that brings them in, so a reader can tell which
part of the pipeline each group serves.
"""

CHECKPOINT_ALLOWED_MSGPACK_MODULES: list[tuple[str, str]] = [
    # state["resume"] -- the parsed resume, and the optimized skills section
    ("src.data_models.resume", "Skill"),
    ("src.data_models.resume", "Experience"),
    ("src.data_models.resume", "Education"),
    ("src.data_models.resume", "SkillCategory"),
    ("src.data_models.resume", "Resume"),
    ("src.data_models.resume", "OptimizedSkillsSection"),
    # state["job_description"]
    ("src.data_models.job", "JobLevel"),
    ("src.data_models.job", "SkillImportance"),
    ("src.data_models.job", "JobRequirement"),
    ("src.data_models.job", "JobDescription"),
    # state["requirement_match_report"] -- and every other tool review result
    ("src.tools.contracts.review", "Severity"),
    ("src.tools.contracts.review", "Confidence"),
    ("src.tools.contracts.review", "Section"),
    ("src.tools.contracts.review", "Location"),
    ("src.tools.contracts.review", "ReviewComment"),
    ("src.tools.contracts.review", "ReviewResult"),
    # state["alignment_strategy"]
    ("src.data_models.strategy", "SkillMatch"),
    ("src.data_models.strategy", "SkillGap"),
    ("src.data_models.strategy", "AlignmentStrategy"),
    # state["professional_summary"]
    ("src.agents.professional_summary.models", "SummaryDraft"),
    ("src.agents.professional_summary.models", "ProfessionalSummary"),
    # state["optimized_experience"]
    ("src.agents.professional_experience.models", "ExperienceRelevance"),
    ("src.agents.professional_experience.models", "OptimizedExperienceSection"),
    # state["experience_clarifications"] and state["clarification_answers"] -- the
    # HITL question sheet, which is exactly what a paused run has to carry across
    ("src.hitl.professional_experience.models", "ExperienceBulletMissingFactCategory"),
    ("src.hitl.professional_experience.models", "CandidateFactGap"),
    ("src.hitl.professional_experience.models", "ExperienceBulletClarification"),
    # state["optimized_resume"]
    ("src.agents.ats_optimizer.models", "AtsOptimizedResume"),
    # state["quality_report"] and state["rendered_structure_evaluation"]
    ("src.data_models.evaluation", "AtsCheckStatus"),
    ("src.data_models.evaluation", "RenderedStructureEvaluation"),
    ("src.data_models.evaluation", "TruthfulnessEvaluation"),
    ("src.data_models.evaluation", "JobAlignmentEvaluation"),
    ("src.data_models.evaluation", "ATSMetrics"),
    ("src.data_models.evaluation", "QualityFeedback"),
    ("src.data_models.evaluation", "ResumeQualityReport"),
    # state["rendered_artifacts"]
    ("src.data_models.rendering", "RenderedResumeArtifacts"),
]
