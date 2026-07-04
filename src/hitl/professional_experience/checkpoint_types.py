"""The msgpack allowlist for LangGraph checkpoint (de)serialization.

LangGraph's JsonPlusSerializer refuses to deserialize a custom type unless it is
explicitly allowed, to stop a malicious checkpoint file from instantiating an
arbitrary class. Every Pydantic model or Enum that can appear anywhere inside
ResumeEnhancementPipelineState (including nested fields) must be listed here, or
resuming a paused run raises once LangGraph enforces this by default -- today it
only warns (LANGGRAPH_STRICT_MSGPACK=false).

Keep this list in sync with ResumeEnhancementPipelineState's field types (see
src/orchestration/state.py): add an entry here whenever a new model or enum
becomes reachable from a state field, including through nesting.
"""

CHECKPOINT_ALLOWED_MSGPACK_MODULES: list[tuple[str, str]] = [
    # src/data_models/resume.py
    ("src.data_models.resume", "Skill"),
    ("src.data_models.resume", "Experience"),
    ("src.data_models.resume", "Education"),
    ("src.data_models.resume", "SkillCategory"),
    ("src.data_models.resume", "OptimizedSkillsSection"),
    ("src.data_models.resume", "Resume"),
    # src/data_models/job.py
    ("src.data_models.job", "JobLevel"),
    ("src.data_models.job", "SkillImportance"),
    ("src.data_models.job", "JobRequirement"),
    ("src.data_models.job", "JobDescription"),
    # src/tools/contracts/review.py -- requirement_match_report
    ("src.tools.contracts.review", "Severity"),
    ("src.tools.contracts.review", "Confidence"),
    ("src.tools.contracts.review", "Section"),
    ("src.tools.contracts.review", "Location"),
    ("src.tools.contracts.review", "ReviewComment"),
    ("src.tools.contracts.review", "ReviewResult"),
    # src/data_models/strategy.py -- alignment_strategy
    ("src.data_models.strategy", "SkillMatch"),
    ("src.data_models.strategy", "SkillGap"),
    ("src.data_models.strategy", "AlignmentStrategy"),
    # src/agents/professional_summary/models.py -- professional_summary
    ("src.agents.professional_summary.models", "SummaryDraft"),
    ("src.agents.professional_summary.models", "ProfessionalSummary"),
    # src/agents/professional_experience/models.py -- optimized_experience
    ("src.agents.professional_experience.models", "ExperienceRelevance"),
    ("src.agents.professional_experience.models", "OptimizedExperienceSection"),
    # src/hitl/professional_experience/models.py -- experience_clarifications / clarification_answers
    ("src.hitl.professional_experience.models", "ExperienceBulletMissingFactCategory"),
    ("src.hitl.professional_experience.models", "ExperienceBulletClarification"),
    # src/agents/ats_optimizer/models.py -- optimized_resume
    ("src.agents.ats_optimizer.models", "AtsOptimizedResume"),
    # src/data_models/evaluation.py -- quality_report / rendered_structure_evaluation
    ("src.data_models.evaluation", "AtsCheckStatus"),
    ("src.data_models.evaluation", "RenderedStructureEvaluation"),
    ("src.data_models.evaluation", "TruthfulnessEvaluation"),
    ("src.data_models.evaluation", "JobAlignmentEvaluation"),
    ("src.data_models.evaluation", "ATSMetrics"),
    ("src.data_models.evaluation", "QualityFeedback"),
    ("src.data_models.evaluation", "ResumeQualityReport"),
    # src/data_models/rendering.py -- rendered_artifacts
    ("src.data_models.rendering", "RenderedResumeArtifacts"),
]
