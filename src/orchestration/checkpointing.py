"""The SQLite-backed LangGraph checkpointer for one pipeline run.

This is generic pipeline plumbing, not a feature: every run opens a checkpointer,
whether or not it ever pauses for human input. It lives here, beside the graph
whose state it serializes, rather than inside the human-in-the-loop package that
happens to be the loudest consumer of it.

Why the explicit allowlist below
--------------------------------
LangGraph's JsonPlusSerializer refuses to reconstruct a custom type during
checkpoint deserialization unless that type is explicitly allowed -- which stops
a tampered checkpoint file from instantiating arbitrary Python classes on load
(the class of vulnerability tracked as CVE-2026-28277). Passing an explicit
allowlist, as this module does, IS strict enforcement: anything unlisted is
rejected regardless of the LANGGRAPH_STRICT_MSGPACK environment variable, which
only changes the serializer's *default* when no allowlist is supplied.

Every Pydantic model or Enum reachable from ResumeEnhancementPipelineState --
including nested fields -- must be listed. Rather than eyeballing that,
tests/unit/orchestration/test_checkpoint_allowlist.py walks the real compiled
graph with LangGraph's own schema walker and fails when this list falls behind.
"""

import sqlite3
from pathlib import Path

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

CHECKPOINT_ALLOWED_MSGPACK_MODULES: list[tuple[str, str]] = [
    # src/data_models/resume.py
    ("src.data_models.resume", "Skill"),
    ("src.data_models.resume", "Experience"),
    ("src.data_models.resume", "Education"),
    ("src.data_models.resume", "SkillCategory"),
    ("src.data_models.resume", "Resume"),
    ("src.data_models.resume", "OptimizedSkillsSection"),
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
    ("src.hitl.professional_experience.models", "CandidateFactGap"),
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


# HITL COMPONENT 4 -- DURABLE STATE (CHECKPOINT). Generic pipeline plumbing
# (every run checkpoints, HITL or not), but it's the mechanism the pause in
# src/orchestration/nodes/experience.py relies on. See
# src/hitl/professional_experience/README.md#7-component-4--durable-state-checkpoint
def open_checkpoint_database(db_path: Path) -> SqliteSaver:
    """Open (creating if needed) the SQLite-backed checkpointer for one run.

    The connection allows cross-thread use because graph nodes run in worker
    threads; SqliteSaver serializes its own writes internally. Close it with
    close_checkpoint_database when the run ends, before moving or deleting the file.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(db_path), check_same_thread=False)
    serde = JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_ALLOWED_MSGPACK_MODULES)
    return SqliteSaver(connection, serde=serde)


def close_checkpoint_database(checkpointer: SqliteSaver) -> None:
    """Close the checkpointer's SQLite connection so its file can be moved or deleted."""
    checkpointer.conn.close()
