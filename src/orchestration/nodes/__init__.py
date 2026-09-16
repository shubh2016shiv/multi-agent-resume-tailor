"""Public graph-node interface for the resume-tailoring pipeline.

Node entrypoints are re-exported here as they are migrated. Consumers should
import nodes from this package instead of depending on implementation modules.
"""

from src.orchestration.nodes.ingestion import analyze_job, extract_resume
from src.orchestration.nodes.professional_experience import (
    await_candidate_clarifications,
    optimize_experience,
)
from src.orchestration.nodes.professional_summary import write_professional_summary
from src.orchestration.nodes.render_resume import render_final_resume
from src.orchestration.nodes.resume_assembly import assemble_ats_resume
from src.orchestration.nodes.resume_quality import evaluate_resume_quality, patch_ats_assembly
from src.orchestration.nodes.skills import optimize_skills
from src.orchestration.nodes.tailoring_strategy import run_gap_analysis

__all__ = [
    "analyze_job",
    "assemble_ats_resume",
    "await_candidate_clarifications",
    "evaluate_resume_quality",
    "extract_resume",
    "optimize_experience",
    "optimize_skills",
    "patch_ats_assembly",
    "render_final_resume",
    "run_gap_analysis",
    "write_professional_summary",
]
