"""Public entrypoint for ATS resume assembly."""

from src.orchestration.nodes.resume_assembly.assemble_ats_resume_node import (
    assemble_ats_resume,
)

__all__ = ["assemble_ats_resume"]
