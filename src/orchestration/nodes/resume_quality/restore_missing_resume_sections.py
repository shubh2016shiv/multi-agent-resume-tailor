"""Restore missing essential sections from canonical pipeline state."""

from typing import Any

from src.agents.professional_experience.models import OptimizedExperienceSection
from src.data_models.resume import OptimizedSkillsSection, Resume


def restore_missing_resume_sections(
    final_resume: Resume,
    optimized_experience: OptimizedExperienceSection,
    optimized_skills: OptimizedSkillsSection,
    original_resume: Resume,
) -> Resume:
    """Restore empty experience, skills, and education sections when sources exist."""
    updates: dict[str, Any] = {}
    if not final_resume.work_experience and optimized_experience.optimized_experiences:
        updates["work_experience"] = optimized_experience.optimized_experiences
    if not final_resume.skills and optimized_skills.optimized_skills:
        updates["skills"] = optimized_skills.optimized_skills
    if not final_resume.education and original_resume.education:
        updates["education"] = original_resume.education
    return final_resume.model_copy(update=updates) if updates else final_resume


__all__ = ["restore_missing_resume_sections"]
