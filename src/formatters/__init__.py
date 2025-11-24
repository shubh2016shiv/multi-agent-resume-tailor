"""
Formatters Module
=================

This module provides formatting and filtering utilities to optimize data before
sending it to LLM agents. The main goal is to reduce token usage and costs by:

1. Filtering redundant or unnecessary fields
2. Converting data to TOON (Token-Oriented Object Notation) or Markdown format
3. Agent-specific formatting that includes only what each agent needs

Key Components:
- base_formatter: Core TOON/Markdown conversion and filtering utilities
- quality_assurance_formatter: Quality Assurance Agent-specific formatter
- ats_optimization_formatter: ATS Optimization Agent-specific formatter
- gap_analysis_formatter: Gap Analysis Agent-specific formatter
- professional_summary_formatter: Professional Summary Writer Agent-specific formatter

Supported Formats:
- TOON: ~46% token reduction vs JSON, optimized for LLM processing
- Markdown: Human-readable format with headings, sections, and tables
"""

from src.formatters.ats_optimization_formatter import format_ats_optimization_context
from src.formatters.base_formatter import (
    estimate_tokens,
    filter_nested_dict,
    format_data,
    to_markdown,
    to_toon,
)
from src.formatters.experience_optimizer_formatter import format_experience_optimizer_context
from src.formatters.gap_analysis_formatter import format_gap_analysis_context
from src.formatters.professional_summary_formatter import format_professional_summary_context
from src.formatters.quality_assurance_formatter import format_quality_assurance_context
from src.formatters.skills_optimizer_formatter import format_skills_optimizer_context

__all__ = [
    "to_toon",
    "to_markdown",
    "format_data",
    "filter_nested_dict",
    "estimate_tokens",
    "format_quality_assurance_context",
    "format_ats_optimization_context",
    "format_gap_analysis_context",
    "format_professional_summary_context",
    "format_experience_optimizer_context",
    "format_skills_optimizer_context",
]
