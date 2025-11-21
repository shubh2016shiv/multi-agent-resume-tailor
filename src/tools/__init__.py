"""
Agent Tools Package
------------------

This package contains all the tools that agents can use to interact with the
external world, process documents, and manipulate data.

WHY TOOLS?
- Separation of Concerns: Tools are independent, testable units
- Reusability: Multiple agents can use the same tool
- Maintainability: Easy to update, debug, and extend
- Type Safety: Clear input/output contracts
"""

from .document_converter import (
    convert_document_to_markdown,
    get_available_converters,
    get_supported_formats,
    is_format_supported,
)
from .ats_validation import (
    calculate_keyword_density,
    validate_ats_formatting,
    check_section_headers,
    get_optimal_keyword_density_range,
    get_standard_headers,
    get_incompatible_patterns,
)

__all__ = [
    # Document Converter Tools
    "convert_document_to_markdown",
    "get_supported_formats",
    "is_format_supported",
    "get_available_converters",
    # ATS Validation Tools
    "calculate_keyword_density",
    "validate_ats_formatting",
    "check_section_headers",
    "get_optimal_keyword_density_range",
    "get_standard_headers",
    "get_incompatible_patterns",
]

