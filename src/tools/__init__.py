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
    get_supported_formats,
    is_format_supported,
    get_available_converters,
)

__all__ = [
    "convert_document_to_markdown",
    "get_supported_formats",
    "is_format_supported",
    "get_available_converters",
]

