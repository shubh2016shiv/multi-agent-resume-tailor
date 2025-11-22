"""
ATS Validation Tools
-------------------

This module contains tools for validating resume ATS (Applicant Tracking System) compatibility.
These tools help ensure resumes are optimized for automated parsing and ranking by ATS software.

TOOLS PROVIDED:
- calculate_keyword_density: Analyze keyword usage and density
- validate_ats_formatting: Check for ATS-incompatible formatting
- check_section_headers: Validate section header conventions

DESIGN PRINCIPLES:
- Deterministic: Same input always produces same output
- Transparent: Clear reporting of what was checked
- Actionable: Provides specific recommendations
- Conservative: Flags potential issues even if uncertain
"""

import re
from crewai.tools import tool

# ==============================================================================
# Module Constants
# ==============================================================================

# Keyword density thresholds
MIN_KEYWORD_DENSITY = 0.02  # 2% - minimum for ATS effectiveness
MAX_KEYWORD_DENSITY = 0.05  # 5% - maximum before keyword stuffing
OPTIMAL_KEYWORD_DENSITY = 0.035  # 3.5% - sweet spot

# Standard ATS-recognized section headers
STANDARD_SECTION_HEADERS = {
    "summary": ["Professional Summary", "Summary", "Profile", "Executive Summary"],
    "experience": ["Work Experience", "Professional Experience", "Experience", "Employment History"],
    "skills": ["Skills", "Technical Skills", "Core Competencies", "Key Skills"],
    "education": ["Education", "Academic Background", "Educational Qualifications"],
    "certifications": ["Certifications", "Professional Certifications", "Licenses & Certifications"],
}

# ATS-incompatible formatting patterns
INCOMPATIBLE_PATTERNS = [
    r"\|.*\|",  # Tables with pipe characters
    r"═+",  # Box drawing characters
    r"┌|┐|└|┘|├|┤|─|│",  # More box drawing
    r"\t",  # Tab characters (can break parsing)
    r"<table>",  # HTML tables
    r"<img>",  # HTML images
]

# Special characters that may break ATS parsing
PROBLEMATIC_CHARACTERS = ["™", "®", "©", "•", "→", "←", "↑", "↓", "★", "☆", "♦", "◆"]


# ==============================================================================
# ATS Validation Tools
# ==============================================================================


@tool("Calculate Keyword Density")
def calculate_keyword_density(resume_text: str, required_keywords: list[str]) -> str:
    """
    Calculate keyword density and coverage in resume text.
    
    Analyzes how well keywords are integrated without stuffing.
    Optimal density is 2-5% of total content.
    
    Args:
        resume_text: The complete resume content as text
        required_keywords: List of keywords from job description
        
    Returns:
        Formatted report of keyword metrics
        
    Example:
        >>> report = calculate_keyword_density(
        ...     "Python developer with AWS experience...",
        ...     ["Python", "AWS", "Docker"]
        ... )
        >>> print(report)
        Keyword Density Analysis:
        ========================
        Total Words: 50
        Keyword Density: 4.0%
        Status: ✓ OPTIMAL
    """
    try:
        # Normalize text
        resume_lower = resume_text.lower()
        words = re.findall(r"\b\w+\b", resume_lower)
        total_words = len(words)

        if total_words == 0:
            return "Error: Resume text is empty"

        # Count keyword occurrences
        keyword_freq: dict[str, int] = {}
        total_keyword_instances = 0

        for keyword in required_keywords:
            keyword_lower = keyword.lower()
            # Count occurrences (handle multi-word keywords)
            count = resume_lower.count(keyword_lower)
            if count > 0:
                keyword_freq[keyword] = count
                total_keyword_instances += count

        # Calculate metrics
        unique_keywords_found = len(keyword_freq)
        keyword_density = total_keyword_instances / total_words if total_words > 0 else 0
        keyword_coverage = unique_keywords_found / len(required_keywords) if required_keywords else 0

        # Determine if optimal
        is_optimal = MIN_KEYWORD_DENSITY <= keyword_density <= MAX_KEYWORD_DENSITY

        # Find missing must-have keywords
        missing = [kw for kw in required_keywords if kw.lower() not in resume_lower]

        # Build report
        report = f"""
Keyword Density Analysis:
========================
Total Words: {total_words}
Total Keyword Instances: {total_keyword_instances}
Unique Keywords Found: {unique_keywords_found}/{len(required_keywords)}
Keyword Density: {keyword_density:.1%}
Keyword Coverage: {keyword_coverage:.1%}
Status: {'[OK] OPTIMAL' if is_optimal else '[!] NEEDS ADJUSTMENT'}

Optimal Range: {MIN_KEYWORD_DENSITY:.1%} - {MAX_KEYWORD_DENSITY:.1%}

Missing Keywords: {', '.join(missing) if missing else 'None'}

Top Keywords:
{chr(10).join([f'  {kw}: {count}x' for kw, count in sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:10]])}
"""
        return report.strip()

    except Exception as e:
        return f"Error calculating keyword density: {str(e)}"


@tool("Validate ATS Formatting")
def validate_ats_formatting(resume_text: str) -> str:
    """
    Check for formatting elements that break ATS parsing.
    
    Scans for tables, graphics indicators, special characters,
    and other patterns that cause ATS failures.
    
    Args:
        resume_text: The complete resume content as text
        
    Returns:
        Formatted report of formatting issues found
        
    Example:
        >>> report = validate_ats_formatting(resume_text)
        >>> print(report)
        ATS Formatting Validation:
        =========================
        [OK] No formatting issues detected
        Status: PASS
    """
    try:
        issues = []

        # Check for incompatible patterns
        for pattern in INCOMPATIBLE_PATTERNS:
            matches = re.findall(pattern, resume_text)
            if matches:
                issues.append(f"Found incompatible pattern '{pattern}': {len(matches)} instances")

        # Check for problematic special characters
        special_chars_found = []
        for char in PROBLEMATIC_CHARACTERS:
            if char in resume_text:
                count = resume_text.count(char)
                special_chars_found.append(f"'{char}' ({count}x)")

        if special_chars_found:
            issues.append(f"Special characters detected: {', '.join(special_chars_found)}")

        # Check for multiple consecutive blank lines (can confuse parsers)
        if re.search(r"\n\s*\n\s*\n", resume_text):
            issues.append("Multiple consecutive blank lines detected (should be single)")

        # Check for tabs
        if "\t" in resume_text:
            tab_count = resume_text.count("\t")
            issues.append(f"Tab characters found ({tab_count}x) - should use spaces")

        # Generate report
        if not issues:
            report = """
ATS Formatting Validation:
=========================
[OK] No formatting issues detected
[OK] No incompatible patterns found
[OK] No problematic special characters
[OK] Resume is ATS-compatible

Status: PASS
"""
        else:
            report = f"""
ATS Formatting Validation:
=========================
[!] {len(issues)} formatting issue(s) detected:

{chr(10).join([f'  {i+1}. {issue}' for i, issue in enumerate(issues)])}

Status: NEEDS FIXES
"""

        return report.strip()

    except Exception as e:
        return f"Error validating ATS formatting: {str(e)}"


@tool("Check Section Headers")
def check_section_headers(resume_text: str) -> str:
    """
    Validate that section headers use standard ATS-recognized conventions.

    ATS systems look for specific header patterns to identify sections.
    Non-standard headers can cause section mis-classification or omission.

    Args:
        resume_text: The complete resume content as text

    Returns:
        Formatted report of section header validation

    Example:
        >>> report = check_section_headers(resume_text)
        >>> print(report)
        Section Header Validation:
        ==================================================
        [OK] Summary: Professional Summary
        [OK] Experience: Work Experience
        [!] Skills: My Technical Expertise
          -> Recommend: 'Skills'
    """
    try:
        results = []
        lines = resume_text.split("\n")

        # Common header patterns (all caps, title case, with/without colons)
        header_pattern = r"^([A-Z][A-Za-z\s&]+):?\s*$"

        detected_headers = []
        for line in lines:
            line = line.strip()
            if re.match(header_pattern, line) and len(line.split()) <= 5:
                detected_headers.append(line.rstrip(":"))

        # Check each expected section
        for section_type, standard_headers in STANDARD_SECTION_HEADERS.items():
            found = False
            matched_header = None

            for detected in detected_headers:
                if any(std.lower() in detected.lower() for std in standard_headers):
                    found = True
                    matched_header = detected
                    break

            is_standard = matched_header in standard_headers if matched_header else False

            results.append({
                "section": section_type.title(),
                "found": found,
                "header": matched_header or "Not found",
                "is_standard": is_standard,
                "recommended": standard_headers[0],
            })

        # Generate report
        report_lines = ["Section Header Validation:", "=" * 50]

        for result in results:
            status = "[OK]" if result["found"] and result["is_standard"] else "[!]"
            report_lines.append(f"{status} {result['section']}: {result['header']}")
            if result["found"] and not result["is_standard"]:
                report_lines.append(f"  -> Recommend: '{result['recommended']}'")
            elif not result["found"]:
                report_lines.append(f"  -> Add section with header: '{result['recommended']}'")

        report_lines.append("")
        report_lines.append("Standard headers improve ATS recognition accuracy.")

        return "\n".join(report_lines)

    except Exception as e:
        return f"Error checking section headers: {str(e)}"


# ==============================================================================
# Utility Functions
# ==============================================================================


def get_optimal_keyword_density_range() -> tuple[float, float]:
    """
    Get the optimal keyword density range.

    Returns:
        Tuple of (min_density, max_density) as percentages (0.0-1.0)
    """
    return (MIN_KEYWORD_DENSITY, MAX_KEYWORD_DENSITY)


def get_standard_headers() -> dict[str, list[str]]:
    """
    Get the dictionary of standard ATS-recognized section headers.

    Returns:
        Dictionary mapping section types to lists of standard headers
    """
    return STANDARD_SECTION_HEADERS.copy()


def get_incompatible_patterns() -> list[str]:
    """
    Get the list of ATS-incompatible formatting patterns.

    Returns:
        List of regex patterns that should be avoided
    """
    return INCOMPATIBLE_PATTERNS.copy()


# ==============================================================================
# Testing Block
# ==============================================================================

if __name__ == "__main__":
    """
    Test the ATS validation tools.
    Run this script directly to verify the tools work correctly.
    """
    print("=" * 70)
    print("ATS Validation Tools - Test")
    print("=" * 70)

    # Test keyword density
    print("\n--- Testing Keyword Density Tool ---")
    sample_text = """
    Senior Software Engineer with 8+ years of experience in Python, AWS, and Docker.
    Expert in building scalable microservices using Python frameworks like FastAPI and Flask.
    Extensive experience with AWS cloud services including EC2, S3, Lambda, and DynamoDB.
    Proficient in Docker containerization and Kubernetes orchestration.
    """

    sample_keywords = ["Python", "AWS", "Docker", "FastAPI", "Kubernetes", "Microservices"]

    # Note: Tools are wrapped with @tool decorator, so we need to call the underlying function
    # For testing, we'll import and call directly
    print("\nSample text word count:", len(sample_text.split()))
    print("Keywords to check:", sample_keywords)
    print("\n[Test would call: calculate_keyword_density(sample_text, sample_keywords)]")

    # Test formatting validation
    print("\n--- Testing ATS Formatting Validation ---")
    print("[Test would call: validate_ats_formatting(sample_text)]")

    # Test section header check
    print("\n--- Testing Section Header Validation ---")
    sample_resume = """
John Doe
email@example.com | 555-0123

Professional Summary
Senior engineer with expertise...

Work Experience
- Software Engineer at Tech Co...

Skills
Python, AWS, Docker

Education
BS in Computer Science
"""
    print("[Test would call: check_section_headers(sample_resume)]")

    # Display utility functions
    print("\n--- Utility Functions ---")
    print(f"Optimal density range: {get_optimal_keyword_density_range()}")
    print(f"Standard headers available: {len(get_standard_headers())} section types")
    print(f"Incompatible patterns: {len(get_incompatible_patterns())} patterns")

    print("\n" + "=" * 70)
    print("Note: Tools are wrapped with @tool decorator for CrewAI usage.")
    print("To test actual tool execution, use them within a CrewAI agent context.")
    print("=" * 70)
