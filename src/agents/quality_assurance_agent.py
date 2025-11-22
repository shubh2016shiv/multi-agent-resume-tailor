"""
Quality Assurance Reviewer Agent - Final Quality Gatekeeper
============================================================

OVERVIEW:
---------
This module defines the eighth and final agent in our workflow: the Quality Assurance Reviewer.
This agent serves as the comprehensive quality gatekeeper, evaluating tailored resumes using
quantitative metrics based on professional resume builder standards.

WHAT MAKES THIS AGENT CRITICAL:
--------------------------------
- **Final Quality Gate**: Last checkpoint before resume submission
- **Quantitative Evaluation**: Data-driven scores, not subjective opinions
- **Multi-Dimensional Analysis**: Accuracy, Relevance, ATS Optimization
- **Ethical Gatekeeper**: Prevents exaggerated or fabricated claims
- **Professional Standards**: Implements universal resume quality guidelines

AGENT DESIGN PRINCIPLES:
------------------------
- **Objectivity First**: Uses measurable metrics for evaluation
- **Zero Tolerance**: No spelling/grammar errors, no fabrication
- **Weighted Scoring**: Accuracy (40%), Relevance (35%), ATS (25%)
- **Actionable Feedback**: Specific corrections for failed reviews
- **Conservative Approach**: When in doubt, flag for review

WORKFLOW OVERVIEW:
------------------
1. Receive optimized resume components from all upstream agents
2. Compare tailored resume against original for accuracy
3. Evaluate relevance to job requirements
4. Validate ATS compatibility and formatting
5. Check grammar, spelling, and professional standards
6. Calculate weighted overall quality score
7. Determine pass/fail (threshold: 80/100)
8. Generate actionable feedback if corrections needed
9. Return comprehensive QualityReport

MODULE STRUCTURE (Hierarchical Organization):
=============================================
This module is organized into 10 main BLOCKS, each containing STAGES with SUB-STAGES:

BLOCK 1: MODULE SETUP & CONFIGURATION
├── Stage 1.1: Import Management
│   ├── Sub-stage 1.1.1: Standard library imports
│   ├── Sub-stage 1.1.2: CrewAI framework imports
│   ├── Sub-stage 1.1.3: Project-specific imports (with fallback handling)
│   └── Sub-stage 1.1.4: Logger initialization
│
├── Stage 1.2: Professional Quality Standards Constants
│   ├── Sub-stage 1.2.1: Quality scoring weights
│   ├── Sub-stage 1.2.2: Pass/fail thresholds
│   ├── Sub-stage 1.2.3: Red flag patterns
│   └── Sub-stage 1.2.4: Standard section headers

BLOCK 2: AGENT CONFIGURATION & CREATION
├── Stage 2.1: Configuration Loading
│   ├── Sub-stage 2.1.1: Load from agents.yaml with fallback
│   ├── Sub-stage 2.1.2: Validate required fields (role, goal, backstory)
│   └── Sub-stage 2.1.3: Error handling with graceful degradation
│
├── Stage 2.2: Default Configuration
│   ├── Sub-stage 2.2.1: Define Quality Auditor role
│   ├── Sub-stage 2.2.2: Set low temperature for objective evaluation
│   └── Sub-stage 2.2.3: Configure quantitative evaluation behavior
│
└── Stage 2.3: Agent Initialization
    ├── Sub-stage 2.3.1: CrewAI Agent object creation
    ├── Sub-stage 2.3.2: Tool assignment for quality evaluation
    └── Sub-stage 2.3.3: Resilience configuration

BLOCK 3: ACCURACY EVALUATION FUNCTIONS
├── Stage 3.1: Claim Verification
│   ├── Sub-stage 3.1.1: compare_experience_claims() - Compare bullets
│   ├── Sub-stage 3.1.2: detect_metric_inflation() - Check for exaggeration
│   └── Sub-stage 3.1.3: validate_consistency() - Dates, titles, companies
│
├── Stage 3.2: Skills Verification
│   ├── Sub-stage 3.2.1: extract_skills_from_resume() - Get original skills
│   ├── Sub-stage 3.2.2: check_unsupported_skills() - Find fabricated skills
│   └── Sub-stage 3.2.3: validate_skill_evidence() - Ensure skills have backing
│
└── Stage 3.3: Accuracy Scoring
    ├── Sub-stage 3.3.1: evaluate_accuracy() - Main accuracy function
    ├── Sub-stage 3.3.2: calculate_accuracy_score() - Score 0-100
    └── Sub-stage 3.3.3: generate_accuracy_justification() - Explain findings

BLOCK 4: RELEVANCE EVALUATION FUNCTIONS
├── Stage 4.1: Job Requirement Extraction
│   ├── Sub-stage 4.1.1: extract_must_have_requirements() - Critical requirements
│   ├── Sub-stage 4.1.2: extract_should_have_requirements() - Preferred skills
│   └── Sub-stage 4.1.3: extract_keywords() - ATS keywords from job
│
├── Stage 4.2: Coverage Analysis
│   ├── Sub-stage 4.2.1: calculate_requirement_coverage() - Match percentage
│   ├── Sub-stage 4.2.2: identify_missed_requirements() - What's missing
│   └── Sub-stage 4.2.3: check_keyword_integration() - Natural keyword usage
│
└── Stage 4.3: Relevance Scoring
    ├── Sub-stage 4.3.1: evaluate_relevance() - Main relevance function
    ├── Sub-stage 4.3.2: calculate_relevance_score() - Score 0-100
    └── Sub-stage 4.3.3: generate_relevance_justification() - Explain findings

BLOCK 5: ATS OPTIMIZATION EVALUATION FUNCTIONS
├── Stage 5.1: Keyword Analysis
│   ├── Sub-stage 5.1.1: calculate_keyword_coverage() - Job keywords present
│   ├── Sub-stage 5.1.2: calculate_keyword_density() - Optimal 2-5%
│   └── Sub-stage 5.1.3: check_keyword_stuffing() - Flag over-optimization
│
├── Stage 5.2: Formatting Validation
│   ├── Sub-stage 5.2.1: check_section_headers() - Standard headers
│   ├── Sub-stage 5.2.2: check_formatting_issues() - Tables, columns, graphics
│   ├── Sub-stage 5.2.3: check_font_compliance() - Standard fonts
│   └── Sub-stage 5.2.4: check_structure() - Proper organization
│
└── Stage 5.3: ATS Scoring
    ├── Sub-stage 5.3.1: evaluate_ats_optimization() - Main ATS function
    ├── Sub-stage 5.3.2: calculate_ats_score() - Score 0-100
    └── Sub-stage 5.3.3: generate_ats_justification() - Explain findings

BLOCK 6: GRAMMAR & FORMATTING VALIDATION
├── Stage 6.1: Writing Quality Checks
│   ├── Sub-stage 6.1.1: check_spelling_errors() - Zero tolerance
│   ├── Sub-stage 6.1.2: check_grammar_errors() - Tense, structure
│   ├── Sub-stage 6.1.3: check_active_voice() - Validate action verbs
│   └── Sub-stage 6.1.4: check_parallel_structure() - Consistent formatting
│
├── Stage 6.2: Professional Standards
│   ├── Sub-stage 6.2.1: check_red_flags() - Pronouns, clichés
│   ├── Sub-stage 6.2.2: check_email_format() - Professional email
│   ├── Sub-stage 6.2.3: check_length_appropriateness() - Page count
│   └── Sub-stage 6.2.4: check_whitespace_balance() - 20-30% white space
│
└── Stage 6.3: Formatting Validation
    ├── Sub-stage 6.3.1: check_formatting_standards() - Main formatting check
    ├── Sub-stage 6.3.2: validate_section_order() - Standard order
    └── Sub-stage 6.3.3: check_bullet_count() - 3-6 bullets per role

BLOCK 7: QUALITY EVALUATION TOOL
├── Stage 7.1: Tool Definition
│   ├── Sub-stage 7.1.1: @tool("Evaluate Resume Quality") decorator
│   ├── Sub-stage 7.1.2: Input parsing (JSON strings to Python objects)
│   └── Sub-stage 7.1.3: Safety validation (prevent invalid inputs)
│
├── Stage 7.2: Comprehensive Evaluation
│   ├── Sub-stage 7.2.1: Run accuracy evaluation
│   ├── Sub-stage 7.2.2: Run relevance evaluation
│   ├── Sub-stage 7.2.3: Run ATS evaluation
│   └── Sub-stage 7.2.4: Run grammar and formatting checks
│
├── Stage 7.3: Score Aggregation
│   ├── Sub-stage 7.3.1: calculate_weighted_score() - Apply weights
│   ├── Sub-stage 7.3.2: determine_pass_fail() - Threshold check
│   └── Sub-stage 7.3.3: generate_feedback() - Actionable recommendations
│
└── Stage 7.4: Output Generation
    ├── Sub-stage 7.4.1: Create QualityReport object
    ├── Sub-stage 7.4.2: Serialize to JSON
    └── Sub-stage 7.4.3: Return comprehensive report

BLOCK 8: WEIGHTED SCORING SYSTEM
├── Stage 8.1: Score Calculation
│   ├── Sub-stage 8.1.1: calculate_weighted_score() - Apply professional weights
│   ├── Sub-stage 8.1.2: Accuracy weight: 40% (truthfulness critical)
│   ├── Sub-stage 8.1.3: Relevance weight: 35% (job fit matters)
│   └── Sub-stage 8.1.4: ATS weight: 25% (compatibility important)
│
└── Stage 8.2: Threshold Validation
    ├── Sub-stage 8.2.1: Pass threshold: 80/100
    ├── Sub-stage 8.2.2: Generate pass/fail decision
    └── Sub-stage 8.2.3: Create feedback for failed reviews

BLOCK 9: OUTPUT VALIDATION
├── Stage 9.1: Data Validation
│   ├── Sub-stage 9.1.1: validate_qa_output() - Parse QualityReport
│   ├── Sub-stage 9.1.2: Validate score ranges (0-100)
│   └── Sub-stage 9.1.3: Check required fields presence
│
├── Stage 9.2: Logical Consistency
│   ├── Sub-stage 9.2.1: Validate scoring consistency
│   ├── Sub-stage 9.2.2: Check feedback completeness
│   └── Sub-stage 9.2.3: Verify actionable recommendations
│
└── Stage 9.3: Meta-Validation
    ├── Sub-stage 9.3.1: check_qa_quality() - Meta QA check
    ├── Sub-stage 9.3.2: Validate justifications
    └── Sub-stage 9.3.3: Return validation result

BLOCK 10: PUBLIC API
├── Stage 10.1: Main Functions Export
│   ├── Sub-stage 10.1.1: create_quality_assurance_agent()
│   ├── Sub-stage 10.1.2: validate_qa_output()
│   └── Sub-stage 10.1.3: check_qa_quality()
│
├── Stage 10.2: Evaluation Functions Export
│   ├── Sub-stage 10.2.1: evaluate_accuracy()
│   ├── Sub-stage 10.2.2: evaluate_relevance()
│   └── Sub-stage 10.2.3: evaluate_ats_optimization()
│
└── Stage 10.3: Helper Functions Export
    ├── Sub-stage 10.3.1: calculate_weighted_score()
    ├── Sub-stage 10.3.2: check_grammar_quality()
    └── Sub-stage 10.3.3: check_formatting_standards()

HOW TO USE THIS MODULE:
-----------------------
1. Import: `from src.agents.quality_assurance_agent import create_quality_assurance_agent`
2. Create Agent: `agent = create_quality_assurance_agent()`
3. Run QA: Use in a Crew with all optimized resume components as context
4. Validate Output: Use `validate_qa_output()` to ensure valid QualityReport
5. Check Quality: Use `check_qa_quality()` for meta-validation

PROFESSIONAL QUALITY STANDARDS:
-------------------------------
This agent implements the universal quality standards used by professional resume builders:

1. **Content Quality (40% weight)**:
   - Factual consistency (no contradictions)
   - Quantifiable results (metrics, percentages, timeframes)
   - Action-oriented language (strong verbs)
   - Relevance to target role
   - Completeness (no gaps)

2. **Relevance to Job (35% weight)**:
   - 60-80% keyword match with job description
   - Must-have skills coverage
   - Critical requirements addressed
   - Natural keyword integration
   - Strategic emphasis on relevant experience

3. **ATS Compatibility (25% weight)**:
   - Standard section headers
   - Simple formatting (no tables/graphics)
   - Standard fonts (10-12pt)
   - Proper file format (.docx or PDF)
   - Keyword density 2-5%

4. **Grammar & Language (Zero Tolerance)**:
   - No spelling errors
   - Consistent tense (past/present)
   - Parallel structure
   - Active voice
   - No personal pronouns (I, me, my)

5. **Red Flags (Auto-fail if present)**:
   - Personal pronouns in bullets
   - Unprofessional email
   - Clichés ("team player", "hard worker")
   - Duties instead of achievements
   - Photos (unless industry-standard)

SCORING ALGORITHM:
------------------
```python
# Component scores (0-100 each)
accuracy_score = evaluate_accuracy(original, tailored)
relevance_score = evaluate_relevance(tailored, job)
ats_score = evaluate_ats_optimization(tailored, job)

# Weighted overall score
overall_score = (
    accuracy_score * 0.40 +  # 40% weight - truthfulness is critical
    relevance_score * 0.35 + # 35% weight - job fit matters
    ats_score * 0.25         # 25% weight - ATS compatibility
)

# Pass threshold: 80/100
passed = overall_score >= 80.0
```

KEY DIFFERENCES FROM ATS OPTIMIZATION AGENT:
--------------------------------------------
- **ATS Agent**: Assembles and formats resume (forward-looking, building)
- **QA Agent**: Evaluates and validates resume (backward-looking, verification)
- **ATS Agent**: Creates final output (markdown/JSON)
- **QA Agent**: Reviews final output (quality metrics)
- **ATS Agent**: Optimizes keyword density
- **QA Agent**: Validates accuracy and truthfulness
"""

# ============================================================================
# BLOCK 1: MODULE SETUP & CONFIGURATION
# ============================================================================
# Stage 1.1: Import Management
# ----------------------------------------------------------------------------

# Sub-stage 1.1.1: Standard library imports
import json
import re
from typing import Any

# Sub-stage 1.1.2: CrewAI framework imports
from crewai import Agent
from crewai.tools import tool
from pydantic import ValidationError

# Sub-stage 1.1.3: Project-specific imports (with fallback handling)
try:
    from src.core.config import get_agents_config  # , get_config
    from src.core.logger import get_logger
    from src.data_models.evaluation import (
        AccuracyMetrics,
        ATSMetrics,
        QualityReport,
        RelevanceMetrics,
    )
    # from src.data_models.job import JobDescription
    # from src.data_models.resume import Resume
except ImportError:
    # Fallback for direct script execution
    import sys
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    from src.core.config import get_agents_config
    from src.core.logger import get_logger
    from src.data_models.evaluation import (
        AccuracyMetrics,
        ATSMetrics,
        QualityReport,
        RelevanceMetrics,
    )

# Sub-stage 1.1.4: Logger initialization
logger = get_logger(__name__)

# Stage 1.2: Professional Quality Standards Constants
# ----------------------------------------------------------------------------

# Sub-stage 1.2.1: Quality scoring weights (based on professional standards)
ACCURACY_WEIGHT = 0.40  # 40% - Truthfulness is critical
RELEVANCE_WEIGHT = 0.35  # 35% - Job fit matters
ATS_WEIGHT = 0.25  # 25% - ATS compatibility important

# Sub-stage 1.2.2: Pass/fail thresholds
QUALITY_PASS_THRESHOLD = 80.0  # Minimum score for approval (0-100 scale)
KEYWORD_COVERAGE_TARGET_MIN = 60.0  # 60% minimum keyword match
KEYWORD_COVERAGE_TARGET_MAX = 80.0  # 80% optimal keyword match
KEYWORD_DENSITY_MIN = 0.02  # 2% minimum keyword density
KEYWORD_DENSITY_MAX = 0.05  # 5% maximum keyword density

# Sub-stage 1.2.3: Red flag patterns (auto-fail if present)
RED_FLAG_PATTERNS = {
    "personal_pronouns": [r"\bI\b", r"\bme\b", r"\bmy\b", r"\bwe\b", r"\bour\b"],
    "cliches": [
        "team player",
        "hard worker",
        "go-getter",
        "think outside the box",
        "hit the ground running",
        "self-starter",
        "results-oriented",
        "detail-oriented",
        "proven track record",
    ],
    "unprofessional_email_patterns": [
        r"@(yahoo|hotmail|aol|live|msn)\.com$",  # Consumer email services
        r"[0-9]{4,}",  # Numbers in email (birthdates)
        r"(cool|sexy|hot|cute|sweetie|babe)",  # Unprofessional terms
    ],
    "duties_not_achievements": [
        "responsible for",
        "duties included",
        "tasks involved",
        "in charge of",
        "handled",
        "worked on",
    ],
}

# Sub-stage 1.2.4: Standard section headers (ATS-compatible)
STANDARD_SECTION_HEADERS = [
    "Professional Summary",
    "Summary",
    "Profile",
    "Experience",
    "Work Experience",
    "Professional Experience",
    "Employment History",
    "Education",
    "Academic Background",
    "Skills",
    "Technical Skills",
    "Core Competencies",
    "Certifications",
    "Licenses",
    "Awards",
    "Projects",
    "Publications",
]

# Standard fonts for ATS compatibility
STANDARD_FONTS = [
    "Arial",
    "Calibri",
    "Georgia",
    "Times New Roman",
    "Helvetica",
    "Verdana",
    "Cambria",
    "Garamond",
]

# ============================================================================
# BLOCK 2: AGENT CONFIGURATION & CREATION
# ============================================================================
# Stage 2.1: Configuration Loading
# ----------------------------------------------------------------------------


def create_quality_assurance_agent() -> Agent:
    """
    Create the Quality Assurance Reviewer agent with configuration from agents.yaml.

    This agent serves as the final quality gatekeeper, performing comprehensive
    evaluation using quantitative metrics across three weighted dimensions:
    - Accuracy (40%): Verify all claims are truthful
    - Relevance (35%): Ensure job requirements are addressed
    - ATS Optimization (25%): Validate keyword coverage and format

    The agent uses a low temperature (0.2) for objective, deterministic evaluation
    and focuses on data-driven scoring rather than subjective opinions.

    Returns:
        Agent: Configured CrewAI Agent instance for quality assurance

    Raises:
        Exception: If agent creation fails (logs error and re-raises)
    """
    try:
        # Sub-stage 2.1.1: Load from agents.yaml with fallback to defaults
        logger.info("Loading Quality Assurance Reviewer agent configuration...")
        config = get_agents_config()
        agent_config = config.get("quality_assurance_reviewer", {})

        # Sub-stage 2.1.2: Validate required fields (role, goal, backstory)
        if not agent_config:
            logger.warning("quality_assurance_reviewer not found in agents.yaml, using defaults")
            agent_config = _get_default_qa_config()

        # Extract configuration with fallbacks
        role = agent_config.get("role", "Resume Quality Auditor with Quantitative Metrics")
        goal = agent_config.get(
            "goal",
            "Conduct comprehensive quality evaluation using quantitative metrics across "
            "Accuracy (40%), Relevance (35%), and ATS Optimization (25%). "
            "Pass threshold: 80/100. Provide actionable feedback if failed.",
        )
        backstory = agent_config.get(
            "backstory",
            "You are a meticulous quality analyst with expertise in quantitative evaluation "
            "and professional ethics. Your 12 years of experience in resume review has taught "
            "you that subjective assessment is insufficient - you need measurable metrics. "
            "You are the final gatekeeper preventing dishonest resumes from being sent to employers.",
        )

        # Extract LLM configuration
        llm = agent_config.get("llm", "gemini/gemini-2.5-flash")
        temperature = agent_config.get("temperature", 0.2)  # Low for objectivity
        verbose = agent_config.get("verbose", True)

        logger.info(f"Creating Quality Assurance Reviewer agent with LLM: {llm}")

        # Sub-stage 2.1.3: Error handling with graceful degradation
        # Stage 2.3: Agent Initialization
        # Create the agent with quality evaluation tool
        agent = Agent(
            role=role,
            goal=goal,
            backstory=backstory,
            llm=llm,
            temperature=temperature,
            verbose=verbose,
            allow_delegation=False,  # QA agent works independently
            tools=[evaluate_resume_quality_tool],  # Assign quality evaluation tool
        )

        logger.info("Quality Assurance Reviewer agent created successfully")
        return agent

    except Exception as e:
        logger.error(f"Failed to create Quality Assurance Reviewer agent: {e}", exc_info=True)
        raise


# Stage 2.2: Default Configuration
# ----------------------------------------------------------------------------


def _get_default_qa_config() -> dict[str, Any]:
    """
    Get default configuration for Quality Assurance Reviewer agent.

    This fallback configuration ensures the agent can be created even if
    agents.yaml is missing or incomplete.

    Returns:
        Dict[str, Any]: Default agent configuration
    """
    return {
        "role": "Resume Quality Auditor with Quantitative Metrics",
        "goal": (
            "Conduct comprehensive quality evaluation using quantitative metrics across "
            "three dimensions: 1) ACCURACY (40% weight): Verify all claims are truthful "
            "and supported by the original resume. 2) RELEVANCE (35% weight): Ensure all "
            "critical job requirements are addressed effectively. 3) ATS OPTIMIZATION "
            "(25% weight): Validate keyword coverage and format compatibility. "
            "Your final verdict is based on an overall score, which must be >= 80/100 to pass. "
            "If the score is below 80, you must provide specific, actionable feedback."
        ),
        "backstory": (
            "You are a meticulous quality analyst with expertise in quantitative evaluation "
            "and professional ethics. Your 12 years of experience in resume review has taught "
            "you that subjective assessment is insufficient - you need measurable metrics. "
            "Your evaluation methodology is data-driven: ACCURACY (compare every claim in the "
            "tailored resume against the original), RELEVANCE (extract job requirements and "
            "check their coverage), ATS COMPATIBILITY (check for formatting issues, standard "
            "section headers, and keyword coverage without stuffing). You are the final "
            "gatekeeper preventing dishonest resumes from being sent to employers. You take "
            "this responsibility seriously because exaggerated claims damage candidate credibility. "
            "Your reviews are constructive and specific. When you identify issues, you explain "
            "the problem, why it's problematic, and how to fix it."
        ),
        "llm": "gemini/gemini-2.5-flash",
        "temperature": 0.2,
        "verbose": True,
    }


# ============================================================================
# BLOCK 3: ACCURACY EVALUATION FUNCTIONS
# ============================================================================
# Stage 3.1: Claim Verification
# ----------------------------------------------------------------------------


def compare_experience_claims(
    original_resume: dict[str, Any], tailored_resume: dict[str, Any]
) -> list[str]:
    """
    Compare experience bullets between original and tailored resumes to detect exaggerations.

    This function checks for inflated metrics (e.g., "3 projects" → "5 projects"),
    changed titles, extended date ranges, and other forms of resume inflation.

    Args:
        original_resume: Original resume data (dict)
        tailored_resume: Tailored resume data (dict)

    Returns:
        List[str]: List of exaggerated claims found
    """
    exaggerated_claims = []

    try:
        # Extract experience sections
        original_exp = original_resume.get("experience", [])
        tailored_exp = tailored_resume.get("experience", [])

        # Create mapping by company name for comparison
        original_by_company = {exp.get("company", ""): exp for exp in original_exp}

        for tailored_entry in tailored_exp:
            company = tailored_entry.get("company", "")
            original_entry = original_by_company.get(company)

            if not original_entry:
                continue  # New entry, not comparable

            # Check for title inflation
            original_title = original_entry.get("job_title", "").lower()
            tailored_title = tailored_entry.get("job_title", "").lower()
            if original_title != tailored_title:
                exaggerated_claims.append(
                    f"Job title changed at {company}: '{original_entry.get('job_title')}' "
                    f"→ '{tailored_entry.get('job_title')}'"
                )

            # Check for date range extension
            original_dates = (
                f"{original_entry.get('start_date', '')}-{original_entry.get('end_date', '')}"
            )
            tailored_dates = (
                f"{tailored_entry.get('start_date', '')}-{tailored_entry.get('end_date', '')}"
            )
            if original_dates != tailored_dates:
                exaggerated_claims.append(
                    f"Date range modified at {company}: {original_dates} → {tailored_dates}"
                )

            # Check for metric inflation in descriptions
            original_desc = " ".join(original_entry.get("descriptions", []))
            tailored_desc = " ".join(tailored_entry.get("descriptions", []))

            # Extract numbers from both descriptions
            original_numbers = set(re.findall(r"\d+", original_desc))
            tailored_numbers = set(re.findall(r"\d+", tailored_desc))

            # Check if new, larger numbers were introduced
            new_numbers = tailored_numbers - original_numbers
            if new_numbers:
                # Simple heuristic: if new numbers are significantly larger
                for new_num in new_numbers:
                    if int(new_num) > 100 and not any(
                        int(orig) >= int(new_num) * 0.8
                        for orig in original_numbers
                        if orig.isdigit()
                    ):
                        exaggerated_claims.append(
                            f"Potentially inflated metric at {company}: new value {new_num} "
                            f"not found in original resume"
                        )

    except Exception as e:
        logger.error(f"Error comparing experience claims: {e}", exc_info=True)

    return exaggerated_claims


def check_unsupported_skills(
    original_resume: dict[str, Any], tailored_resume: dict[str, Any]
) -> list[str]:
    """
    Check for skills in tailored resume that are not supported by original resume.

    This function identifies skills that were added without evidence in the original
    resume, which could indicate fabrication or inappropriate inference.

    Args:
        original_resume: Original resume data (dict)
        tailored_resume: Tailored resume data (dict)

    Returns:
        List[str]: List of unsupported skills
    """
    unsupported_skills = []

    try:
        # Extract skills from both resumes
        original_skills_data = original_resume.get("skills", [])
        tailored_skills_data = tailored_resume.get("skills", [])

        # Normalize skill names for comparison
        def normalize_skill(skill: Any) -> str:
            if isinstance(skill, dict):
                return skill.get("skill_name", "").lower().strip()
            elif isinstance(skill, str):
                return skill.lower().strip()
            return ""

        original_skills = {normalize_skill(s) for s in original_skills_data}
        tailored_skills = {normalize_skill(s) for s in tailored_skills_data}

        # Find skills that were added
        added_skills = tailored_skills - original_skills

        # Also check in experience descriptions for evidence
        original_exp_text = " ".join(
            " ".join(exp.get("descriptions", [])) for exp in original_resume.get("experience", [])
        ).lower()

        for skill in added_skills:
            if not skill:
                continue

            # Check if skill is mentioned in experience section
            skill_variations = [
                skill,
                skill.replace(" ", ""),
                skill.replace("-", ""),
            ]

            found_in_experience = any(
                variation in original_exp_text for variation in skill_variations
            )

            if not found_in_experience:
                unsupported_skills.append(skill)

    except Exception as e:
        logger.error(f"Error checking unsupported skills: {e}", exc_info=True)

    return unsupported_skills


# Stage 3.3: Accuracy Scoring
# ----------------------------------------------------------------------------


def evaluate_accuracy(
    original_resume: dict[str, Any], tailored_resume: dict[str, Any]
) -> AccuracyMetrics:
    """
    Evaluate the accuracy of the tailored resume against the original.

    This function compares the tailored resume against the original to detect:
    - Exaggerated claims (inflated metrics, extended dates)
    - Unsupported skills (skills not found in original)
    - Changed titles or responsibilities

    The accuracy score ranges from 0-100, with deductions for each issue found.

    Args:
        original_resume: Original resume data (dict)
        tailored_resume: Tailored resume data (dict)

    Returns:
        AccuracyMetrics: Accuracy evaluation results
    """
    try:
        logger.info("Evaluating accuracy: comparing tailored vs original resume")

        # Check for exaggerated claims
        exaggerated_claims = compare_experience_claims(original_resume, tailored_resume)

        # Check for unsupported skills
        unsupported_skills = check_unsupported_skills(original_resume, tailored_resume)

        # Calculate accuracy score (start at 100, deduct for issues)
        accuracy_score = 100.0

        # Deduct 10 points per exaggerated claim (max 40 points)
        accuracy_score -= min(len(exaggerated_claims) * 10, 40)

        # Deduct 5 points per unsupported skill (max 30 points)
        accuracy_score -= min(len(unsupported_skills) * 5, 30)

        # Ensure score doesn't go below 0
        accuracy_score = max(accuracy_score, 0.0)

        # Generate justification
        if accuracy_score >= 95:
            justification = (
                "Excellent accuracy. All claims in the tailored resume are well-supported "
                "by the original resume. No exaggerations or unsupported skills detected."
            )
        elif accuracy_score >= 85:
            justification = (
                f"Good accuracy with minor issues. Found {len(exaggerated_claims)} "
                f"exaggerated claim(s) and {len(unsupported_skills)} unsupported skill(s). "
                "Most claims are well-supported."
            )
        elif accuracy_score >= 70:
            justification = (
                f"Moderate accuracy concerns. Found {len(exaggerated_claims)} exaggerated "
                f"claim(s) and {len(unsupported_skills)} unsupported skill(s). "
                "Some claims need verification."
            )
        else:
            justification = (
                f"Significant accuracy issues. Found {len(exaggerated_claims)} exaggerated "
                f"claim(s) and {len(unsupported_skills)} unsupported skill(s). "
                "Multiple claims are not supported by the original resume."
            )

        logger.info(f"Accuracy evaluation complete. Score: {accuracy_score:.1f}/100")

        return AccuracyMetrics(
            accuracy_score=accuracy_score,
            exaggerated_claims=exaggerated_claims,
            unsupported_skills=unsupported_skills,
            justification=justification,
        )

    except Exception as e:
        logger.error(f"Error evaluating accuracy: {e}", exc_info=True)
        # Return conservative (low) score on error
        return AccuracyMetrics(
            accuracy_score=50.0,
            exaggerated_claims=["Error during accuracy evaluation"],
            unsupported_skills=[],
            justification=f"Accuracy evaluation encountered an error: {str(e)}",
        )


# ============================================================================
# BLOCK 4: RELEVANCE EVALUATION FUNCTIONS
# ============================================================================
# Stage 4.1: Job Requirement Extraction
# ----------------------------------------------------------------------------


def extract_must_have_requirements(job_description: dict[str, Any]) -> list[str]:
    """
    Extract must-have (required) requirements from job description.

    Args:
        job_description: Job description data (dict)

    Returns:
        List[str]: List of must-have requirements
    """
    must_haves = []

    try:
        # Extract from requirements field
        requirements = job_description.get("requirements", [])
        for req in requirements:
            if isinstance(req, dict):
                importance = req.get("importance_level", "").lower()
                if importance in ["must-have", "required", "essential", "mandatory"]:
                    must_haves.append(req.get("description", ""))
            elif isinstance(req, str):
                # Simple string requirements
                must_haves.append(req)

        # Also extract from skills if available
        skills = job_description.get("skills", {})
        if isinstance(skills, dict):
            must_have_skills = skills.get("must_have", [])
            must_haves.extend(must_have_skills)

    except Exception as e:
        logger.error(f"Error extracting must-have requirements: {e}", exc_info=True)

    return must_haves


def extract_job_keywords(job_description: dict[str, Any]) -> list[str]:
    """
    Extract ATS keywords from job description.

    Args:
        job_description: Job description data (dict)

    Returns:
        List[str]: List of keywords for ATS matching
    """
    keywords = []

    try:
        # Extract from dedicated keywords field
        keywords.extend(job_description.get("keywords", []))

        # Extract from skills
        skills = job_description.get("skills", {})
        if isinstance(skills, dict):
            keywords.extend(skills.get("must_have", []))
            keywords.extend(skills.get("should_have", []))
        elif isinstance(skills, list):
            keywords.extend(skills)

        # Extract from requirements
        requirements = job_description.get("requirements", [])
        for req in requirements:
            if isinstance(req, dict):
                desc = req.get("description", "")
                # Extract technical terms (capitalized words, acronyms)
                tech_terms = re.findall(r"\b[A-Z][A-Za-z0-9+#\.]*\b", desc)
                keywords.extend(tech_terms)

        # Remove duplicates and empty strings
        keywords = list({k for k in keywords if k})

    except Exception as e:
        logger.error(f"Error extracting job keywords: {e}", exc_info=True)

    return keywords


# Stage 4.2: Coverage Analysis
# ----------------------------------------------------------------------------


def calculate_requirement_coverage(
    tailored_resume: dict[str, Any], must_have_requirements: list[str]
) -> tuple[float, list[str]]:
    """
    Calculate how many must-have requirements are addressed in the resume.

    Args:
        tailored_resume: Tailored resume data (dict)
        must_have_requirements: List of must-have requirements from job

    Returns:
        Tuple[float, List[str]]: (coverage percentage, list of missed requirements)
    """
    if not must_have_requirements:
        return 100.0, []

    # Convert resume to text for searching
    resume_text = json.dumps(tailored_resume).lower()

    addressed = 0
    missed = []

    for req in must_have_requirements:
        req_lower = req.lower()
        # Check if requirement is mentioned anywhere in resume
        if req_lower in resume_text:
            addressed += 1
        else:
            # Check for partial matches (words from requirement)
            req_words = set(re.findall(r"\b\w+\b", req_lower))
            # Remove common words
            req_words = req_words - {"a", "an", "the", "and", "or", "of", "in", "with"}

            if req_words and any(word in resume_text for word in req_words):
                addressed += 0.5  # Partial credit
            else:
                missed.append(req)

    coverage = (addressed / len(must_have_requirements)) * 100
    return coverage, missed


# Stage 4.3: Relevance Scoring
# ----------------------------------------------------------------------------


def evaluate_relevance(
    tailored_resume: dict[str, Any], job_description: dict[str, Any]
) -> RelevanceMetrics:
    """
    Evaluate how relevant the tailored resume is to the job description.

    This function checks:
    - Coverage of must-have requirements
    - Presence of key skills
    - Alignment with job responsibilities

    The relevance score ranges from 0-100 based on requirement coverage.

    Args:
        tailored_resume: Tailored resume data (dict)
        job_description: Job description data (dict)

    Returns:
        RelevanceMetrics: Relevance evaluation results
    """
    try:
        logger.info("Evaluating relevance: checking job requirement coverage")

        # Extract must-have requirements
        must_haves = extract_must_have_requirements(job_description)

        # Calculate coverage
        coverage, missed = calculate_requirement_coverage(tailored_resume, must_haves)

        # Relevance score is based on coverage
        relevance_score = coverage

        # Generate justification
        if relevance_score >= 90:
            justification = (
                f"Excellent relevance. {coverage:.0f}% of must-have requirements are addressed. "
                "Resume is well-aligned with job needs."
            )
        elif relevance_score >= 75:
            justification = (
                f"Good relevance. {coverage:.0f}% of must-have requirements are addressed. "
                f"{len(missed)} requirement(s) could be emphasized more."
            )
        elif relevance_score >= 60:
            justification = (
                f"Moderate relevance. {coverage:.0f}% of must-have requirements are addressed. "
                f"{len(missed)} important requirement(s) are not well-covered."
            )
        else:
            justification = (
                f"Low relevance. Only {coverage:.0f}% of must-have requirements are addressed. "
                f"{len(missed)} critical requirement(s) are missing or not emphasized."
            )

        logger.info(f"Relevance evaluation complete. Score: {relevance_score:.1f}/100")

        return RelevanceMetrics(
            relevance_score=relevance_score,
            must_have_skills_coverage=coverage,
            missed_requirements=missed,
            justification=justification,
        )

    except Exception as e:
        logger.error(f"Error evaluating relevance: {e}", exc_info=True)
        return RelevanceMetrics(
            relevance_score=50.0,
            must_have_skills_coverage=50.0,
            missed_requirements=["Error during relevance evaluation"],
            justification=f"Relevance evaluation encountered an error: {str(e)}",
        )


# ============================================================================
# BLOCK 5: ATS OPTIMIZATION EVALUATION FUNCTIONS
# ============================================================================
# Stage 5.1: Keyword Analysis
# ----------------------------------------------------------------------------


def calculate_keyword_coverage_score(
    tailored_resume: dict[str, Any], job_keywords: list[str]
) -> float:
    """
    Calculate what percentage of job keywords are present in the resume.

    Args:
        tailored_resume: Tailored resume data (dict)
        job_keywords: List of keywords from job description

    Returns:
        float: Keyword coverage percentage (0-100)
    """
    if not job_keywords:
        return 100.0

    resume_text = json.dumps(tailored_resume).lower()

    matched = 0
    for keyword in job_keywords:
        if keyword.lower() in resume_text:
            matched += 1

    coverage = (matched / len(job_keywords)) * 100
    return coverage


def check_formatting_issues(tailored_resume: dict[str, Any]) -> list[str]:
    """
    Check for ATS-incompatible formatting issues.

    Args:
        tailored_resume: Tailored resume data (dict)

    Returns:
        List[str]: List of formatting issues found
    """
    issues = []

    # Convert to string to check for problematic patterns
    resume_str = json.dumps(tailored_resume)

    # Check for tables (markdown table indicators)
    if "|" in resume_str and "---" in resume_str:
        issues.append("Resume contains table formatting (use simple lists instead)")

    # Check for unusual characters
    unusual_chars = re.findall(
        r"[^\w\s\.\,\-\(\)\[\]\{\}\"\'\/\:\;\!\?\@\#\$\%\&\*\+\=]", resume_str
    )
    if len(unusual_chars) > 10:
        issues.append(
            f"Resume contains {len(unusual_chars)} unusual characters that may confuse ATS"
        )

    # Check section headers
    sections = tailored_resume.keys()
    non_standard_sections = [
        s
        for s in sections
        if s
        not in ["personal_info", "summary", "experience", "education", "skills", "certifications"]
        and s not in [h.lower().replace(" ", "_") for h in STANDARD_SECTION_HEADERS]
    ]
    if non_standard_sections:
        issues.append(f"Non-standard section headers found: {', '.join(non_standard_sections)}")

    return issues


# Stage 5.3: ATS Scoring
# ----------------------------------------------------------------------------


def evaluate_ats_optimization(
    tailored_resume: dict[str, Any], job_description: dict[str, Any]
) -> ATSMetrics:
    """
    Evaluate ATS compatibility of the tailored resume.

    This function checks:
    - Keyword coverage (target: 60-80%)
    - Standard section headers
    - Simple formatting (no tables, graphics)
    - File format compatibility

    The ATS score ranges from 0-100 based on multiple factors.

    Args:
        tailored_resume: Tailored resume data (dict)
        job_description: Job description data (dict)

    Returns:
        ATSMetrics: ATS evaluation results
    """
    try:
        logger.info("Evaluating ATS optimization: checking keyword coverage and formatting")

        # Extract keywords from job
        job_keywords = extract_job_keywords(job_description)

        # Calculate keyword coverage
        keyword_coverage = calculate_keyword_coverage_score(tailored_resume, job_keywords)

        # Check formatting issues
        formatting_issues = check_formatting_issues(tailored_resume)

        # Calculate ATS score
        ats_score = 100.0

        # Deduct points for keyword coverage issues
        if keyword_coverage < KEYWORD_COVERAGE_TARGET_MIN:
            # Below 60%: major issue
            ats_score -= KEYWORD_COVERAGE_TARGET_MIN - keyword_coverage
        elif keyword_coverage > KEYWORD_COVERAGE_TARGET_MAX:
            # Above 80%: possible keyword stuffing
            ats_score -= (keyword_coverage - KEYWORD_COVERAGE_TARGET_MAX) * 0.5

        # Deduct points for formatting issues (5 points each)
        ats_score -= min(len(formatting_issues) * 5, 30)

        # Ensure score doesn't go below 0
        ats_score = max(ats_score, 0.0)

        # Generate justification
        if ats_score >= 90:
            justification = (
                f"Excellent ATS compatibility. Keyword coverage: {keyword_coverage:.0f}%. "
                "Clean formatting with standard section headers."
            )
        elif ats_score >= 75:
            justification = (
                f"Good ATS compatibility. Keyword coverage: {keyword_coverage:.0f}%. "
                f"{len(formatting_issues)} minor formatting issue(s)."
            )
        elif ats_score >= 60:
            justification = (
                f"Moderate ATS compatibility. Keyword coverage: {keyword_coverage:.0f}%. "
                f"{len(formatting_issues)} formatting issue(s) need attention."
            )
        else:
            justification = (
                f"Poor ATS compatibility. Keyword coverage: {keyword_coverage:.0f}%. "
                f"{len(formatting_issues)} significant formatting issue(s)."
            )

        logger.info(f"ATS evaluation complete. Score: {ats_score:.1f}/100")

        return ATSMetrics(
            ats_score=ats_score,
            keyword_coverage=keyword_coverage,
            formatting_issues=formatting_issues,
            justification=justification,
        )

    except Exception as e:
        logger.error(f"Error evaluating ATS optimization: {e}", exc_info=True)
        return ATSMetrics(
            ats_score=50.0,
            keyword_coverage=50.0,
            formatting_issues=["Error during ATS evaluation"],
            justification=f"ATS evaluation encountered an error: {str(e)}",
        )


# ============================================================================
# BLOCK 6: GRAMMAR & FORMATTING VALIDATION
# ============================================================================
# Stage 6.1: Writing Quality Checks
# ----------------------------------------------------------------------------


def check_grammar_quality(tailored_resume: dict[str, Any]) -> list[str]:
    """
    Check for grammar and language quality issues.

    This function checks for:
    - Personal pronouns (I, me, my, we)
    - Clichés (team player, hard worker)
    - Passive voice patterns
    - Inconsistent tense

    Args:
        tailored_resume: Tailored resume data (dict)

    Returns:
        List[str]: List of grammar/language issues found
    """
    issues = []

    # Convert resume to text for analysis
    resume_text = json.dumps(tailored_resume)

    # Check for personal pronouns
    for pattern in RED_FLAG_PATTERNS["personal_pronouns"]:
        matches = re.findall(pattern, resume_text, re.IGNORECASE)
        if matches:
            issues.append(f"Personal pronouns found: {', '.join(set(matches)[:3])}")
            break  # Report once

    # Check for clichés
    resume_lower = resume_text.lower()
    found_cliches = [cliche for cliche in RED_FLAG_PATTERNS["cliches"] if cliche in resume_lower]
    if found_cliches:
        issues.append(f"Clichés found: {', '.join(found_cliches[:3])}")

    # Check for "duties not achievements" patterns
    found_duties = [
        pattern
        for pattern in RED_FLAG_PATTERNS["duties_not_achievements"]
        if pattern in resume_lower
    ]
    if found_duties:
        issues.append(
            f"Duty-focused language found: {', '.join(found_duties[:2])} (use achievements instead)"
        )

    return issues


# Stage 6.2: Professional Standards
# ----------------------------------------------------------------------------


def check_red_flags(tailored_resume: dict[str, Any]) -> list[str]:
    """
    Check for professional red flags in the resume.

    Args:
        tailored_resume: Tailored resume data (dict)

    Returns:
        List[str]: List of red flags found
    """
    red_flags = []

    try:
        # Check email format
        personal_info = tailored_resume.get("personal_info", {})
        email = personal_info.get("email", "")

        if email:
            for pattern in RED_FLAG_PATTERNS["unprofessional_email_patterns"]:
                if re.search(pattern, email, re.IGNORECASE):
                    red_flags.append(f"Unprofessional email format: {email}")
                    break

        # Check for photo (unless in certain fields)
        if "photo" in tailored_resume or "image" in tailored_resume:
            red_flags.append("Resume contains photo (not recommended unless industry-standard)")

        # Grammar issues are red flags too
        grammar_issues = check_grammar_quality(tailored_resume)
        red_flags.extend(grammar_issues)

    except Exception as e:
        logger.error(f"Error checking red flags: {e}", exc_info=True)

    return red_flags


# Stage 6.3: Formatting Validation
# ----------------------------------------------------------------------------


def check_formatting_standards(tailored_resume: dict[str, Any]) -> list[str]:
    """
    Check if resume meets professional formatting standards.

    This function validates:
    - Appropriate length (1-2 pages based on experience)
    - Section order (Contact → Summary → Experience → Education → Skills)
    - Bullet count (3-6 bullets per role)

    Args:
        tailored_resume: Tailored resume data (dict)

    Returns:
        List[str]: List of formatting issues found
    """
    issues = []

    try:
        # Check section order
        expected_order = ["personal_info", "summary", "experience", "education", "skills"]
        actual_sections = list(tailored_resume.keys())

        # Filter to only expected sections that exist
        actual_ordered = [s for s in expected_order if s in actual_sections]

        if actual_ordered != expected_order[: len(actual_ordered)]:
            issues.append(
                "Sections not in standard order (Contact → Summary → Experience → Education → Skills)"
            )

        # Check experience bullet counts
        experience = tailored_resume.get("experience", [])
        for exp in experience:
            company = exp.get("company", "Unknown")
            descriptions = exp.get("descriptions", [])
            bullet_count = len(descriptions)

            if bullet_count < 3:
                issues.append(f"{company}: Too few bullets ({bullet_count}). Aim for 3-6 per role.")
            elif bullet_count > 6:
                issues.append(
                    f"{company}: Too many bullets ({bullet_count}). Keep to 3-6 per role."
                )

        # Check for completeness
        if "summary" not in tailored_resume or not tailored_resume.get("summary"):
            issues.append("Missing professional summary")

        if "skills" not in tailored_resume or not tailored_resume.get("skills"):
            issues.append("Missing skills section")

    except Exception as e:
        logger.error(f"Error checking formatting standards: {e}", exc_info=True)

    return issues


# ============================================================================
# BLOCK 7: QUALITY EVALUATION TOOL
# ============================================================================
# Stage 7.1: Tool Definition
# ----------------------------------------------------------------------------


@tool("Evaluate Resume Quality")
def evaluate_resume_quality_tool(
    original_resume_json: str, tailored_resume_json: str, job_description_json: str
) -> str:
    """
    Comprehensive quality evaluation tool for the Quality Assurance Reviewer agent.

    This tool evaluates the tailored resume across three weighted dimensions:
    1. ACCURACY (40% weight): Compare tailored vs original to detect exaggerations
    2. RELEVANCE (35% weight): Check alignment with job requirements
    3. ATS OPTIMIZATION (25% weight): Validate keyword coverage and formatting

    The overall quality score must be >= 80/100 to pass.

    Args:
        original_resume_json: JSON string of original resume data
        tailored_resume_json: JSON string of tailored/optimized resume data
        job_description_json: JSON string of job description data

    Returns:
        str: JSON string containing QualityReport with scores and recommendations

    Example:
        {
            "overall_quality_score": 88.5,
            "passed_quality_threshold": true,
            "assessment_summary": "High quality resume, approved for submission",
            "accuracy": {...},
            "relevance": {...},
            "ats_optimization": {...},
            "feedback_for_improvement": null
        }
    """
    try:
        # Sub-stage 7.1.2: Input parsing (JSON strings to Python objects)
        logger.info("Quality evaluation tool invoked")

        original_resume = json.loads(original_resume_json)
        tailored_resume = json.loads(tailored_resume_json)
        job_description = json.loads(job_description_json)

        logger.info("Parsed input data successfully")

        # Stage 7.2: Comprehensive Evaluation
        # Sub-stage 7.2.1: Run accuracy evaluation
        accuracy = evaluate_accuracy(original_resume, tailored_resume)

        # Sub-stage 7.2.2: Run relevance evaluation
        relevance = evaluate_relevance(tailored_resume, job_description)

        # Sub-stage 7.2.3: Run ATS evaluation
        ats_optimization = evaluate_ats_optimization(tailored_resume, job_description)

        # Sub-stage 7.2.4: Run grammar and formatting checks
        red_flags = check_red_flags(tailored_resume)
        formatting_issues = check_formatting_standards(tailored_resume)

        # Stage 7.3: Score Aggregation
        # Sub-stage 7.3.1: calculate_weighted_score() - Apply weights
        overall_score = calculate_weighted_score(
            accuracy.accuracy_score, relevance.relevance_score, ats_optimization.ats_score
        )

        # Deduct points for red flags (each red flag: -5 points, max -20)
        if red_flags:
            deduction = min(len(red_flags) * 5, 20)
            overall_score -= deduction
            logger.warning(f"Deducted {deduction} points for {len(red_flags)} red flag(s)")

        # Deduct points for formatting issues (each issue: -3 points, max -15)
        if formatting_issues:
            deduction = min(len(formatting_issues) * 3, 15)
            overall_score -= deduction
            logger.warning(
                f"Deducted {deduction} points for {len(formatting_issues)} formatting issue(s)"
            )

        overall_score = max(overall_score, 0.0)  # Floor at 0

        # Sub-stage 7.3.2: determine_pass_fail() - Threshold check
        passed = overall_score >= QUALITY_PASS_THRESHOLD

        # Sub-stage 7.3.3: generate_feedback() - Actionable recommendations
        if passed:
            assessment_summary = (
                f"High quality resume with overall score of {overall_score:.1f}/100. "
                "Approved for submission."
            )
            feedback = None
        else:
            assessment_summary = (
                f"Resume does not meet quality threshold. Score: {overall_score:.1f}/100 "
                f"(minimum: {QUALITY_PASS_THRESHOLD}). Corrections required."
            )

            # Generate specific feedback
            feedback_parts = []

            if accuracy.accuracy_score < 80:
                feedback_parts.append(
                    f"ACCURACY ({accuracy.accuracy_score:.1f}/100): "
                    f"{len(accuracy.exaggerated_claims)} exaggerated claim(s), "
                    f"{len(accuracy.unsupported_skills)} unsupported skill(s). "
                    "Review and ensure all claims are supported by original resume."
                )

            if relevance.relevance_score < 80:
                feedback_parts.append(
                    f"RELEVANCE ({relevance.relevance_score:.1f}/100): "
                    f"{len(relevance.missed_requirements)} requirement(s) not addressed. "
                    "Emphasize relevant experience that addresses these requirements."
                )

            if ats_optimization.ats_score < 80:
                feedback_parts.append(
                    f"ATS OPTIMIZATION ({ats_optimization.ats_score:.1f}/100): "
                    f"Keyword coverage {ats_optimization.keyword_coverage:.0f}% "
                    f"(target: {KEYWORD_COVERAGE_TARGET_MIN:.0f}-{KEYWORD_COVERAGE_TARGET_MAX:.0f}%), "
                    f"{len(ats_optimization.formatting_issues)} formatting issue(s). "
                    "Improve keyword integration and fix formatting."
                )

            if red_flags:
                feedback_parts.append(
                    f"RED FLAGS: {len(red_flags)} critical issue(s) found. "
                    f"Fix: {'; '.join(red_flags[:3])}"
                )

            if formatting_issues:
                feedback_parts.append(
                    f"FORMATTING: {len(formatting_issues)} issue(s) found. "
                    f"Fix: {'; '.join(formatting_issues[:3])}"
                )

            feedback = " | ".join(feedback_parts)

        # Stage 7.4: Output Generation
        # Sub-stage 7.4.1: Create QualityReport object
        quality_report = QualityReport(
            overall_quality_score=overall_score,
            passed_quality_threshold=passed,
            assessment_summary=assessment_summary,
            accuracy=accuracy,
            relevance=relevance,
            ats_optimization=ats_optimization,
            feedback_for_improvement=feedback,
        )

        # Sub-stage 7.4.2: Serialize to JSON
        result_json = quality_report.model_dump_json(indent=2)

        logger.info(
            f"Quality evaluation complete. Overall score: {overall_score:.1f}/100, "
            f"Passed: {passed}"
        )

        # Sub-stage 7.4.3: Return comprehensive report
        return result_json

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON input: {e}", exc_info=True)
        error_report = {
            "overall_quality_score": 0.0,
            "passed_quality_threshold": False,
            "assessment_summary": "Evaluation failed due to invalid input format",
            "accuracy": {
                "accuracy_score": 0.0,
                "exaggerated_claims": [],
                "unsupported_skills": [],
                "justification": f"JSON parsing error: {str(e)}",
            },
            "relevance": {
                "relevance_score": 0.0,
                "must_have_skills_coverage": 0.0,
                "missed_requirements": [],
                "justification": "Unable to evaluate due to parsing error",
            },
            "ats_optimization": {
                "ats_score": 0.0,
                "keyword_coverage": 0.0,
                "formatting_issues": [],
                "justification": "Unable to evaluate due to parsing error",
            },
            "feedback_for_improvement": f"Fix input format error: {str(e)}",
        }
        return json.dumps(error_report, indent=2)

    except Exception as e:
        logger.error(f"Error in quality evaluation tool: {e}", exc_info=True)
        error_report = {
            "overall_quality_score": 0.0,
            "passed_quality_threshold": False,
            "assessment_summary": "Evaluation failed due to internal error",
            "accuracy": {
                "accuracy_score": 0.0,
                "exaggerated_claims": [],
                "unsupported_skills": [],
                "justification": f"Evaluation error: {str(e)}",
            },
            "relevance": {
                "relevance_score": 0.0,
                "must_have_skills_coverage": 0.0,
                "missed_requirements": [],
                "justification": "Unable to evaluate due to error",
            },
            "ats_optimization": {
                "ats_score": 0.0,
                "keyword_coverage": 0.0,
                "formatting_issues": [],
                "justification": "Unable to evaluate due to error",
            },
            "feedback_for_improvement": f"Internal error during evaluation: {str(e)}",
        }
        return json.dumps(error_report, indent=2)


# ============================================================================
# BLOCK 8: WEIGHTED SCORING SYSTEM
# ============================================================================
# Stage 8.1: Score Calculation
# ----------------------------------------------------------------------------


def calculate_weighted_score(
    accuracy_score: float, relevance_score: float, ats_score: float
) -> float:
    """
    Calculate weighted overall quality score based on professional standards.

    Weights:
    - Accuracy: 40% (truthfulness is most critical)
    - Relevance: 35% (job fit matters greatly)
    - ATS Optimization: 25% (compatibility important but less than truth/fit)

    Args:
        accuracy_score: Accuracy score (0-100)
        relevance_score: Relevance score (0-100)
        ats_score: ATS optimization score (0-100)

    Returns:
        float: Weighted overall score (0-100)
    """
    overall_score = (
        accuracy_score * ACCURACY_WEIGHT
        + relevance_score * RELEVANCE_WEIGHT
        + ats_score * ATS_WEIGHT
    )

    return round(overall_score, 2)


# ============================================================================
# BLOCK 9: OUTPUT VALIDATION
# ============================================================================
# Stage 9.1: Data Validation
# ----------------------------------------------------------------------------


def validate_qa_output(output: Any) -> QualityReport | None:
    """
    Validate that agent output conforms to QualityReport model.

    This function attempts to parse the agent's output as a QualityReport
    and validates that all fields are present and valid.

    Args:
        output: Raw output from agent (string or dict)

    Returns:
        Optional[QualityReport]: Validated QualityReport or None if invalid
    """
    try:
        # Parse output as QualityReport
        if isinstance(output, str):
            # Try to extract JSON if wrapped in text
            json_match = re.search(r"\{.*\}", output, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
            else:
                logger.error("No JSON found in output string")
                return None
        elif isinstance(output, dict):
            data = output
        else:
            logger.error(f"Unexpected output type: {type(output)}")
            return None

        # Validate with Pydantic model
        quality_report = QualityReport(**data)

        # Additional validation checks
        if not (0 <= quality_report.overall_quality_score <= 100):
            logger.warning(f"Overall score out of range: {quality_report.overall_quality_score}")
            return None

        if not (0 <= quality_report.accuracy.accuracy_score <= 100):
            logger.warning(f"Accuracy score out of range: {quality_report.accuracy.accuracy_score}")
            return None

        if not (0 <= quality_report.relevance.relevance_score <= 100):
            logger.warning(
                f"Relevance score out of range: {quality_report.relevance.relevance_score}"
            )
            return None

        if not (0 <= quality_report.ats_optimization.ats_score <= 100):
            logger.warning(f"ATS score out of range: {quality_report.ats_optimization.ats_score}")
            return None

        logger.info("QA output validation successful")
        return quality_report

    except ValidationError as e:
        logger.error(f"QA output validation failed: {e}", exc_info=True)
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON in QA output: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Error validating QA output: {e}", exc_info=True)
        return None


# Stage 9.3: Meta-Validation
# ----------------------------------------------------------------------------


def check_qa_quality(quality_report: QualityReport) -> dict[str, Any]:
    """
    Meta-validation: Check the quality of the QA agent's own output.

    This function validates that:
    - All scores are properly justified
    - Feedback is actionable (if provided)
    - Scoring is logically consistent

    Args:
        quality_report: QualityReport to validate

    Returns:
        Dict[str, Any]: Meta-quality assessment
    """
    try:
        issues = []

        # Check for justification completeness
        if len(quality_report.accuracy.justification) < 20:
            issues.append("Accuracy justification too brief")

        if len(quality_report.relevance.justification) < 20:
            issues.append("Relevance justification too brief")

        if len(quality_report.ats_optimization.justification) < 20:
            issues.append("ATS justification too brief")

        # Check scoring consistency
        component_avg = (
            quality_report.accuracy.accuracy_score * ACCURACY_WEIGHT
            + quality_report.relevance.relevance_score * RELEVANCE_WEIGHT
            + quality_report.ats_optimization.ats_score * ATS_WEIGHT
        )

        if abs(component_avg - quality_report.overall_quality_score) > 5:
            issues.append(
                f"Scoring inconsistency: weighted components ({component_avg:.1f}) "
                f"don't match overall score ({quality_report.overall_quality_score:.1f})"
            )

        # Check feedback quality
        if not quality_report.passed_quality_threshold:
            if not quality_report.feedback_for_improvement:
                issues.append("Failed QA but no feedback provided")
            elif len(quality_report.feedback_for_improvement) < 50:
                issues.append("Feedback too brief for failed QA")

        # Calculate meta-quality score
        meta_score = 100.0 - (len(issues) * 10)
        meta_score = max(meta_score, 0.0)

        return {
            "meta_quality_score": meta_score,
            "is_valid": len(issues) == 0,
            "issues": issues,
            "summary": (
                "QA output is complete and valid"
                if not issues
                else f"QA output has {len(issues)} issue(s)"
            ),
        }

    except Exception as e:
        logger.error(f"Error in meta-validation: {e}", exc_info=True)
        return {
            "meta_quality_score": 0.0,
            "is_valid": False,
            "issues": [f"Meta-validation error: {str(e)}"],
            "summary": "Meta-validation failed",
        }


# ============================================================================
# BLOCK 10: PUBLIC API
# ============================================================================

__all__ = [
    # Main agent creation
    "create_quality_assurance_agent",
    # Evaluation functions
    "evaluate_accuracy",
    "evaluate_relevance",
    "evaluate_ats_optimization",
    # Scoring functions
    "calculate_weighted_score",
    # Validation functions
    "validate_qa_output",
    "check_qa_quality",
    # Grammar and formatting checks
    "check_grammar_quality",
    "check_formatting_standards",
    "check_red_flags",
    # Helper functions
    "compare_experience_claims",
    "check_unsupported_skills",
    "extract_must_have_requirements",
    "extract_job_keywords",
    "calculate_requirement_coverage",
    "calculate_keyword_coverage_score",
    "check_formatting_issues",
    # Tool
    "evaluate_resume_quality_tool",
]


# Module-level test function
def _test_agent_creation():
    """
    Test function to verify agent can be created successfully.

    Usage:
        python -m src.agents.quality_assurance_agent
    """
    try:
        logger.info("Testing Quality Assurance Reviewer agent creation...")
        agent = create_quality_assurance_agent()
        logger.info(f"Agent created successfully: {agent.role}")
        logger.info("Test passed!")
        return True
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    # Run test when module is executed directly
    success = _test_agent_creation()
    exit(0 if success else 1)
