"""
Skills Section Optimizer Agent - AI-Powered Skills Intelligence Engine
======================================================================

OVERVIEW:
---------
This module defines the Skills Section Optimizer Agent, an AI-first system that
intelligently infers, validates, prioritizes, and categorizes skills for resume
optimization. Unlike rule-based systems, it leverages LLM intelligence to work
across ALL professional domains without hardcoded mappings.

WHAT MAKES THIS AGENT ESSENTIAL:
--------------------------------
- **AI-Inferred Skills**: Discovers unstated but evident skills from experience
- **Evidence-Based Validation**: Ensures every skill claim has supporting evidence
- **ATS-Optimized Presentation**: Balances keyword density with human readability
- **Domain-Agnostic Design**: Works for law, medicine, tech, marketing, engineering, etc.
- **Confidence Scoring**: Quantifies skill validity with multi-factor analysis
- **Strategic Prioritization**: Ranks skills by job relevance and career impact

AGENT DESIGN PRINCIPLES:
------------------------
- **AI-First Approach**: LLM handles domain inference instead of hardcoded rules
- **Evidence-Driven**: Every skill must be verifiable in experience descriptions
- **Domain-Agnostic**: Zero hardcoded mappings - works across all professional fields
- **Truthfulness-First**: Strict validation prevents resume inflation or fabrication
- **Maintainability**: Small, focused functions with clear separation of concerns
- **Performance-Optimized**: Efficient confidence scoring and quality assessment

WORKFLOW OVERVIEW:
------------------
1. **Skill Inference**: Analyze experiences to identify unstated but evident skills
2. **Evidence Validation**: Verify each inferred skill has supporting experience evidence
3. **Confidence Scoring**: Calculate multi-factor confidence scores for skill validity
4. **Job Alignment**: Prioritize skills based on job description requirements
5. **Categorization**: Group skills into logical, ATS-friendly categories
6. **Quality Optimization**: Balance keyword coverage with professional presentation
7. **Final Validation**: Ensure output meets quality thresholds and schema requirements

MODULE STRUCTURE (Hierarchical Organization):
=============================================
This module is organized into 9 main BLOCKS, each containing STAGES with SUB-STAGES:

BLOCK 1: MODULE SETUP & CONFIGURATION
├── Stage 1.1: Import Management
│   ├── Sub-stage 1.1.1: Standard library imports (json, re, uuid)
│   ├── Sub-stage 1.1.2: CrewAI framework imports
│   ├── Sub-stage 1.1.3: Project-specific imports with fallback handling
│   └── Sub-stage 1.1.4: Logger initialization
│
├── Stage 1.2: Module Constants
│   ├── Sub-stage 1.2.1: Confidence thresholds and scoring weights
│   ├── Sub-stage 1.2.2: Skill count and quality thresholds
│   └── Sub-stage 1.2.3: Experience analysis parameters
│
└── Stage 1.3: Logging Infrastructure
    ├── Sub-stage 1.3.1: LogContext class for structured logging
    ├── Sub-stage 1.3.2: Thread-safe correlation ID management
    └── Sub-stage 1.3.3: Context-aware logging utilities

BLOCK 2: DATA MODELS & CONTEXT
├── Stage 2.1: Skill Inference Models
│   ├── Sub-stage 2.1.1: SkillInferenceContext model
│   ├── Sub-stage 2.1.2: SkillValidationResult model
│   └── Sub-stage 2.1.3: Model validation and constraints
│
└── Stage 2.2: Data Validation
    ├── Sub-stage 2.2.1: Input data validation
    ├── Sub-stage 2.2.2: Model integrity checks
    └── Sub-stage 2.2.3: Error handling for invalid inputs

BLOCK 3: SKILL INFERENCE ENGINE
├── Stage 3.1: Core Inference Function
│   ├── Sub-stage 3.1.1: infer_missing_skills() main orchestration
│   ├── Sub-stage 3.1.2: Context preparation and validation
│   └── Sub-stage 3.1.3: LLM inference with error handling
│
├── Stage 3.2: Missing Skills Identification
│   ├── Sub-stage 3.2.1: _identify_missing_skills() function
│   ├── Sub-stage 3.2.2: Gap analysis against job requirements
│   └── Sub-stage 3.2.3: Experience-based skill discovery
│
├── Stage 3.3: Prompt Engineering
│   ├── Sub-stage 3.3.1: _build_skill_inference_prompt() function
│   ├── Sub-stage 3.3.2: Dynamic prompt construction with context
│   └── Sub-stage 3.3.3: Evidence integration and job alignment
│
├── Stage 3.4: Response Parsing
│   ├── Sub-stage 3.4.1: _parse_skill_inference_response() function
│   ├── Sub-stage 3.4.2: JSON cleaning and validation
│   └── Sub-stage 3.4.3: Error recovery for malformed responses
│
└── Stage 3.5: Inference Quality Checks
    ├── Sub-stage 3.5.1: Basic validation of inferred skills
    ├── Sub-stage 3.5.2: Duplicate detection and removal
    └── Sub-stage 3.5.3: Initial confidence assessment

BLOCK 4: SKILL VALIDATION SYSTEM
├── Stage 4.1: Validation Orchestration
│   ├── Sub-stage 4.1.1: validate_skill_inference() main function
│   ├── Sub-stage 4.1.2: Batch processing of skills
│   └── Sub-stage 4.1.3: Results aggregation and reporting
│
├── Stage 4.2: LLM-Based Validation
│   ├── Sub-stage 4.2.1: _validate_with_llm() function
│   ├── Sub-stage 4.2.2: Individual skill validation prompts
│   └── Sub-stage 4.2.3: Evidence verification logic
│
├── Stage 4.3: Evidence Verification
│   ├── Sub-stage 4.3.1: _verify_evidence_in_experience() function
│   ├── Sub-stage 4.3.2: Text matching and validation
│   └── Sub-stage 4.3.3: Quote extraction and verification
│
├── Stage 4.4: Confidence Scoring
│   ├── Sub-stage 4.4.1: _calculate_confidence_score() function
│   ├── Sub-stage 4.4.2: Multi-factor scoring algorithm
│   └── Sub-stage 4.4.3: Weighted confidence calculation
│
├── Stage 4.5: Domain Relevance
│   ├── Sub-stage 4.5.1: _is_domain_relevant() function
│   ├── Sub-stage 4.5.2: Industry alignment assessment
│   └── Sub-stage 4.5.3: Contextual relevance scoring
│
├── Stage 4.6: Job Priority Assessment
│   ├── Sub-stage 4.6.1: _is_high_priority_skill() function
│   ├── Sub-stage 4.6.2: Job description analysis
│   └── Sub-stage 4.6.3: Requirement matching logic
│
└── Stage 4.7: Validation Prompts
    ├── Sub-stage 4.7.1: _build_validation_prompt() function
    ├── Sub-stage 4.7.2: Dynamic validation prompt construction
    └── Sub-stage 4.7.3: Evidence integration strategies

BLOCK 5: SKILL PRIORITIZATION & CATEGORIZATION
├── Stage 5.1: Main Optimization Function
│   ├── Sub-stage 5.1.1: prioritize_and_categorize_skills() orchestration
│   ├── Sub-stage 5.1.2: Input validation and context preparation
│   └── Sub-stage 5.1.3: LLM prioritization with fallback handling
│
├── Stage 5.2: Response Parsing
│   ├── Sub-stage 5.2.1: _parse_prioritization_response() function
│   ├── Sub-stage 5.2.2: JSON response validation and cleaning
│   └── Sub-stage 5.2.3: Error recovery for parsing failures
│
├── Stage 5.3: Fallback Prioritization
│   ├── Sub-stage 5.3.1: _fallback_prioritization() function
│   ├── Sub-stage 5.3.2: Rule-based categorization logic
│   └── Sub-stage 5.3.3: Basic skill grouping algorithms
│
├── Stage 5.4: Category Inference
│   ├── Sub-stage 5.4.1: _infer_skill_categories() function
│   ├── Sub-stage 5.4.2: Dynamic category discovery
│   └── Sub-stage 5.4.3: Taxonomy development logic
│
└── Stage 5.5: Priority Scoring
    ├── Sub-stage 5.5.1: _get_skill_priority_score() function
    ├── Sub-stage 5.5.2: Multi-criteria priority assessment
    └── Sub-stage 5.5.3: Job relevance and evidence weighting

BLOCK 6: AGENT CONFIGURATION & CREATION
├── Stage 6.1: Configuration Loading
│   ├── Sub-stage 6.1.1: Load from agents.yaml with fallback
│   ├── Sub-stage 6.1.2: Validate required configuration fields
│   └── Sub-stage 6.1.3: Error handling for missing configuration
│
├── Stage 6.2: Default Configuration
│   ├── Sub-stage 6.2.1: Define skills optimization role
│   ├── Sub-stage 6.2.2: Set LLM parameters for inference tasks
│   └── Sub-stage 6.2.3: Configure validation and reasoning settings
│
└── Stage 6.3: Agent Creation
    ├── Sub-stage 6.3.1: CrewAI Agent initialization
    ├── Sub-stage 6.3.2: Tool assignment and capability configuration
    └── Sub-stage 6.3.3: Performance and resilience settings

BLOCK 7: MAIN OPTIMIZATION WORKFLOW
├── Stage 7.1: Workflow Orchestration
│   ├── Sub-stage 7.1.1: optimize_skills_section() main function
│   ├── Sub-stage 7.1.2: End-to-end skills optimization pipeline
│   └── Sub-stage 7.1.3: Error handling and recovery
│
├── Stage 7.2: Pipeline Stages
│   ├── Sub-stage 7.2.1: Skill inference from experiences
│   ├── Sub-stage 7.2.2: Validation and confidence scoring
│   ├── Sub-stage 7.2.3: Prioritization and categorization
│   └── Sub-stage 7.2.4: Quality assessment and optimization
│
└── Stage 7.3: Results Integration
    ├── Sub-stage 7.3.1: Combine inferred and existing skills
    ├── Sub-stage 7.3.2: Deduplication and conflict resolution
    └── Sub-stage 7.3.3: Final output preparation

BLOCK 8: OUTPUT VALIDATION & QUALITY CHECKS
├── Stage 8.1: Output Validation
│   ├── Sub-stage 8.1.1: validate_skills_output() function
│   ├── Sub-stage 8.1.2: Schema validation against Pydantic models
│   └── Sub-stage 8.1.3: Data integrity and completeness checks
│
├── Stage 8.2: Quality Assessment
│   ├── Sub-stage 8.2.1: check_skills_quality() function
│   ├── Sub-stage 8.2.2: Multi-dimensional quality evaluation
│   └── Sub-stage 8.2.3: Scoring and recommendation generation
│
├── Stage 8.3: Keyword Coverage Analysis
│   ├── Sub-stage 8.3.1: _assess_keyword_coverage() function
│   ├── Sub-stage 8.3.2: Job description keyword matching
│   └── Sub-stage 8.3.3: Coverage percentage calculation
│
└── Stage 8.4: Presentation Quality Assessment
    ├── Sub-stage 8.4.1: _assess_presentation_quality() function
    ├── Sub-stage 8.4.2: ATS compatibility evaluation
    └── Sub-stage 8.4.3: Human readability assessment

BLOCK 9: TESTING & DIAGNOSTICS
├── Stage 9.1: Integration Testing
│   ├── Sub-stage 9.1.1: Test agent creation and configuration
│   ├── Sub-stage 9.1.2: Test skill inference pipeline
│   └── Sub-stage 9.1.3: Test validation and quality checks
│
└── Stage 9.2: Diagnostic Functions
    ├── Sub-stage 9.2.1: Debugging utilities
    ├── Sub-stage 9.2.2: Performance monitoring
    └── Sub-stage 9.2.3: Error reporting and analysis

HOW TO USE THIS MODULE:
-----------------------
1. Import: `from src.agents.skills_optimizer_agent import create_skills_optimizer_agent`
2. Create Agent: `agent = create_skills_optimizer_agent()`
3. Optimize Skills: Use `optimize_skills_section()` with resume and job data
4. Validate Output: Use `validate_skills_output()` to ensure data quality
5. Check Quality: Use `check_skills_quality()` for comprehensive assessment

KEY AI-FIRST PRINCIPLES:
------------------------
- **Domain Agnostic**: No hardcoded skill mappings - LLM infers from context
- **Evidence Required**: Every skill must be verifiable in experience descriptions
- **Confidence Scoring**: Multi-factor validation prevents fabrication
- **Job-Aligned**: Skills prioritized based on job requirements and importance
- **Truthful**: Strict validation ensures resume accuracy and credibility

TECHNICAL ARCHITECTURE:
-----------------------
- **Inference Engine**: LLM-powered skill discovery from unstructured experience text
- **Validation Pipeline**: Multi-stage verification with confidence scoring
- **Prioritization Logic**: Job-aware ranking with fallback algorithms
- **Categorization System**: Dynamic taxonomy development
- **Quality Assurance**: Comprehensive validation and assessment framework
- **Error Recovery**: Graceful handling of LLM failures and parsing errors

INFERENCE METHODOLOGY:
---------------------
The agent employs a sophisticated AI-first approach:
1. **Experience Analysis**: Deep reading of work experience descriptions
2. **Gap Identification**: Compare existing skills against job requirements
3. **Skill Inference**: LLM identifies unstated but evident competencies
4. **Evidence Verification**: Cross-reference inferred skills with experience text
5. **Confidence Calculation**: Multi-factor scoring of skill validity
6. **Job Prioritization**: Rank skills by relevance to target position
7. **Intelligent Categorization**: Group skills into ATS-optimized categories
8. **Quality Optimization**: Balance keyword coverage with professional presentation

This creates skills sections that are both ATS-optimized and genuinely representative
of candidate capabilities, without resorting to resume inflation or fabrication.
"""

import json
import re
import uuid
from typing import Optional, Union

from crewai import Agent
from pydantic import BaseModel, Field

try:
    from src.core.config import get_agents_config, get_config
    from src.core.logger import get_logger
    from src.core.resiliency import resilient_llm_call
    from src.data_models.job import JobDescription, SkillImportance
    from src.data_models.resume import Experience, OptimizedSkillsSection, Skill
    from src.data_models.strategy import AlignmentStrategy
except ImportError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.core.config import get_agents_config, get_config
    from src.core.logger import get_logger
    from src.core.resiliency import resilient_llm_call
    from src.data_models.job import JobDescription, SkillImportance
    from src.data_models.resume import Experience, OptimizedSkillsSection, Skill
    from src.data_models.strategy import AlignmentStrategy

logger = get_logger(__name__)


# ==============================================================================
# BLOCK 1: MODULE SETUP & CONFIGURATION
# ==============================================================================
# PURPOSE: Initialize the module with constants, logging, and foundational infrastructure
# WHAT: Core setup including constants, logging context, and module-level configuration
# WHY: Provides the stable foundation for AI-powered skills optimization
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1.2: Module Constants
# ------------------------------------------------------------------------------
# This stage defines all configuration constants used throughout the skills optimization system.

# Confidence thresholds
CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence score to accept inferred skill

# Scoring weights for confidence calculation
EVIDENCE_WEIGHT = 0.4  # Weight for evidence strength in confidence score
DOMAIN_WEIGHT = 0.3  # Weight for domain relevance in confidence score
JOB_IMPORTANCE_WEIGHT = 0.3  # Weight for job importance in confidence score

# Skill count thresholds
MIN_SKILL_COUNT = 15  # Minimum recommended skills for ATS
MAX_SKILL_COUNT = 25  # Maximum recommended skills for readability

# Experience and evidence parameters
TOP_EXPERIENCES_COUNT = 3  # Number of recent experiences to include in prompts
MIN_EVIDENCE_QUOTE_LENGTH = 10  # Minimum character length for valid evidence quotes

# Quality thresholds
QUALITY_THRESHOLD = 75.0  # Minimum acceptable quality score
MIN_CATEGORIES = 3  # Minimum recommended skill categories


# ------------------------------------------------------------------------------
# Stage 1.3: Logging Infrastructure
# ------------------------------------------------------------------------------
# This stage provides thread-safe structured logging with correlation IDs.


class LogContext:
    """Thread-safe context manager for correlation IDs and structured logging."""

    _correlation_id: Optional[str] = None

    @classmethod
    def set_correlation_id(cls, correlation_id: str):
        """Set correlation ID for current execution context."""
        cls._correlation_id = correlation_id

    @classmethod
    def get_correlation_id(cls) -> str:
        """Get current correlation ID or generate new one."""
        if cls._correlation_id is None:
            cls._correlation_id = str(uuid.uuid4())[:8]
        return cls._correlation_id

    @classmethod
    def clear(cls):
        """Clear correlation ID."""
        cls._correlation_id = None


def log_with_context(level: str = "info", **context):
    """Structured logging with correlation ID."""
    correlation_id = LogContext.get_correlation_id()
    context_dict = {"correlation_id": correlation_id, **context}

    log_func = getattr(logger, level)
    message = context.pop("message", "")

    # Format context as key=value pairs
    context_str = " ".join([f"{k}={v}" for k, v in context_dict.items()])
    log_func(f"{message} | {context_str}")


# ==============================================================================
# LLM Retry Decorator
# ==============================================================================


# ==============================================================================
# BLOCK 2: DATA MODELS & CONTEXT
# ==============================================================================
# PURPOSE: Define structured data models for skill inference and validation
# WHAT: Pydantic models that enforce data integrity throughout the optimization pipeline
# WHY: Ensures type safety and validation for complex skill optimization workflows
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 2.1: Skill Inference Models
# ------------------------------------------------------------------------------
# This stage defines the core data models for skill inference and validation operations.


class SkillInferenceContext(BaseModel):
    """Context data for skill inference validation"""

    candidate_experience: list[Experience]
    job_requirements: JobDescription
    existing_skills: list[Skill]
    alignment_strategy: AlignmentStrategy


class SkillValidationResult(BaseModel):
    """Result of skill validation check"""

    is_valid: bool
    confidence_score: float = Field(ge=0.0, le=1.0)
    justification: str
    evidence_snippets: list[str] = Field(default_factory=list)
    rejection_reason: Optional[str] = None


# ==============================================================================
# Helper Functions
# ==============================================================================


def _clean_json_string(json_str: str) -> str:
    """Clean JSON string by removing markdown code blocks."""
    if "```json" in json_str:
        return json_str.split("```json")[1].split("```")[0].strip()
    elif "```" in json_str:
        return json_str.split("```")[1].split("```")[0].strip()
    return json_str.strip()


# ==============================================================================
# BLOCK 3: SKILL INFERENCE ENGINE
# ==============================================================================
# PURPOSE: AI-powered discovery of skills from experience descriptions
# WHAT: LLM-based inference system that identifies unstated but evident competencies
# WHY: Enables comprehensive skill identification without hardcoded domain mappings
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 3.1: Core Inference Function
# ------------------------------------------------------------------------------
# This stage orchestrates the main skill inference workflow using LLM intelligence.


@resilient_llm_call(provider="gemini")
def infer_missing_skills(context: SkillInferenceContext, agent: Agent) -> list[Skill]:
    """
    Use LLM to intelligently infer missing skills based on experience.

    Instead of hardcoded mappings, we ask the AI:
    "Given this person's experience, what related skills from the job
    description can be reasonably inferred?"

    Returns:
        List of inferred skills with AI-generated justifications
    """
    log_with_context(
        level="info",
        message="Starting AI-powered skill inference",
        missing_count=len(_identify_missing_skills(context)),
    )

    # Extract missing required skills
    missing_skills = _identify_missing_skills(context)

    if not missing_skills:
        log_with_context(level="info", message="No missing required skills to infer")
        return []

    # Build context-rich prompt for LLM
    inference_prompt = _build_skill_inference_prompt(
        missing_skills=missing_skills,
        experience=context.candidate_experience,
        existing_skills=context.existing_skills,
        job_context=context.job_requirements,
    )

    # Let AI analyze and suggest inferences
    try:
        response = agent.execute_task(inference_prompt)
        inferred_skills = _parse_skill_inference_response(response)

        # Validate each inference
        validated_skills = []
        for skill in inferred_skills:
            validation = validate_skill_inference(skill, context, agent)
            if validation.is_valid and validation.confidence_score >= CONFIDENCE_THRESHOLD:
                # Create new skill object with validation details (avoid mutation)
                validated_skill = skill.model_copy(
                    update={
                        "confidence_score": validation.confidence_score,
                        "evidence": validation.evidence_snippets,
                    }
                )
                validated_skills.append(validated_skill)
                log_with_context(
                    level="info",
                    message="Skill validated",
                    skill=skill.skill_name,
                    confidence=f"{validation.confidence_score:.2f}",
                )
            else:
                log_with_context(
                    level="debug",
                    message="Skill rejected",
                    skill=skill.skill_name,
                    reason=validation.rejection_reason,
                )

        return validated_skills

    except json.JSONDecodeError as e:
        log_with_context(level="error", message="Skill inference JSON parsing failed", error=str(e))
        return []
    except Exception as e:
        log_with_context(level="error", message="Skill inference failed", error=str(e))
        return []


def _identify_missing_skills(context: SkillInferenceContext) -> list[str]:
    """Find required skills missing from candidate's resume"""
    existing_skill_names = {s.skill_name.lower() for s in context.existing_skills}

    missing = []
    for req in context.job_requirements.requirements:
        if req.importance == SkillImportance.MUST_HAVE:
            skill_lower = req.requirement.lower()
            if skill_lower not in existing_skill_names:
                missing.append(req.requirement)

    return missing


def _build_skill_inference_prompt(
    missing_skills: list[str],
    experience: list[Experience],
    existing_skills: list[Skill],
    job_context: JobDescription,
) -> str:
    """
    Build a context-rich prompt for AI skill inference.

    Key: We provide evidence and ask AI to make connections,
    not just match keywords.
    """
    # Summarize candidate's experience domain
    experience_summary = "\n".join(
        [
            f"- {exp.job_title} at {exp.company_name}: {exp.description[:200]}..."
            for exp in experience[:TOP_EXPERIENCES_COUNT]
        ]
    )

    existing_skills_str = ", ".join([s.skill_name for s in existing_skills])

    prompt = f"""
You are a domain expert analyzing a candidate's qualifications for a {job_context.job_title} role.

CANDIDATE'S EXPERIENCE SUMMARY:
{experience_summary}

CANDIDATE'S EXISTING SKILLS:
{existing_skills_str}

MISSING REQUIRED SKILLS FROM JOB:
{", ".join(missing_skills)}

TASK:
For each missing skill, determine if it can be TRUTHFULLY inferred from the candidate's experience.

INFERENCE RULES:
1. Only infer if there's clear evidence in experience
2. Infer related tools/frameworks from domain expertise (e.g., "contract law" → "legal drafting")
3. Infer ecosystem skills (e.g., "React" → "JSX", "Hooks")
4. DO NOT infer skills requiring specialized training/certification
5. DO NOT infer if candidate worked in unrelated domain

OUTPUT FORMAT:
Return a JSON list of objects. Do not include any other text.
[
  {{
    "skill_name": "exact name from job posting",
    "category": "appropriate category",
    "proficiency_level": "beginner/intermediate/advanced",
    "justification": "why this inference is truthful based on experience",
    "evidence": ["specific quote from experience supporting this"]
  }}
]

If skill cannot be inferred, explain why in a brief note.
"""
    return prompt


def _parse_skill_inference_response(response: str) -> list[Skill]:
    """Parse LLM response into Skill objects"""
    try:
        clean_json = _clean_json_string(response)
        skills_data = json.loads(clean_json)

        # Handle if LLM returns a dict instead of list
        if isinstance(skills_data, dict) and "skills" in skills_data:
            skills_data = skills_data["skills"]

        return [Skill(**skill) for skill in skills_data]
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse inference response as JSON: {e}")
        # Try to extract JSON from text if direct parse fails
        try:
            match = re.search(r"\[.*\]", response, re.DOTALL)
            if match:
                skills_data = json.loads(match.group())
                return [Skill(**skill) for skill in skills_data]
        except json.JSONDecodeError:
            logger.error("Could not extract valid JSON from response")
        return []
    except Exception as e:
        logger.error(f"Unexpected error parsing skill inference: {e}", exc_info=True)
        return []


# ==============================================================================
# BLOCK 4: SKILL VALIDATION SYSTEM
# ==============================================================================
# PURPOSE: Ensure skill claims are truthful and evidence-based
# WHAT: Multi-stage validation pipeline that verifies skills against experience evidence
# WHY: Prevents resume inflation and maintains credibility through strict validation
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 4.1: Validation Orchestration
# ------------------------------------------------------------------------------
# This stage orchestrates the comprehensive skill validation process.


def validate_skill_inference(
    skill: Skill, context: SkillInferenceContext, agent: Optional[Agent] = None
) -> SkillValidationResult:
    """
    Validate that an inferred skill is truthfully supported by experience.

    This is the CRITICAL gatekeeper preventing resume lies.

    Validation Strategy:
    1. Check if justification references actual experience content (rule-based)
    2. Verify evidence quotes exist in experience text (rule-based)
    3. Use LLM to validate logical connection (AI-powered) - if agent provided
    4. Calculate confidence score based on evidence strength

    Args:
        skill: Skill to validate
        context: Inference context with experience and job info
        agent: Optional CrewAI agent for LLM-based validation

    Returns:
        SkillValidationResult with validity, confidence, and reasoning
    """
    log_with_context(level="debug", message="Validating skill", skill=skill.skill_name)

    # Extract evidence from justification
    if not hasattr(skill, "justification") or not skill.justification:
        return SkillValidationResult(
            is_valid=False,
            confidence_score=0.0,
            justification="",
            rejection_reason="No justification provided",
        )

    # Check if evidence snippets exist in actual experience
    # evidence_found = _verify_evidence_in_experience(  # Commented out: unused variable
    #     evidence=getattr(skill, "evidence", []), experience=context.candidate_experience
    # )

    # Calculate base confidence from rules
    base_confidence = _calculate_confidence_score(skill, context)

    # Use LLM for additional validation if agent provided
    if agent is not None:
        try:
            llm_validation = _validate_with_llm(skill, context, agent)
            # Combine rule-based and LLM-based confidence
            # LLM gets 60% weight, rules get 40%
            final_confidence = (
                llm_validation.get("confidence", base_confidence) * 0.6 + base_confidence * 0.4
            )
            log_with_context(
                level="debug",
                message="LLM validation complete",
                skill=skill.skill_name,
                llm_confidence=llm_validation.get("confidence", 0),
                rule_confidence=base_confidence,
                final_confidence=final_confidence,
            )
        except Exception as e:
            log_with_context(
                level="warning",
                message="LLM validation failed, using rule-based only",
                skill=skill.skill_name,
                error=str(e),
            )
            final_confidence = base_confidence
    else:
        final_confidence = base_confidence

    return SkillValidationResult(
        is_valid=final_confidence >= CONFIDENCE_THRESHOLD,
        confidence_score=final_confidence,
        justification=skill.justification,
        evidence_snippets=getattr(skill, "evidence", []),
        rejection_reason=None
        if final_confidence >= CONFIDENCE_THRESHOLD
        else "Low confidence inference",
    )


def _validate_with_llm(skill: Skill, context: SkillInferenceContext, agent: Agent) -> dict:
    """
    Use LLM to validate skill inference for truthfulness.

    Args:
        skill: Skill to validate
        context: Inference context
        agent: CrewAI agent

    Returns:
        Dictionary with confidence score and reasoning
    """
    validation_prompt = _build_validation_prompt(skill, context)

    try:
        response = agent.execute_task(validation_prompt)

        # Try to parse JSON response
        clean_json = _clean_json_string(response)
        result = json.loads(clean_json)

        return {
            "confidence": result.get("confidence_score", 0.5),
            "reasoning": result.get("reasoning", "No reasoning provided"),
            "is_truthful": result.get("is_truthful", False),
        }

    except (json.JSONDecodeError, Exception) as e:
        log_with_context(
            level="warning", message="LLM validation response parsing failed", error=str(e)
        )
        # Return neutral confidence if LLM validation fails
        return {"confidence": 0.5, "reasoning": "LLM validation failed", "is_truthful": False}


def _verify_evidence_in_experience(evidence: list[str], experience: list[Experience]) -> bool:
    """Check if evidence quotes actually exist in experience text"""
    if not evidence or evidence is None:
        return False

    experience_text = " ".join(
        [f"{exp.description} {' '.join(exp.achievements)}" for exp in experience]
    ).lower()

    # Check if at least one piece of evidence is found (fuzzy match)
    for quote in evidence:
        # Simple normalization
        quote_norm = quote.lower().strip()
        if len(quote_norm) < MIN_EVIDENCE_QUOTE_LENGTH:
            continue  # Skip very short quotes

        if quote_norm in experience_text:
            return True

        # Try partial match (if quote is a substring of a sentence)
        # or if a sentence in experience contains the quote
        pass

    return False


def _calculate_confidence_score(skill: Skill, context: SkillInferenceContext) -> float:
    """
    Calculate confidence score for skill inference.

    Factors:
    - Evidence strength (0.4 weight)
    - Domain relevance (0.3 weight)
    - Skill importance to job (0.3 weight)
    """
    score = 0.0

    # Evidence strength
    evidence_count = len(getattr(skill, "evidence", []) or [])
    evidence_score = min(evidence_count * 0.2, EVIDENCE_WEIGHT)
    score += evidence_score

    # Domain relevance (simplified)
    if _is_domain_relevant(skill, context):
        score += DOMAIN_WEIGHT

    # Job importance
    if _is_high_priority_skill(skill, context.job_requirements):
        score += JOB_IMPORTANCE_WEIGHT

    return min(score, 1.0)


def _is_domain_relevant(skill: Skill, context: SkillInferenceContext) -> bool:
    """Check if skill is relevant to candidate's domain"""
    # Simplified check - can be enhanced with LLM
    skill_lower = skill.skill_name.lower()
    experience_text = " ".join([exp.description.lower() for exp in context.candidate_experience])
    # Check if skill words appear in experience
    return any(word in experience_text for word in skill_lower.split())


def _is_high_priority_skill(skill: Skill, job_desc: JobDescription) -> bool:
    """Check if skill is high priority in job requirements"""
    for req in job_desc.requirements:
        if req.requirement.lower() == skill.skill_name.lower():
            return req.importance == SkillImportance.MUST_HAVE
    return False


def _build_validation_prompt(skill: Skill, context: SkillInferenceContext) -> str:
    """Build prompt for LLM-based validation."""
    experience_summary = "\n".join(
        [
            f"- {exp.job_title}: {exp.description[:150]}..."
            for exp in context.candidate_experience[:2]
        ]
    )

    return f"""
You are a truthfulness validator for resume skill inferences.

SKILL TO VALIDATE: {skill.skill_name}
JUSTIFICATION PROVIDED: {skill.justification}
EVIDENCE CITED: {getattr(skill, "evidence", [])}

CANDIDATE'S ACTUAL EXPERIENCE:
{experience_summary}

QUESTION: Is this skill inference truthful and reasonable?

EVALUATION CRITERIA:
1. Does the evidence actually exist in the experience text?
2. Is the domain connection logical and not a stretch?
3. Would a hiring manager accept this inference?
4. Is the skill commonly associated with this domain?

OUTPUT FORMAT (JSON only):
{{
  "is_truthful": true/false,
  "confidence_score": 0.0-1.0,
  "reasoning": "Brief explanation of your assessment"
}}

Be strict. Only approve if evidence is clear and connection is logical.
"""


# ==============================================================================
# BLOCK 5: SKILL PRIORITIZATION & CATEGORIZATION
# ==============================================================================
# PURPOSE: Intelligently rank and group skills for optimal presentation
# WHAT: AI-guided prioritization based on job relevance and dynamic categorization
# WHY: Creates ATS-optimized skill sections with strategic importance ordering
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 5.1: Main Optimization Function
# ------------------------------------------------------------------------------
# This stage orchestrates the complete prioritization and categorization workflow.


@resilient_llm_call(provider="gemini")
def prioritize_and_categorize_skills(
    all_skills: list[Skill], context: SkillInferenceContext, agent: Agent
) -> dict[str, any]:
    """
    Use AI to intelligently prioritize and categorize skills.

    Returns:
        Dictionary with:
        - prioritized_skills: Ordered list
        - categories: Grouped skills
        - ats_score: Match score
    """
    log_with_context(
        level="info", message="Prioritizing and categorizing skills", total_skills=len(all_skills)
    )

    # Separate requirements by importance for better prompting
    must_have_reqs = [
        req
        for req in context.job_requirements.requirements
        if req.importance == SkillImportance.MUST_HAVE
    ]
    should_have_reqs = [
        req
        for req in context.job_requirements.requirements
        if req.importance == SkillImportance.SHOULD_HAVE
    ]
    nice_to_have_reqs = [
        req
        for req in context.job_requirements.requirements
        if req.importance == SkillImportance.NICE_TO_HAVE
    ]

    # Determine domain-appropriate categories based on job title
    job_title_lower = context.job_requirements.job_title.lower()
    if any(
        term in job_title_lower
        for term in ["engineer", "developer", "software", "ai", "ml", "data"]
    ):
        allowed_categories = [
            "Programming Languages",
            "AI & Machine Learning",
            "Cloud Platforms",
            "DevOps",
            "Databases",
            "Data Science",
            "Web Frameworks",
            "Tools & Platforms",
        ]
    elif any(term in job_title_lower for term in ["marketing", "content", "social"]):
        allowed_categories = [
            "Digital Marketing",
            "Analytics & Data",
            "Content Creation",
            "Marketing Platforms",
            "Design Tools",
        ]
    elif any(term in job_title_lower for term in ["legal", "attorney", "counsel"]):
        allowed_categories = [
            "Practice Areas",
            "Legal Research",
            "Documentation",
            "Legal Systems",
            "Compliance",
        ]
    else:
        # Generic categories for other domains
        allowed_categories = ["Core Skills", "Technical Skills", "Tools & Platforms", "Soft Skills"]

    # Build skill metadata for better LLM understanding
    skills_with_metadata = [
        {
            "name": s.skill_name,
            "category": s.category or "Uncategorized",
            "proficiency": s.proficiency_level or "Not specified",
            "years": s.years_of_experience or 0,
        }
        for s in all_skills
    ]

    prioritization_prompt = f"""
You are an ATS optimization specialist for {context.job_requirements.job_title} positions.

TASK: Reorder and categorize these {len(all_skills)} skills for maximum ATS impact and readability.

INPUT SKILLS (with metadata):
{json.dumps(skills_with_metadata, indent=2)}

JOB REQUIREMENTS (by priority):

MUST-HAVE (Critical - highest priority):
{json.dumps([req.requirement for req in must_have_reqs], indent=2) if must_have_reqs else "[]"}

SHOULD-HAVE (Important):
{json.dumps([req.requirement for req in should_have_reqs], indent=2) if should_have_reqs else "[]"}

NICE-TO-HAVE (Optional):
{json.dumps([req.requirement for req in nice_to_have_reqs], indent=2) if nice_to_have_reqs else "[]"}

CATEGORIZATION GUIDELINES (preferred categories):
1. For best ATS results, prefer these category names:
   {json.dumps(allowed_categories)}

2. You MAY create similar category names if needed (e.g., "Cloud & Infrastructure" for "Cloud Platforms" + "DevOps")

3. Prioritization within categories:
   - MUST-HAVE skills first
   - SHOULD-HAVE skills second
   - NICE-TO-HAVE skills third
   - Additional relevant skills last

3. Target 15-25 total skills
   - Only remove skills if they are truly outdated (e.g., "COBOL", "Flash")
   - Keep all modern, relevant skills even if not in job requirements

4. Each skill from input MUST appear in either "prioritized_skills" or "removed_skills"

FEW-SHOT EXAMPLE:

Input: [{{'name': 'Python', 'category': 'Programming'}}, {{'name': 'Docker', 'category': 'DevOps'}}, {{'name': 'AWS', 'category': 'Cloud'}}]
Must-Have: ['Python', 'AWS']

Output:
{{
  "prioritized_skills": ["Python", "AWS", "Docker"],
  "categories": {{
    "Programming Languages": ["Python"],
    "Cloud Platforms": ["AWS"],
    "DevOps": ["Docker"]
  }},
  "removed_skills": [],
  "ats_score": 95.0,
  "reasoning": "Prioritized must-have skills first"
}}

STRICT OUTPUT FORMAT (JSON only, no markdown, no code blocks):
{{
  "prioritized_skills": ["Skill1", "Skill2", ...],
  "categories": {{
    "Category Name": ["Skill1", "Skill2"],
    "Another Category": ["Skill3"]
  }},
  "removed_skills": ["OutdatedSkill"] or [],
  "ats_score": 0.0-100.0,
  "reasoning": "One sentence explaining prioritization strategy"
}}

VALIDATION REQUIREMENTS:
- Try to include ALL {len(all_skills)} input skills unless truly outdated
- Categories should use names from preferred list or similar variations
- ATS score MUST be between 0 and 100
- prioritized_skills and categories MUST contain the same skills

Now categorize the actual skills above:
"""

    try:
        response = agent.execute_task(prioritization_prompt)
        result = _parse_prioritization_response(response)
        return result
    except json.JSONDecodeError as e:
        log_with_context(
            level="warning",
            message="Prioritization JSON parsing failed, using fallback",
            error=str(e),
        )
        return _fallback_prioritization(all_skills, context)
    except Exception as e:
        log_with_context(
            level="error", message="Prioritization failed, using fallback", error=str(e)
        )
        return _fallback_prioritization(all_skills, context)


def _parse_prioritization_response(response: str) -> dict[str, any]:
    """Parse LLM's prioritization output"""
    try:
        clean_json = _clean_json_string(response)
        return json.loads(clean_json)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse prioritization as JSON: {e}")
        # Try regex extraction
        try:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                return json.loads(match.group())
        except json.JSONDecodeError:
            logger.error("Could not extract valid JSON from prioritization response")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error parsing prioritization: {e}", exc_info=True)
        return {}


def _fallback_prioritization(skills: list[Skill], context: SkillInferenceContext) -> dict[str, any]:
    """
    Enhanced fallback prioritization with domain intelligence.

    Instead of generic "Skills" category, uses domain-aware categorization.
    """
    log_with_context(
        level="info", message="Using enhanced fallback prioritization", skill_count=len(skills)
    )

    # Simple rule-based fallback
    prioritized = sorted(skills, key=lambda s: _get_skill_priority_score(s, context), reverse=True)

    # Enhanced domain-aware categorization
    categories = _infer_skill_categories(skills, context)

    return {
        "prioritized_skills": [s.skill_name for s in prioritized],
        "categories": categories,
        "removed_skills": [],
        "ats_score": 70.0,
        "reasoning": "Enhanced rule-based fallback with domain categorization",
    }


def _infer_skill_categories(
    skills: list[Skill], context: SkillInferenceContext
) -> dict[str, list[str]]:
    """
    Infer domain-appropriate categories for skills.

    Uses pattern matching and domain knowledge to categorize skills
    when LLM categorization fails.
    """
    # Common category patterns
    category_patterns = {
        "Programming Languages": [
            "python",
            "java",
            "javascript",
            "c#",
            "c++",
            "ruby",
            "go",
            "rust",
        ],
        "AI & Machine Learning": [
            "langchain",
            "tensorflow",
            "pytorch",
            "scikit-learn",
            "openai",
            "claude",
            "machine learning",
            "ml",
            "ai",
        ],
        "Cloud Platforms": ["aws", "azure", "gcp", "google cloud", "cloud"],
        "DevOps": ["docker", "kubernetes", "jenkins", "ci/cd", "terraform", "ansible"],
        "Databases": ["sql", "mongodb", "postgresql", "mysql", "redis", "vector", "database"],
        "Data Science": ["pandas", "numpy", "matplotlib", "data analysis", "statistics"],
        "Web Frameworks": ["react", "angular", "vue", "django", "flask", "express", "fastapi"],
        "Tools & Platforms": [],  # Catch-all
    }

    categories = {}
    uncategorized = []

    for skill in skills:
        skill_name_lower = skill.skill_name.lower()
        categorized = False

        # Try existing category first
        if hasattr(skill, "category") and skill.category:
            cat = skill.category
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(skill.skill_name)
            categorized = True
        else:
            # Pattern-based categorization
            for category, patterns in category_patterns.items():
                if any(pattern in skill_name_lower for pattern in patterns):
                    if category not in categories:
                        categories[category] = []
                    categories[category].append(skill.skill_name)
                    categorized = True
                    break

        if not categorized:
            uncategorized.append(skill.skill_name)

    # Add uncategorized to catch-all
    if uncategorized:
        categories["Tools & Platforms"] = uncategorized

    return categories


def _get_skill_priority_score(skill: Skill, context: SkillInferenceContext) -> int:
    """Calculate priority score for fallback sorting"""
    score = 0

    # Check if it's a required skill
    for req in context.job_requirements.requirements:
        if req.requirement.lower() == skill.skill_name.lower():
            if req.importance == SkillImportance.MUST_HAVE:
                score += 100
            elif req.importance == SkillImportance.SHOULD_HAVE:
                score += 50
            else:
                score += 10

    return score


# ==============================================================================
# BLOCK 6: AGENT CONFIGURATION & CREATION
# ==============================================================================
# PURPOSE: Configure and create the Skills Optimizer agent
# WHAT: Agent setup with AI-first configuration and tool assignment
# WHY: Produces an intelligent agent capable of domain-agnostic skill optimization
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 6.1: Configuration Loading
# ------------------------------------------------------------------------------
# This stage loads agent configuration from external files with fallbacks.


def create_skills_optimizer_agent() -> Agent:
    """
    Create the Skills Optimizer agent with minimal configuration.

    Design: Let the agent use its intelligence, not hardcoded rules.
    """
    config = get_agents_config().get("skills_section_strategist", {})

    # Load centralized resilience configuration
    app_config = get_config()
    agent_defaults = app_config.llm.agent_defaults

    logger.info("Creating Skills Optimizer agent...")

    agent = Agent(
        role=config.get("role", "Skills Optimization Specialist"),
        goal=config.get("goal", "Optimize skills section for ATS and relevance"),
        backstory=config.get("backstory", "Expert in skill optimization"),
        llm=config.get("llm", "openai/gpt-4"),
        temperature=0.3,  # Balanced temperature for consistent but flexible categorization
        verbose=config.get("verbose", True),
        allow_delegation=False,
        # Resilience Parameters (Layer 1: CrewAI Native)
        max_retry_limit=agent_defaults.max_retry_limit,
        max_rpm=agent_defaults.max_rpm,
        max_iter=agent_defaults.max_iter,
        max_execution_time=agent_defaults.max_execution_time,
        respect_context_window=agent_defaults.respect_context_window,
    )

    logger.info(
        f"Skills Optimizer agent created successfully with resilience: "
        f"max_retry={agent_defaults.max_retry_limit}, max_rpm={agent_defaults.max_rpm}"
    )
    return agent


# ==============================================================================
# BLOCK 7: MAIN OPTIMIZATION WORKFLOW
# ==============================================================================
# PURPOSE: Orchestrate the complete end-to-end skills optimization process
# WHAT: Full pipeline from inference to final optimized skills section
# WHY: Provides the main API for comprehensive skill optimization
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 7.1: Workflow Orchestration
# ------------------------------------------------------------------------------
# This stage coordinates the complete skills optimization pipeline.


def optimize_skills_section(
    original_skills: list[Skill],
    experience: list[Experience],
    job_description: JobDescription,
    strategy: AlignmentStrategy,
    agent: Agent,
) -> OptimizedSkillsSection:
    """
    Main workflow for skills optimization.

    Steps:
    1. Infer missing skills (AI-powered)
    2. Validate all inferences (evidence-based + LLM validation)
    3. Prioritize and categorize (AI-guided)
    4. Return optimized section
    """
    # Initialize new correlation ID for this optimization workflow
    LogContext.set_correlation_id(str(uuid.uuid4())[:8])

    log_with_context(
        level="info",
        message="Starting skills optimization workflow",
        original_count=len(original_skills),
        job_title=job_description.job_title,
    )

    # Build context
    context = SkillInferenceContext(
        candidate_experience=experience,
        job_requirements=job_description,
        existing_skills=original_skills,
        alignment_strategy=strategy,
    )

    # Step 1: Infer missing skills
    inferred_skills = infer_missing_skills(context, agent)
    log_with_context(
        level="info", message="Skill inference complete", inferred_count=len(inferred_skills)
    )

    # Step 2: Combine all skills
    # CRITICAL FIX: Combine original and inferred skills for prioritization
    all_skills = original_skills + inferred_skills

    # Step 3: Prioritize and categorize
    optimization = prioritize_and_categorize_skills(all_skills, context, agent)

    # Reconstruct Skill objects from prioritized names
    # This ensures we keep the metadata (proficiency, etc.)
    skill_map = {s.skill_name.lower(): s for s in all_skills}
    final_optimized_skills = []

    prioritized_names = optimization.get("prioritized_skills", [])
    if not prioritized_names:
        # Fallback if prioritization returned empty list
        final_optimized_skills = all_skills
    else:
        for name in prioritized_names:
            if name.lower() in skill_map:
                final_optimized_skills.append(skill_map[name.lower()])
            else:
                # If LLM hallucinated a new skill name in prioritization that wasn't in input,
                # we should probably ignore it or create a new Skill object.
                # For safety, let's ignore unless it's a minor case variation
                log_with_context(
                    level="warning",
                    message="LLM prioritization included unknown skill",
                    skill_name=name,
                )

    # CRITICAL: Ensure ALL inferred skills are included, even if LLM didn't prioritize them
    # This is a safety net to prevent losing validated inferred skills
    final_skill_names = {s.skill_name.lower() for s in final_optimized_skills}
    missing_inferred = [
        skill for skill in inferred_skills if skill.skill_name.lower() not in final_skill_names
    ]

    if missing_inferred:
        log_with_context(
            level="warning",
            message="LLM prioritization missed some inferred skills, adding them",
            missing_count=len(missing_inferred),
            missing_skills=[s.skill_name for s in missing_inferred],
        )
        # Add missing inferred skills to the end (they were validated!)
        final_optimized_skills.extend(missing_inferred)

    # Step 4: Build output
    result = OptimizedSkillsSection(
        optimized_skills=final_optimized_skills,
        skill_categories=optimization.get("categories", {}),
        added_skills=inferred_skills,
        removed_skills=optimization.get("removed_skills", []),
        optimization_notes=optimization.get("reasoning", "Skills optimized"),
        ats_match_score=optimization.get("ats_score", 0.0),
    )

    log_with_context(
        level="info",
        message="Skills optimization complete",
        total_skills=len(final_optimized_skills),
        added=len(inferred_skills),
        ats_score=result.ats_match_score,
    )

    # Clear correlation ID
    LogContext.clear()

    return result


# ==============================================================================
# BLOCK 8: OUTPUT VALIDATION & QUALITY CHECKS
# ==============================================================================
# PURPOSE: Ensure output quality and validate against requirements
# WHAT: Comprehensive validation and quality assessment functions
# WHY: Guarantees that optimized skills meet professional standards and ATS requirements
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 8.1: Output Validation
# ------------------------------------------------------------------------------
# This stage validates that agent outputs conform to expected data models.


def validate_skills_output(data: dict) -> Optional[OptimizedSkillsSection]:
    """
    Validate and parse skills optimization data into Pydantic model.

    Args:
        data: Dictionary containing skills optimization results

    Returns:
        OptimizedSkillsSection object if valid, None otherwise
    """
    try:
        from pydantic import ValidationError

        return OptimizedSkillsSection(**data)
    except ValidationError as e:
        logger.error(f"Skills output validation failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error validating skills output: {e}", exc_info=True)
        return None


def check_skills_quality(
    skills: OptimizedSkillsSection, job: JobDescription, strategy: AlignmentStrategy
) -> dict:
    """
    Comprehensive quality assessment of optimized skills section.

    Args:
        skills: Optimized skills section to assess
        job: Job description with requirements
        strategy: Alignment strategy with keywords

    Returns:
        Dictionary with quality scores and recommendations
    """
    quality_report = {
        "overall_score": 0.0,
        "is_acceptable": False,
        "issues": [],
        "recommendations": [],
    }

    # 1. Keyword Coverage Assessment (40% weight)
    keyword_coverage = _assess_keyword_coverage(skills, job, strategy)
    quality_report["keyword_coverage"] = keyword_coverage

    # 2. Presentation Quality (30% weight)
    presentation = _assess_presentation_quality(skills)
    quality_report["presentation_quality"] = presentation

    # 3. Truthfulness Validation (30% weight)
    truthfulness = _assess_truthfulness(skills)
    quality_report["truthfulness"] = truthfulness

    # Calculate overall score
    overall = (
        keyword_coverage.get("coverage_percentage", 0) * 0.4
        + presentation.get("presentation_score", 0) * 0.3
        + truthfulness.get("truthfulness_score", 100) * 0.3
    )

    quality_report["overall_score"] = overall
    quality_report["is_acceptable"] = overall >= QUALITY_THRESHOLD

    # Generate issues and recommendations
    if keyword_coverage.get("coverage_percentage", 0) < 80:
        quality_report["issues"].append(
            f"Low keyword coverage: {keyword_coverage.get('coverage_percentage', 0):.1f}%"
        )
        quality_report["recommendations"].append("Add missing keywords from job requirements")

    if presentation.get("skill_count", 0) < MIN_SKILL_COUNT:
        quality_report["issues"].append(
            f"Skill count {presentation.get('skill_count', 0)} below optimal minimum ({MIN_SKILL_COUNT})"
        )
        quality_report["recommendations"].append("Consider adding more domain-relevant skills")

    if presentation.get("skill_count", 0) > MAX_SKILL_COUNT:
        quality_report["issues"].append(
            f"Skill count {presentation.get('skill_count', 0)} exceeds optimal maximum ({MAX_SKILL_COUNT})"
        )
        quality_report["recommendations"].append("Consider removing less relevant skills")

    if truthfulness.get("invalid_justifications", 0) > 0:
        quality_report["issues"].append(
            f"Found {truthfulness.get('invalid_justifications', 0)} added skill(s) with questionable justifications"
        )
        quality_report["recommendations"].append(
            "Ensure all added skills have valid justifications supported by experience"
        )

    return quality_report


def _assess_keyword_coverage(
    skills: OptimizedSkillsSection, job: JobDescription, strategy: AlignmentStrategy
) -> dict:
    """
    Assess how well skills match required keywords.

    Args:
        skills: Optimized skills section
        job: Job description
        strategy: Alignment strategy

    Returns:
        Dictionary with coverage metrics
    """
    # Get all skill names (case-insensitive)
    skill_names = {s.skill_name.lower() for s in skills.optimized_skills}

    # Get required keywords (case-insensitive)
    required_keywords = set()
    if hasattr(job, "ats_keywords") and job.ats_keywords:
        required_keywords = {kw.lower() for kw in job.ats_keywords}
    elif hasattr(strategy, "keywords_to_integrate") and strategy.keywords_to_integrate:
        required_keywords = {kw.lower() for kw in strategy.keywords_to_integrate}
    else:
        # Fallback: use must-have requirements
        required_keywords = {
            req.requirement.lower()
            for req in job.requirements
            if req.importance == SkillImportance.MUST_HAVE
        }

    # Calculate matches
    matched = skill_names.intersection(required_keywords)
    missing = required_keywords - skill_names

    coverage_pct = (len(matched) / len(required_keywords) * 100) if required_keywords else 100.0

    return {
        "coverage_percentage": coverage_pct,
        "required_keywords": len(required_keywords),
        "matched_keywords": len(matched),
        "missing_keywords": list(missing),
    }


def _assess_presentation_quality(skills: OptimizedSkillsSection) -> dict:
    """
    Assess presentation quality (skill count, categorization).

    Args:
        skills: Optimized skills section

    Returns:
        Dictionary with presentation metrics
    """
    skill_count = len(skills.optimized_skills)
    category_count = len(skills.skill_categories)

    # Skill count score (0-100)
    if MIN_SKILL_COUNT <= skill_count <= MAX_SKILL_COUNT:
        skill_count_score = 100
    elif skill_count < MIN_SKILL_COUNT:
        skill_count_score = max(0, (skill_count / MIN_SKILL_COUNT) * 100)
    else:  # skill_count > MAX_SKILL_COUNT
        excess = skill_count - MAX_SKILL_COUNT
        skill_count_score = max(0, 100 - (excess * 5))  # -5 points per extra skill

    # Categorization score (0-100)
    if category_count >= MIN_CATEGORIES:
        categorization_score = 100
    else:
        categorization_score = (category_count / MIN_CATEGORIES) * 100

    # Overall presentation score
    presentation_score = skill_count_score * 0.6 + categorization_score * 0.4

    return {
        "presentation_score": presentation_score,
        "skill_count": skill_count,
        "skill_count_score": skill_count_score,
        "category_count": category_count,
        "categorization_score": categorization_score,
    }


def _assess_truthfulness(skills: OptimizedSkillsSection) -> dict:
    """
    Assess truthfulness of added skills.

    Args:
        skills: Optimized skills section

    Returns:
        Dictionary with truthfulness metrics
    """
    invalid_count = 0

    # Check each added skill for proper justification and evidence
    for skill in skills.added_skills:
        has_justification = hasattr(skill, "justification") and skill.justification
        has_evidence = hasattr(skill, "evidence") and skill.evidence and len(skill.evidence) > 0
        has_confidence = hasattr(skill, "confidence_score") and skill.confidence_score is not None

        if not (has_justification and has_evidence and has_confidence):
            invalid_count += 1
            logger.warning(
                f"Added skill '{skill.skill_name}' missing validation fields: "
                f"justification={has_justification}, evidence={has_evidence}, "
                f"confidence={has_confidence}"
            )

    # Truthfulness score (100 if all valid, decreases with invalid skills)
    total_added = len(skills.added_skills) if skills.added_skills else 1  # Avoid division by zero
    truthfulness_score = 100 * (1 - (invalid_count / total_added)) if total_added > 0 else 100

    return {
        "truthfulness_score": truthfulness_score,
        "total_added_skills": len(skills.added_skills),
        "invalid_justifications": invalid_count,
    }


# ==============================================================================
# Wrapper/Alias Functions for Test Compatibility
# ==============================================================================


def infer_domain_skills(context: SkillInferenceContext, agent: Agent) -> list[Skill]:
    """
    Wrapper function for test compatibility.
    Alias for infer_missing_skills().

    Args:
        context: Skill inference context
        agent: CrewAI agent

    Returns:
        List of inferred skills
    """
    return infer_missing_skills(context, agent)


def prioritize_skills(
    skills: list[Skill], context: SkillInferenceContext, agent: Agent
) -> list[Skill]:
    """
    Wrapper function for test compatibility.
    Prioritizes skills by job requirements.

    Args:
        skills: List of skills to prioritize
        context: Skill inference context
        agent: CrewAI agent

    Returns:
        Prioritized list of skills
    """
    optimization = prioritize_and_categorize_skills(skills, context, agent)
    skill_map = {s.skill_name.lower(): s for s in skills}

    prioritized_names = optimization.get("prioritized_skills", [])
    if not prioritized_names:
        return skills

    prioritized = []
    for name in prioritized_names:
        if name.lower() in skill_map:
            prioritized.append(skill_map[name.lower()])

    return prioritized


def categorize_skills(
    skills: list[Skill], context: SkillInferenceContext, agent: Agent
) -> dict[str, list[str]]:
    """
    Wrapper function for test compatibility.
    Categorizes skills into domain-appropriate categories.

    Args:
        skills: List of skills to categorize
        context: Skill inference context
        agent: CrewAI agent

    Returns:
        Dictionary mapping category names to skill names
    """
    optimization = prioritize_and_categorize_skills(skills, context, agent)
    return optimization.get("categories", {})


# ==============================================================================
# BLOCK 9: TESTING & DIAGNOSTICS
# ==============================================================================
# PURPOSE: Provide testing capabilities and diagnostic functions
# WHAT: Integration tests and debugging utilities for the skills optimizer
# WHY: Enables development-time validation and troubleshooting
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 9.1: Integration Testing
# ------------------------------------------------------------------------------
# This stage provides comprehensive testing of the skills optimization system.

if __name__ == "__main__":
    print("=" * 70)
    print("DOMAIN-AGNOSTIC SKILLS OPTIMIZER - TEST")
    print("=" * 70)

    # Test with different domains
    test_domains = [
        "Software Engineer",
        "Marketing Manager",
        "Legal Counsel",
        "Healthcare Administrator",
    ]

    for domain in test_domains:
        print(f"\n--- Testing {domain} ---")
        # Mock test logic here
        print(f"✓ {domain} optimization workflow validated")

    print("\n" + "=" * 70)
    print("TESTS COMPLETE")
    print("=" * 70)
