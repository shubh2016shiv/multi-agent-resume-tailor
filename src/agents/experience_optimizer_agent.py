"""
Experience Section Optimizer Agent
----------------------------------

This module defines the fifth agent in our workflow: the Experience Section Optimizer.
This agent is responsible for rewriting work experience entries to align with job
requirements while maintaining complete truthfulness. It incorporates keywords naturally,
quantifies achievements, and reorders content by relevance.

AGENT DESIGN PRINCIPLES:
- Single Responsibility: Optimize work experience entries only
- Modularity: Clear separation between agent creation, optimization, and validation
- Robustness: Comprehensive error handling with graceful degradation
- Type Safety: Uses Pydantic models for validated inputs and outputs
- Observability: Detailed logging at every step for debugging

WORKFLOW:
1. Receive Resume, JobDescription, and AlignmentStrategy as input
2. Follow the experience_guidance from the strategy
3. For each experience entry:
   - Reorder bullets by relevance to target job
   - Incorporate job-relevant keywords naturally
   - Reframe achievements to emphasize relevant aspects
   - Quantify achievements using CAR (Challenge-Action-Result) format
   - Use strong action verbs (Led, Architected, Increased)
4. Return optimized experience section with metadata

KEY OPTIMIZATION PRINCIPLES:
- Truthfulness: Never fabricate or exaggerate
- Relevance: Most relevant achievements first
- Quantification: Include metrics wherever possible
- Action-Oriented: Strong verbs that convey impact
- Keyword-Rich: Naturally integrate ATS keywords
- CAR Format: Challenge-Action-Result structure

CONTENT GENERATION STRATEGY:
- Use gemini-2.5-flash-lite for cost-effective rewriting with iterative improvement
- Moderate temperature (0.5) for balanced creativity
- Focus on impact and outcomes over tasks
- Maintain authenticity and accuracy
- Optimize for both ATS and human readers

OUTPUT VALIDATION:
- Action verb strength check
- Quantification rate verification
- Keyword integration analysis
- Relevance ordering validation
- Bullet count compliance (3-6 per entry)
"""

import json
from datetime import date

from crewai import Agent
from crewai.tools import tool
from pydantic import BaseModel, ConfigDict, Field, ValidationError

# Handle imports for both package usage and direct script execution
try:
    from src.core.config import get_agents_config
    from src.core.logger import get_logger
    from src.data_models.resume import Experience
    from src.data_models.strategy import AlignmentStrategy
except ImportError:
    # Fallback for when running this file directly
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.core.config import get_agents_config
    from src.core.logger import get_logger
    from src.data_models.resume import Experience
    from src.data_models.strategy import AlignmentStrategy

logger = get_logger(__name__)


# ==============================================================================
# STAGE 1: Configuration for Iterative Improvement Framework
# ==============================================================================
# PURPOSE: Configure the agentic iterative improvement system
# WHAT: Global settings that control the self-improvement behavior
# WHY: These parameters allow tuning the trade-off between quality and speed
#
# AGENTIC BEHAVIOR:
# The agent uses these thresholds to autonomously decide when to:
# 1. Continue improving (score < QUALITY_THRESHOLD)
# 2. Stop iterating (score >= QUALITY_THRESHOLD OR iterations >= MAX)
# 3. Request human feedback (optional, not implemented yet)
#
# TUNING GUIDE:
# - QUALITY_THRESHOLD: Higher = stricter quality bar (more iterations)
#   • 85+ = Professional/Production ready
#   • 70-84 = Good but could improve
#   • <70 = Needs significant work
# - MAX_IMPROVEMENT_ITERATIONS: Higher = more refinement opportunities
#   • 1 = Single-pass optimization (faster)
#   • 3 = Balanced (recommended)
#   • 5+ = Deep refinement (slower, diminishing returns)
# ==============================================================================

# Toggle iterative improvement on/off for backward compatibility
ENABLE_ITERATIVE_IMPROVEMENT = True

# Maximum number of improvement iterations per experience entry
MAX_IMPROVEMENT_ITERATIONS = 3

# Quality score threshold to stop iterating (0-100)
QUALITY_THRESHOLD = 85


# ==============================================================================
# STAGE 2: CrewAI Tool Infrastructure for Self-Evaluation
# ==============================================================================
# PURPOSE: Provide the LLM agent with a self-evaluation capability
# WHAT: A CrewAI @tool that the agent can invoke to evaluate its own output
# WHY: Enables true agentic behavior - the agent critiques and improves itself
#
# AGENTIC WORKFLOW:
# 1. Agent generates initial bullets (via LLM reasoning)
# 2. Agent invokes evaluate_experience_bullets tool on its output
# 3. Tool returns objective quality score + specific critique
# 4. Agent reads critique and regenerates bullets addressing issues
# 5. Loop continues until threshold met or max iterations reached
#
# KEY DESIGN DECISIONS:
# - Tool is DETERMINISTIC (no LLM calls) for consistent scoring
# - Returns ACTIONABLE critique (not just "bad" but "add metrics to bullet 2")
# - Uses JSON for structured communication with agent
# - Scoped access to evaluation functions via imports
# ==============================================================================

# Global tracking for observability (thread-safe for single-threaded execution)
_iteration_tracker = {
    "current_iteration": 0,
    "call_count": 0,
    "iteration_history": {},
    "tool_call_log": [],  # Log of all tool calls with inputs/outputs
    "max_calls_per_session": 50,  # Safety limit to prevent excessive API calls
}


def reset_iteration_tracker():
    """
    Reset the iteration tracker for a new optimization session.

    Call this before starting a new experience optimization to ensure
    clean tracking and observability.
    """
    global _iteration_tracker
    _iteration_tracker = {
        "current_iteration": 0,
        "call_count": 0,
        "iteration_history": {},
        "tool_call_log": [],
        "max_calls_per_session": 50,
    }
    logger.info("[ITERATION TRACKER] Reset iteration tracker for new session")


def get_iteration_tracker_state() -> dict:
    """
    Get current state of iteration tracker for debugging/observability.

    Returns:
        Dictionary with current iteration state including:
        - call_count: Total number of tool calls
        - current_iteration: Current iteration number (estimated)
        - tool_call_log: List of recent tool calls with inputs/outputs
    """
    return _iteration_tracker.copy()


def log_iteration_tracker_summary():
    """
    Log a comprehensive summary of all tool calls for observability.

    This function provides complete visibility into:
    - How many times the tool was called
    - What bullets were input in each call
    - What scores were returned
    - What critique was generated
    - Whether iterations are improving
    """
    global _iteration_tracker

    logger.info("\n" + "=" * 80)
    logger.info("ITERATION TRACKER SUMMARY - Complete Observability")
    logger.info("=" * 80)

    call_count = _iteration_tracker["call_count"]
    tool_calls = _iteration_tracker["tool_call_log"]

    logger.info(f"Total Tool Calls: {call_count}")
    logger.info(f"Tracked Tool Calls: {len(tool_calls)}")

    if not tool_calls:
        logger.warning(
            "[OBSERVABILITY] No tool calls tracked yet. Agent may not be calling the tool."
        )
        logger.info("=" * 80 + "\n")
        return

    logger.info("\nDetailed Tool Call History:")
    logger.info("-" * 80)

    for i, call_record in enumerate(tool_calls, 1):
        logger.info(f"\nTool Call #{call_record.get('call_id', i)}:")
        logger.info(f"  Input Bullets ({call_record.get('bullet_count', 0)} bullets):")
        for j, bullet in enumerate(call_record.get("input_bullets", [])[:3], 1):  # Show first 3
            logger.info(f"    {j}. {bullet[:80]}{'...' if len(bullet) > 80 else ''}")

        if "output_score" in call_record:
            logger.info(f"  Output Score: {call_record['output_score']}/100")
            logger.info(f"  Meets Threshold: {call_record.get('output_meets_threshold', False)}")
            logger.info(f"  Issues Found: {call_record.get('output_issues_count', 0)}")
            if call_record.get("output_critique_preview"):
                logger.info(
                    f"  Critique Preview: {call_record['output_critique_preview'][:150]}..."
                )

        # Check if bullets improved from previous call
        if i > 1:
            prev_score = tool_calls[i - 2].get("output_score", 0)
            curr_score = call_record.get("output_score", 0)
            if curr_score > prev_score:
                logger.info(
                    f"  [IMPROVEMENT] Score improved: {prev_score} -> {curr_score} (+{curr_score - prev_score})"
                )
            elif curr_score < prev_score:
                logger.warning(
                    f"  [DEGRADATION] Score decreased: {prev_score} -> {curr_score} ({curr_score - prev_score})"
                )
            else:
                logger.info(f"  [NO CHANGE] Score unchanged: {prev_score}")

    logger.info("\n" + "=" * 80)
    logger.info("END ITERATION TRACKER SUMMARY")
    logger.info("=" * 80 + "\n")


def validate_iteration_progress() -> dict:
    """
    Validate that iteration is actually happening and progressing.

    Returns:
        Dictionary with validation results:
        - is_iterating: bool - Whether tool is being called multiple times
        - tool_call_count: int - Number of tool calls
        - has_improvement: bool - Whether scores are improving
        - meets_expectations: bool - Whether iteration meets minimum expectations
        - recommendations: list[str] - Recommendations for improvement
    """
    global _iteration_tracker

    tool_calls = _iteration_tracker["tool_call_log"]
    call_count = _iteration_tracker["call_count"]

    validation = {
        "is_iterating": len(tool_calls) > 1,
        "tool_call_count": call_count,
        "tracked_calls": len(tool_calls),
        "has_improvement": False,
        "meets_expectations": False,
        "recommendations": [],
    }

    if len(tool_calls) < 2:
        validation["recommendations"].append(
            f"Only {len(tool_calls)} tool call(s) tracked. Expected at least 2-3 for iterative improvement."
        )
        validation["recommendations"].append(
            "Agent may not be following iterative workflow. Check task instructions."
        )
    else:
        # Check for score improvement
        scores = [call.get("output_score", 0) for call in tool_calls if "output_score" in call]
        if len(scores) >= 2:
            if scores[-1] > scores[0]:
                validation["has_improvement"] = True
            else:
                validation["recommendations"].append(
                    f"Scores not improving: {scores[0]} -> {scores[-1]}"
                )

        # Check if threshold was met
        if tool_calls:
            last_call = tool_calls[-1]
            if last_call.get("output_meets_threshold", False):
                validation["meets_expectations"] = True
            elif len(tool_calls) >= MAX_IMPROVEMENT_ITERATIONS:
                validation["recommendations"].append(
                    f"Max iterations ({MAX_IMPROVEMENT_ITERATIONS}) reached but threshold not met."
                )

    if validation["is_iterating"] and validation["has_improvement"]:
        validation["meets_expectations"] = True

    return validation


@tool("Evaluate Experience Bullets")
def evaluate_experience_bullets(bullets_json: str, keywords: str, strategy_json: str) -> str:
    """
    STAGE 2A: Self-Evaluation Tool - Agent's Quality Assessment Interface

    PURPOSE: Enable the LLM agent to objectively evaluate its own generated
    bullets and receive actionable feedback for iterative improvement.

    THIS IS THE CRITICAL TOOL THAT ENABLES ITERATIVE IMPROVEMENT:
    The agent calls this tool after generating bullets to:
    1. Get objective quality scores (0-100)
    2. Receive specific issues identified
    3. Get actionable critique for improvement
    4. Know if quality threshold (85) is met

    HOW IT FITS INTO THE ITERATIVE WORKFLOW:

    ITERATION 1:
    [1.1] Agent generates initial bullets (via LLM reasoning)
    [1.2] Agent calls this tool -> receives score (e.g., 65/100)
    [1.3] Agent reads critique -> "Add metrics, specify method..."
    [1.4] Agent decides: score < 85 -> regenerate

    ITERATION 2:
    [2.1] Agent regenerates bullets addressing critique
    [2.2] Agent calls this tool again -> receives new score (e.g., 78/100)
    [2.3] Agent reads updated critique -> "Add scope indicators..."
    [2.4] Agent decides: score < 85 -> regenerate again

    ITERATION 3:
    [3.1] Agent regenerates bullets with all improvements
    [3.2] Agent calls this tool again -> receives final score (e.g., 88/100)
    [3.3] Agent reads: meets_threshold = True
    [3.4] Agent decides: STOP (quality threshold met)

    EXECUTION FLOW (Sub-Stages):
    [2A.1] Parse Input: Deserialize bullets, keywords, and strategy
    [2A.2] Evaluate Each Bullet: Call evaluate_single_bullet() for each
    [2A.3] Aggregate Scores: Calculate average score across all bullets
    [2A.4] Generate Critique: Create actionable feedback for low-scoring bullets
    [2A.5] Check Threshold: Determine if quality bar is met
    [2A.6] Return JSON: Serialize results for agent consumption

    Args:
        bullets_json: JSON array string of bullet strings (REQUIRED: Must be a JSON string, not a Python list!)
                     CORRECT: '["Led team...", "Built system..."]' (JSON string)
                     WRONG: ["Led team...", "Built system..."] (Python list)
                     IMPORTANT: Use json.dumps(your_list) to convert your list to JSON string before calling this tool.
        keywords: Comma-separated keywords to integrate (e.g., "Python,AWS,Docker")
        strategy_json: JSON string of AlignmentStrategy object (use json.dumps())

    Returns:
        JSON string with: {
            "average_score": int (0-100),
            "per_bullet_scores": [int],
            "issues": [str],
            "critique": str (actionable improvement suggestions),
            "meets_threshold": bool (score >= 85)
        }

    Example (CORRECT usage):
        import json
        bullets = ["Managed team projects", "Built features"]
        result = evaluate_experience_bullets(
            bullets_json=json.dumps(bullets),  # Convert list to JSON string
            keywords="Python,AWS,Kubernetes",
            strategy_json=json.dumps(strategy_dict)  # Convert dict to JSON string
        )
    """
    global _iteration_tracker

    # Track tool calls for observability
    _iteration_tracker["call_count"] += 1
    call_id = _iteration_tracker["call_count"]

    # SAFETY: Prevent excessive API calls that could cause "too many requests" errors
    max_calls = _iteration_tracker.get("max_calls_per_session", 50)
    if call_id > max_calls:
        error_msg = (
            f"Tool call limit exceeded ({call_id} > {max_calls}). "
            f"This may indicate an infinite loop or excessive retries. "
            f"Check agent's iteration logic."
        )
        logger.error("=" * 80)
        logger.error(f"[SAFETY LIMIT] {error_msg}")
        logger.error("=" * 80)
        return json.dumps(
            {
                "error": error_msg,
                "error_type": "CallLimitExceeded",
                "average_score": 0,
                "per_bullet_scores": [],
                "issues": [error_msg],
                "critique": "Tool call limit exceeded. Stop iterating and use current best bullets.",
                "meets_threshold": False,
                "call_id": call_id,
            }
        )

    try:
        # [SUB-STAGE 2A.1] Parse Input
        # WHAT: Deserialize JSON inputs into Python objects
        # WHY: Tool receives strings from agent; need structured data
        # NOTE: CrewAI sometimes passes lists directly instead of JSON strings,
        #       so we handle both cases for robustness

        # OBSERVABILITY: Log input bullets for this iteration
        logger.info("=" * 80)
        logger.info(f"[ITERATION OBSERVABILITY] Tool Call #{call_id}")
        logger.info("=" * 80)

        # Parse JSON string to list
        # Expected format: '["bullet1", "bullet2", ...]'
        if not isinstance(bullets_json, str):
            error_msg = (
                f"bullets_json must be a JSON string, not {type(bullets_json).__name__}. "
                f"Use json.dumps(your_list) to convert your list to a JSON string before calling this tool."
            )
            logger.error(f"[TOOL ERROR] {error_msg}")
            return json.dumps(
                {
                    "error": error_msg,
                    "error_type": "InvalidInputType",
                    "average_score": 0,
                    "per_bullet_scores": [],
                    "issues": [error_msg],
                    "critique": "Convert your Python list to JSON string using json.dumps(bullets) before calling this tool.",
                    "meets_threshold": False,
                    "call_id": call_id,
                    "help": "Example: evaluate_experience_bullets(bullets_json=json.dumps(my_bullets_list), ...)",
                }
            )

        try:
            # JSON string - parse it (expected path)
            bullets = json.loads(bullets_json)
            logger.info(
                f"[ITERATION INPUT] Parsed bullets from JSON string (length: {len(bullets)})"
            )
        except json.JSONDecodeError as e:
            logger.error(f"[TOOL ERROR] Invalid JSON in bullets_json: {e}")
            logger.error(
                f"[TOOL ERROR] Received bullets_json (first 200 chars): {bullets_json[:200]}"
            )
            return json.dumps(
                {
                    "error": f"Invalid JSON format: {str(e)}",
                    "error_type": "JSONDecodeError",
                    "average_score": 0,
                    "per_bullet_scores": [],
                    "issues": [f"JSON parsing failed: {str(e)}"],
                    "critique": 'Ensure bullets_json is a valid JSON array string like \'["bullet1", "bullet2"]\'',
                    "meets_threshold": False,
                    "call_id": call_id,
                }
            )

        keywords_list = [k.strip() for k in keywords.split(",")] if keywords else []

        try:
            strategy = AlignmentStrategy(**json.loads(strategy_json))
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"[TOOL ERROR] Invalid strategy_json: {e}")
            logger.error(
                f"[TOOL ERROR] Received strategy_json (first 200 chars): {strategy_json[:200]}"
            )
            raise

        # OBSERVABILITY: Log what bullets are being evaluated
        logger.info(f"[ITERATION INPUT] Evaluating {len(bullets)} bullets:")
        for i, bullet in enumerate(bullets, 1):
            logger.info(f"  Bullet {i}: {bullet[:100]}{'...' if len(bullet) > 100 else ''}")
        logger.info(f"[ITERATION INPUT] Keywords: {keywords_list}")
        logger.info(f"[ITERATION INPUT] Strategy keywords: {strategy.keywords_to_integrate[:5]}")

        # Track this tool call for observability
        tool_call_record = {
            "call_id": call_id,
            "timestamp": str(date.today()),  # Simple timestamp
            "input_bullets": bullets.copy(),
            "input_keywords": keywords_list.copy(),
            "bullet_count": len(bullets),
        }
        _iteration_tracker["tool_call_log"].append(tool_call_record)

        # Import evaluation functions (they're defined later in this file)
        # This circular dependency is OK because @tool decorator delays execution

        # ---------------------------------------------------------------------
        # [SUB-STAGE 2A.2] Evaluate Each Bullet
        # ---------------------------------------------------------------------
        # WHAT: Score each bullet individually using comprehensive checks
        # WHY: Per-bullet feedback allows targeted improvements
        # CALLS: evaluate_single_bullet() - see STAGE 3 below
        evaluations = []
        for bullet in bullets:
            eval_result = evaluate_single_bullet(bullet, strategy, keywords_list)
            evaluations.append(eval_result)

        # OBSERVABILITY: Log per-bullet scores
        logger.info("[ITERATION SCORES] Per-bullet quality scores:")
        for i, (_bullet, eval_result) in enumerate(zip(bullets, evaluations, strict=True), 1):
            logger.info(
                f"  Bullet {i}: {eval_result['score']}/100 - Issues: {len(eval_result['issues'])}"
            )
            if eval_result["issues"]:
                logger.info(f"    -> {eval_result['issues'][:2]}")  # Show first 2 issues

        # ---------------------------------------------------------------------
        # [SUB-STAGE 2A.3] Aggregate Scores
        # ---------------------------------------------------------------------
        # WHAT: Calculate overall quality by averaging individual scores
        # WHY: Single metric for agent to decide continue/stop
        avg_score = sum(e["score"] for e in evaluations) / len(evaluations) if evaluations else 0

        # ---------------------------------------------------------------------
        # [SUB-STAGE 2A.4] Generate Critique
        # ---------------------------------------------------------------------
        # WHAT: Create actionable improvement instructions
        # WHY: Agent needs specific guidance, not just a score
        # CALLS: generate_bullet_critique() - see STAGE 3B below
        critique_parts = []
        # Only critique bullets below threshold (focused feedback)
        for i, eval_result in enumerate(evaluations):
            if eval_result["score"] < QUALITY_THRESHOLD:
                critique = generate_bullet_critique(bullets[i], eval_result, strategy)
                critique_parts.append(f"Bullet {i + 1}: {critique}")

        # Collect all issues for transparency
        issues = []
        for i, eval_result in enumerate(evaluations):
            if eval_result["issues"]:
                issues.append(f"Bullet {i + 1}: {'; '.join(eval_result['issues'])}")

        # ---------------------------------------------------------------------
        # [SUB-STAGE 2A.5] Check Threshold
        # ---------------------------------------------------------------------
        # WHAT: Determine if quality bar is met
        # WHY: Agent uses this boolean to decide iteration continuation
        # CRITICAL: This drives the agentic loop termination condition
        result = {
            "average_score": int(avg_score),
            "per_bullet_scores": [e["score"] for e in evaluations],
            "issues": issues,
            "critique": " | ".join(critique_parts)
            if critique_parts
            else "All bullets meet quality standards",
            "meets_threshold": avg_score >= QUALITY_THRESHOLD,  # DECISION POINT
        }

        # OBSERVABILITY: Log comprehensive evaluation results
        logger.info(f"[ITERATION RESULT] Average Score: {int(avg_score)}/100")
        logger.info(f"[ITERATION RESULT] Quality Threshold: {QUALITY_THRESHOLD}/100")
        logger.info(f"[ITERATION RESULT] Meets Threshold: {result['meets_threshold']}")
        logger.info(f"[ITERATION RESULT] Total Issues Found: {len(issues)}")

        if result["critique"] and result["critique"] != "All bullets meet quality standards":
            logger.info("[ITERATION CRITIQUE] Improvement Suggestions:")
            # Split critique by bullet and log each
            critique_lines = result["critique"].split(" | ")
            for critique_line in critique_lines[:5]:  # Show first 5 critique items
                logger.info(
                    f"  -> {critique_line[:150]}{'...' if len(critique_line) > 150 else ''}"
                )

        # OBSERVABILITY: Decision guidance for agent
        if result["meets_threshold"]:
            logger.info("[ITERATION DECISION] STOP ITERATING - Quality threshold met!")
            logger.info("[ITERATION DECISION] Final bullets are ready. Include in output.")
        else:
            logger.info(
                f"[ITERATION DECISION] CONTINUE ITERATING - Score {int(avg_score)} < {QUALITY_THRESHOLD}"
            )
            logger.info(
                "[ITERATION DECISION] Agent should regenerate bullets addressing the critique above."
            )
            logger.info("[ITERATION DECISION] Then call this tool again with improved bullets.")

        # Update tool call record with results
        if _iteration_tracker["tool_call_log"]:
            _iteration_tracker["tool_call_log"][-1].update(
                {
                    "output_score": int(avg_score),
                    "output_meets_threshold": result["meets_threshold"],
                    "output_issues_count": len(issues),
                    "output_critique_preview": result["critique"][:200]
                    if result["critique"]
                    else "",
                }
            )

        logger.info("=" * 80)

        # ---------------------------------------------------------------------
        # [SUB-STAGE 2A.6] Return JSON
        # ---------------------------------------------------------------------
        # WHAT: Serialize result as JSON string
        # WHY: CrewAI tools must return strings; agent will parse
        return json.dumps(result, indent=2)

    except json.JSONDecodeError as e:
        # OBSERVABILITY: Log JSON parsing errors with context
        logger.error("=" * 80)
        logger.error(f"[TOOL ERROR] JSON Parse Error in Tool Call #{call_id}")
        logger.error(f"[TOOL ERROR] Error: {e}")
        logger.error(f"[TOOL ERROR] bullets_json type: {type(bullets_json)}")
        logger.error(
            f"[TOOL ERROR] bullets_json length: {len(bullets_json) if isinstance(bullets_json, str) else 'N/A'}"
        )
        logger.error(f"[TOOL ERROR] bullets_json preview: {str(bullets_json)[:300]}")
        logger.error("=" * 80)

        # Return clear error message to prevent retry loops
        return json.dumps(
            {
                "error": f"Invalid JSON format in bullets_json: {str(e)}",
                "error_type": "JSONDecodeError",
                "average_score": 0,
                "per_bullet_scores": [],
                "issues": [f"JSON parsing failed: {str(e)}"],
                "critique": 'Fix JSON format: bullets_json must be a JSON array string like \'["bullet1", "bullet2"]\'',
                "meets_threshold": False,
                "call_id": call_id,
            }
        )

    except ValidationError as e:
        # OBSERVABILITY: Log validation errors
        logger.error("=" * 80)
        logger.error(f"[TOOL ERROR] Validation Error in Tool Call #{call_id}")
        logger.error(f"[TOOL ERROR] Error: {e}")
        logger.error(f"[TOOL ERROR] Validation errors: {e.errors()}")
        logger.error("=" * 80)

        return json.dumps(
            {
                "error": f"Strategy validation failed: {str(e)}",
                "error_type": "ValidationError",
                "average_score": 0,
                "per_bullet_scores": [],
                "issues": [f"Strategy validation failed: {str(e)}"],
                "critique": "Fix strategy_json: must include all required AlignmentStrategy fields",
                "meets_threshold": False,
                "call_id": call_id,
            }
        )

    except Exception as e:
        # OBSERVABILITY: Log unexpected errors
        logger.error("=" * 80)
        logger.error(f"[TOOL ERROR] Unexpected Error in Tool Call #{call_id}")
        logger.error(f"[TOOL ERROR] Error type: {type(e).__name__}")
        logger.error(f"[TOOL ERROR] Error message: {str(e)}")
        logger.error("=" * 80)
        logger.error(f"Error in evaluate_experience_bullets: {e}", exc_info=True)

        return json.dumps(
            {
                "error": str(e),
                "error_type": type(e).__name__,
                "average_score": 0,
                "per_bullet_scores": [],
                "issues": [f"Evaluation failed: {str(e)}"],
                "critique": "Unable to evaluate bullets due to error. Check logs for details.",
                "meets_threshold": False,
                "call_id": call_id,
            }
        )


# ==============================================================================
# Output Model for Optimized Experience Section
# ==============================================================================


class OptimizedExperienceSection(BaseModel):
    """
    Structured output containing optimized work experience entries.

    This model ensures all experience entries are properly optimized and includes
    metadata for downstream validation and quality assurance.
    """

    optimized_experiences: list[Experience] = Field(
        ...,
        description="List of optimized experience entries from the resume",
        min_length=1,
    )

    optimization_notes: str = Field(
        default="",
        description="Overall notes about optimization decisions and trade-offs",
    )

    keywords_integrated: list[str] = Field(
        default_factory=list,
        description="List of keywords from the strategy that were integrated",
    )

    relevance_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Relevance scores per experience entry (key: company_name, value: score 0-100)",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "optimized_experiences": [
                    {
                        "job_title": "Senior Software Engineer",
                        "company_name": "Tech Corp",
                        "start_date": "2020-01-15",
                        "end_date": None,
                        "is_current_position": True,
                        "location": "San Francisco, CA",
                        "description": "Led development of cloud-native microservices platform",
                        "achievements": [
                            "Architected scalable microservices infrastructure using Python and AWS, reducing deployment time by 65%",
                            "Led cross-functional team of 8 engineers to deliver $2M cost-saving automation initiative",
                            "Implemented CI/CD pipeline with Docker and Kubernetes, improving release frequency by 300%",
                        ],
                        "skills_used": ["Python", "AWS", "Docker", "Kubernetes"],
                    }
                ],
                "optimization_notes": "Emphasized cloud and Python experience per strategy guidance",
                "keywords_integrated": ["Python", "AWS", "microservices", "Docker"],
                "relevance_scores": {"Tech Corp": 95.0},
            }
        }
    )


class BulletDraft(BaseModel):
    """
    A single iteration of bullet point improvement.

    This model captures the state of bullets at each iteration,
    including quality metrics and identified issues.
    """

    iteration: int = Field(..., description="Iteration number (1-indexed)", ge=1)

    content: str = Field(
        ..., description="The bullet content at this iteration (all bullets joined)"
    )

    quality_score: int = Field(
        ..., description="Average quality score across all bullets (0-100)", ge=0, le=100
    )

    issues_found: list[str] = Field(
        default_factory=list, description="List of issues identified in this iteration"
    )

    critique: str = Field(default="", description="Self-critique and improvement suggestions")


class IterativeExperienceOptimization(BaseModel):
    """
    Extended output model with iteration history for self-improvement tracking.

    This model extends OptimizedExperienceSection with iteration tracking,
    allowing us to see how bullets improved over multiple iterations.
    """

    optimized_experiences: list[Experience] = Field(
        ...,
        description="List of optimized experience entries from the resume",
        min_length=1,
    )

    optimization_notes: str = Field(
        default="",
        description="Overall notes about optimization decisions and trade-offs",
    )

    keywords_integrated: list[str] = Field(
        default_factory=list,
        description="List of keywords from the strategy that were integrated",
    )

    relevance_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Relevance scores per experience entry (key: company_name, value: score 0-100)",
    )

    # NEW: Iteration tracking
    iteration_history: dict[str, list[BulletDraft]] = Field(
        default_factory=dict,
        description="Per-company iteration history: {company_name: [BulletDraft, ...]}",
    )

    iterations_used: int = Field(
        default=0, description="Total number of iterations performed", ge=0
    )

    final_quality_score: int = Field(
        default=0,
        description="Final average quality score across all bullets (0-100)",
        ge=0,
        le=100,
    )


# ==============================================================================
# Agent Configuration Loading
# ==============================================================================


def _load_agent_config() -> dict:
    """
    Load the agent configuration from agents.yaml.

    This function provides a single point of configuration loading with
    proper error handling. If the config fails to load, it returns sensible
    defaults so the agent can still function.

    Returns:
        Dictionary containing agent configuration (role, goal, backstory, etc.)

    Design Note:
        Separating config loading into its own function makes the code more
        modular and testable. We can mock this function in tests.
    """
    try:
        agents_config = get_agents_config()
        config = agents_config.get("experience_section_optimizer", {})

        # Validate that required fields are present
        required_fields = ["role", "goal", "backstory"]
        missing_fields = [f for f in required_fields if f not in config]

        if missing_fields:
            logger.warning(f"Agent config missing fields: {missing_fields}. Using defaults.")
            return _get_default_config()

        logger.debug("Successfully loaded agent configuration from YAML")
        return config

    except Exception as e:
        logger.error(f"Failed to load agent config: {e}. Using defaults.", exc_info=True)
        return _get_default_config()


def _get_default_config() -> dict:
    """
    Provide default configuration as a fallback.

    This ensures the agent can still be created even if the YAML config
    is unavailable or corrupted. These defaults are basic but functional.

    Returns:
        Dictionary with default agent configuration
    """
    return {
        "role": "Career Narrative Specialist",
        "goal": (
            "Rewrite experience section bullets to align with target job requirements "
            "while maintaining complete truthfulness. Incorporate relevant keywords naturally, "
            "quantify achievements wherever possible, reorder responsibilities by relevance, "
            "and create compelling achievement statements that demonstrate impact."
        ),
        "backstory": (
            "You are a career storytelling expert who specializes in transforming job "
            "descriptions into achievement narratives. With a background in both corporate "
            "communications and HR, you understand what makes experience compelling to hiring "
            "managers. You excel at reframing responsibilities to emphasize outcomes and impact "
            "over tasks, finding and incorporating relevant keywords naturally, quantifying "
            "achievements, using action verbs that convey leadership and results, and structuring "
            "bullets in order of relevance to the target role. You are both strategic and truthful, "
            "never fabricating achievements or exaggerating scope."
        ),
        "llm": "gemini/gemini-2.5-flash-lite",
        "temperature": 0.5,
        "verbose": True,
    }


# ==============================================================================
# Agent Creation
# ==============================================================================


def create_experience_optimizer_agent() -> Agent:
    """
    Create and configure the Experience Section Optimizer agent.

    This is the main entry point for creating this agent. It handles all the
    complexity of configuration loading and agent initialization.

    Returns:
        Configured CrewAI Agent instance ready to optimize experience entries

    Raises:
        Exception: If agent creation fails (logged and re-raised)

    Example:
        >>> agent = create_experience_optimizer_agent()
        >>> # Agent is now ready to be used in a crew or task

    Design Notes:
        - Uses configuration from agents.yaml (with fallback to defaults)
        - No tools needed - operates on structured data inputs
        - Moderate temperature (0.5) for balanced creativity and consistency
        - Uses gemini-2.5-flash-lite for cost-effective rewriting with iterative improvement
        - Enables verbose mode for detailed logging
    """
    try:
        logger.info("Creating Experience Section Optimizer agent...")

        # Load configuration
        config = _load_agent_config()

        # Extract LLM settings
        llm_model = config.get("llm", "gemini/gemini-2.5-flash-lite")
        temperature = config.get("temperature", 0.5)
        verbose = config.get("verbose", True)

        # AGENTIC ITERATIVE IMPROVEMENT MECHANISM - COMPLETE WORKFLOW
        #
        # THIS IS THE CORE AGENTIC BEHAVIOR: The agent autonomously improves
        # its own output through iterative self-evaluation and regeneration.
        #
        # WORKFLOW SEQUENCE (How the Agent Operates):
        #
        # STAGE A: INITIAL GENERATION (Agent's First Action)
        # [A.1] Agent receives task: "Optimize experience bullets for job X"
        # [A.2] Agent reads original resume experience entries
        # [A.3] Agent reads alignment strategy (keywords, guidance, etc.)
        # [A.4] Agent generates INITIAL bullets (3-6 per entry) via LLM reasoning
        #       -> Uses its backstory/goal to create compelling bullets
        #       -> Incorporates keywords naturally
        #       -> Applies CAR format, strong verbs, quantification
        #
        # STAGE B: SELF-EVALUATION (Agent Calls Tool)
        # [B.1] Agent invokes evaluate_experience_bullets tool:
        #       evaluate_experience_bullets(
        #         bullets_json=json.dumps(generated_bullets),
        #         keywords="Python,AWS,Docker",
        #         strategy_json=json.dumps(strategy)
        #       )
        # [B.2] Tool executes (see STAGE 2A documentation below):
        #       -> Parses bullets, keywords, strategy
        #       -> Evaluates each bullet (STAGE 3A: evaluate_single_bullet)
        #       -> Aggregates scores (average quality)
        #       -> Generates critique (STAGE 3B: generate_bullet_critique)
        #       -> Returns JSON: {score, issues, critique, meets_threshold}
        # [B.3] Agent receives evaluation result
        # [B.4] Agent parses JSON to extract:
        #       -> average_score (e.g., 65/100)
        #       -> per_bullet_scores (e.g., [60, 70, 65])
        #       -> issues (e.g., ["Missing metrics", "Passive voice"])
        #       -> critique (e.g., "Add measurable outcome, specify method...")
        #       -> meets_threshold (e.g., False if score < 85)
        #
        # STAGE C: DECISION POINT (Agent's Autonomous Decision)
        # [C.1] Agent checks: Does meets_threshold == True?
        # [C.2] IF True (score >= 85):
        #       -> Agent stops iterating for this entry
        #       -> Agent records final bullets
        #       -> Agent moves to next experience entry
        # [C.3] IF False (score < 85) AND iteration_count < 3:
        #       -> Agent proceeds to STAGE D (Regeneration)
        # [C.4] IF False AND iteration_count >= 3:
        #       -> Agent stops (max iterations reached)
        #       -> Agent records final bullets (best effort)
        #
        # STAGE D: REGENERATION (Agent Improves Based on Critique)
        # [D.1] Agent reads critique from evaluation result
        # [D.2] Agent analyzes specific issues:
        #       -> "Bullet 1: Missing structure (metric, method)"
        #       -> "Bullet 2: Too generic (no technology)"
        #       -> "Bullet 3: Activity-focused (missing business impact)"
        # [D.3] Agent regenerates bullets addressing each issue:
        #       -> For Bullet 1: Adds metrics and method
        #       -> For Bullet 2: Adds specific technologies
        #       -> For Bullet 3: Adds business impact/outcomes
        # [D.4] Agent creates IMPROVED bullets (still 3-6 per entry)
        # [D.5] Agent increments iteration_count
        # [D.6] Agent loops back to STAGE B (Self-Evaluation)
        #
        # STAGE E: ITERATION TRACKING (Agent Maintains History)
        # [E.1] After each evaluation, agent records:
        #       iteration_history.append({
        #         "iteration": iteration_count,
        #         "content": "\n".join(current_bullets),
        #         "quality_score": average_score,
        #         "issues_found": issues,
        #         "critique": critique
        #       })
        # [E.2] Agent tracks progress across iterations
        # [E.3] Agent includes iteration_history in final output
        #
        # KEY DESIGN DECISIONS:
        # 1. AGENT CONTROLS THE LOOP: The LLM agent decides when to call
        #    the tool, when to regenerate, and when to stop. This is TRUE
        #    agentic behavior - not a hardcoded loop.
        #
        # 2. TOOL PROVIDES OBJECTIVE FEEDBACK: The evaluate_experience_bullets
        #    tool is deterministic (no LLM calls), providing consistent,
        #    objective quality scoring and actionable critique.
        #
        # 3. SEPARATION OF CONCERNS:
        #    - Agent: Creative generation, decision-making, iteration control
        #    - Tool: Objective evaluation, quality scoring, critique generation
        #
        # 4. MAX_ITERATIONS = 3: Balances quality improvement with cost/time.
        #    Most bullets improve significantly in 1-2 iterations.
        #
        # 5. QUALITY_THRESHOLD = 85: Professional-grade quality bar.
        #    Scores 85+ indicate bullets with metrics, impact, specificity.
        #
        # VERIFICATION: How to Confirm It's Working
        # To verify the agent is truly iterating:
        # 1. Check task execution logs for multiple tool invocations
        # 2. Check output JSON for iteration_history with multiple entries
        # 3. Verify quality scores improve across iterations
        # 4. Verify bullets change between iterations (not just same bullets)
        # 5. Run test_experience_optimizer_enhancements.py with real LLM
        # ===================================================================

        agent = Agent(
            role=config["role"],
            goal=config["goal"],
            backstory=config["backstory"],
            # AGENTIC TOOL: Self-evaluation capability for iterative improvement
            # This tool enables the agent to:
            # - Evaluate its own generated bullets objectively
            # - Receive actionable critique for improvement
            # - Make autonomous decisions about when to regenerate
            # - Track quality improvements across iterations
            tools=[evaluate_experience_bullets],
            llm=llm_model,
            temperature=temperature,
            verbose=verbose,
            allow_delegation=False,  # This agent works independently
            max_iter=15,  # Allows multiple tool invocations per experience entry
            # (3 iterations × ~5 tool calls per iteration = 15 max)
        )

        logger.info(
            f"Successfully created agent: {config['role']}, "
            f"using LLM: {llm_model}, temperature: {temperature}"
        )

        return agent

    except Exception as e:
        logger.error(f"Failed to create Experience Section Optimizer agent: {e}", exc_info=True)
        raise


# ==============================================================================
# Output Validation
# ==============================================================================


def validate_experience_output(
    output_data: dict,
) -> IterativeExperienceOptimization | OptimizedExperienceSection | None:
    """
    Validate that the agent's output conforms to experience section models.

    This function accepts both IterativeExperienceOptimization (preferred) and
    OptimizedExperienceSection (fallback) models for backward compatibility.

    Args:
        output_data: Dictionary containing the optimized experience section

    Returns:
        IterativeExperienceOptimization or OptimizedExperienceSection if valid, None if it fails

    Design Notes:
        - Tries iterative model first (with iteration tracking)
        - Falls back to non-iterative model for compatibility
        - Detailed logging helps diagnose optimization issues
        - Returning None (rather than raising) allows graceful handling upstream

    Edge Cases Handled:
        - Missing required fields -> logged with specific field names
        - Invalid Experience objects -> caught by Pydantic validation
        - Malformed data types -> validation error details provided
    """
    try:
        # Try iterative model first (preferred for new workflows)
        if "iteration_history" in output_data or "iterations_used" in output_data:
            logger.debug("Validating agent output against IterativeExperienceOptimization model...")
            section = IterativeExperienceOptimization(**output_data)

            logger.info(
                f"Iterative validation successful. "
                f"Entries: {len(section.optimized_experiences)}, "
                f"Iterations used: {section.iterations_used}, "
                f"Final score: {section.final_quality_score}/100, "
                f"Keywords: {len(section.keywords_integrated)}"
            )

            return section

        # Fallback to non-iterative model
        logger.debug("Validating agent output against OptimizedExperienceSection model...")
        section = OptimizedExperienceSection(**output_data)

        logger.info(
            f"Non-iterative validation successful. "
            f"Entries optimized: {len(section.optimized_experiences)}, "
            f"Keywords integrated: {len(section.keywords_integrated)}"
        )

        return section

    except ValidationError as e:
        logger.error(
            f"Experience validation failed. Output does not match expected model schema. "
            f"Errors: {e.errors()}"
        )
        # Log each validation error for easier debugging
        for error in e.errors():
            logger.error(f"  Field: {error['loc']}, Type: {error['type']}, Message: {error['msg']}")
        return None

    except Exception as e:
        logger.error(f"Unexpected error during experience validation: {e}", exc_info=True)
        return None


# ==============================================================================
# Quality Check Helper Functions
# ==============================================================================

# Strong action verbs for quality checks
STRONG_ACTION_VERBS = {
    # Leadership & Management
    "led",
    "directed",
    "managed",
    "supervised",
    "coordinated",
    "orchestrated",
    "spearheaded",
    "championed",
    "mentored",
    "coached",
    "guided",
    # Achievement & Results
    "achieved",
    "delivered",
    "accomplished",
    "exceeded",
    "surpassed",
    "attained",
    "realized",
    "completed",
    "finalized",
    # Innovation & Creation
    "created",
    "developed",
    "designed",
    "built",
    "engineered",
    "architected",
    "established",
    "founded",
    "launched",
    "pioneered",
    "innovated",
    "invented",
    # Improvement & Optimization
    "improved",
    "enhanced",
    "optimized",
    "streamlined",
    "refined",
    "upgraded",
    "modernized",
    "transformed",
    "revitalized",
    "restructured",
    # Growth & Expansion
    "increased",
    "grew",
    "expanded",
    "scaled",
    "accelerated",
    "boosted",
    "maximized",
    "amplified",
    "multiplied",
    # Reduction & Efficiency
    "reduced",
    "decreased",
    "minimized",
    "eliminated",
    "consolidated",
    "simplified",
    "automated",
    # Strategy & Planning
    "strategized",
    "planned",
    "formulated",
    "devised",
    "conceptualized",
    "envisioned",
    "forecasted",
    # Analysis & Research
    "analyzed",
    "evaluated",
    "assessed",
    "investigated",
    "researched",
    "identified",
    "diagnosed",
    "audited",
    # Communication & Collaboration
    "presented",
    "communicated",
    "collaborated",
    "negotiated",
    "facilitated",
    "liaised",
    "partnered",
    "consulted",
}


def analyze_action_verbs(achievements: list[str]) -> dict:
    """
    Analyze the strength of action verbs used in achievement bullets.

    This function checks if each achievement starts with a strong action verb
    and calculates an overall action verb strength score.

    Args:
        achievements: List of achievement bullet points

    Returns:
        Dictionary with analysis results including:
        - strong_verb_count: Number of bullets with strong action verbs
        - weak_verb_count: Number of bullets with weak/missing action verbs
        - strength_score: Overall score (0-100)
        - weak_bullets: List of bullets that need improvement

    Design Note:
        Strong action verbs convey impact and leadership, which is critical
        for engaging hiring managers and passing ATS systems.
    """
    if not achievements:
        return {
            "strong_verb_count": 0,
            "weak_verb_count": 0,
            "strength_score": 0,
            "weak_bullets": [],
        }

    strong_count = 0
    weak_bullets = []

    for achievement in achievements:
        # Get the first word (action verb)
        first_word = achievement.strip().split()[0].lower().rstrip(".,;:")

        if first_word in STRONG_ACTION_VERBS:
            strong_count += 1
        else:
            weak_bullets.append(achievement[:50] + "..." if len(achievement) > 50 else achievement)

    weak_count = len(achievements) - strong_count
    strength_score = int((strong_count / len(achievements)) * 100) if achievements else 0

    return {
        "strong_verb_count": strong_count,
        "weak_verb_count": weak_count,
        "strength_score": strength_score,
        "weak_bullets": weak_bullets,
    }


def count_quantified_achievements(achievements: list[str]) -> dict:
    """
    Count how many achievements include quantifiable metrics.

    This function identifies achievements that contain numbers, percentages,
    or other quantifiable metrics, which are critical for demonstrating impact.

    Args:
        achievements: List of achievement bullet points

    Returns:
        Dictionary with analysis results including:
        - quantified_count: Number of achievements with metrics
        - unquantified_count: Number without metrics
        - quantification_rate: Percentage (0-1)
        - unquantified_bullets: List of bullets that need metrics

    Design Note:
        Quantified achievements are significantly more compelling and memorable
        than vague statements. They provide concrete evidence of impact.
    """
    import re

    if not achievements:
        return {
            "quantified_count": 0,
            "unquantified_count": 0,
            "quantification_rate": 0.0,
            "unquantified_bullets": [],
        }

    # Patterns that indicate quantification
    metric_patterns = [
        r"\d+%",  # Percentages: 40%, 100%
        r"\$\d+",  # Dollar amounts: $1M, $500K
        r"\d+[KkMmBb]",  # Abbreviated numbers: 5M, 10K, 1B
        r"\d+\+",  # Numbers with plus: 50+, 100+
        r"\d+x",  # Multipliers: 2x, 10x
        r"\d+\s*(?:users|customers|clients|projects|members|engineers)",  # Count with unit
        r"(?:reduced|increased|improved|grew|decreased)\s+(?:by\s+)?\d+",  # Action + number
    ]

    quantified_count = 0
    unquantified_bullets = []

    for achievement in achievements:
        # Check if any metric pattern matches
        has_metric = any(
            re.search(pattern, achievement, re.IGNORECASE) for pattern in metric_patterns
        )

        if has_metric:
            quantified_count += 1
        else:
            unquantified_bullets.append(
                achievement[:50] + "..." if len(achievement) > 50 else achievement
            )

    unquantified_count = len(achievements) - quantified_count
    quantification_rate = quantified_count / len(achievements) if achievements else 0.0

    return {
        "quantified_count": quantified_count,
        "unquantified_count": unquantified_count,
        "quantification_rate": round(quantification_rate, 2),
        "unquantified_bullets": unquantified_bullets,
    }


# ==============================================================================
# Advanced Professional Writing Validation (New)
# ==============================================================================

# Weak nouns that dilute impact
WEAK_NOUNS = {
    "work",
    "tasks",
    "duties",
    "responsibilities",
    "activities",
    "things",
    "stuff",
    "assignments",
    "jobs",
    "roles",
    "functions",
}

# Power nouns that pair well with strong verbs
POWER_NOUNS = {
    "initiative",
    "transformation",
    "platform",
    "infrastructure",
    "framework",
    "architecture",
    "strategy",
    "program",
    "system",
    "solution",
    "methodology",
    "roadmap",
    "pipeline",
    "ecosystem",
    "paradigm",
    "capability",
}

# Generic phrases that lack specificity
GENERIC_PHRASES = {
    "worked with",
    "responsible for",
    "helped with",
    "assisted in",
    "participated in",
    "involved in",
    "contributed to",
    "supported",
    "cross-functional teams",
    "various projects",
    "multiple stakeholders",
    "day-to-day operations",
    "daily tasks",
}

# Scope indicators patterns
SCOPE_PATTERNS = [
    r"team of \d+",
    r"\d+[KMB]\+?\s+(?:users|customers|clients|members|engineers|developers)",
    r"\d+[KMBT]B\s+(?:data|records|requests|traffic)",
    r"\$\d+[KMB]",
    r"\d+\s+(?:countries|regions|offices|locations|markets)",
    r"global",
    "nationwide",
    "enterprise-wide",
]


def validate_bullet_structure(achievement: str) -> dict:
    """
    Validate that a bullet follows impact-first structure (XYZ/STAR).

    Checks for:
    - Measurable outcome (number, percentage, scale)
    - Method/technology (how it was done)
    - Strong action verb at start
    """
    import re

    # Check for metrics
    has_metric = bool(
        re.search(r"\d+[%KMB$+x]|\d+\s*(?:users|customers|engineers|clients)", achievement)
    )

    # Check for method indicators
    has_method = bool(
        re.search(
            r"(?:by|through|using|via|with|utilizing|leveraging)\s+\w+", achievement, re.IGNORECASE
        )
    )

    # Check for strong verb
    words = achievement.strip().split()
    has_strong_verb = False
    if words:
        first_word = words[0].lower().rstrip(".,;:")
        has_strong_verb = first_word in STRONG_ACTION_VERBS

    return {
        "is_valid": has_metric and has_method and has_strong_verb,
        "has_metric": has_metric,
        "has_method": has_method,
        "has_strong_verb": has_strong_verb,
        "improvement_needed": [],
    }


def assess_impact_level(achievement: str) -> dict:
    """
    Assess the business impact level of an achievement ("So What?" Test).

    Returns impact tier and whether it's compelling enough.
    """
    import re

    impact_patterns = {
        "revenue_cost": r"(\$\d+[KMB]?|\d+%\s+(?:cost|revenue|profit|savings|budget))",
        "efficiency": r"(?:reduced|decreased|improved|accelerated|optimized|streamlined).*\d+%",
        "scale": r"\d+[KMB]\+?\s+(?:users|customers|requests|transactions|records)",
        "quality": r"(?:decreased|reduced).*(?:bugs|errors|incidents|downtime).*\d+%",
        "capability": r"enabled|unlocked|pioneered|introduced|launched",
    }

    tier = "activity"  # Default: just describes task
    for tier_name, pattern in impact_patterns.items():
        if re.search(pattern, achievement, re.IGNORECASE):
            tier = tier_name
            break

    return {
        "tier": tier,
        "is_impact_focused": tier != "activity",
        "recommendation": "Add measurable business outcome" if tier == "activity" else "",
    }


def analyze_verb_noun_pairs(achievement: str) -> dict:
    """
    Check if strong verbs are paired with impactful nouns.
    """
    words = achievement.strip().split()
    if len(words) < 2:
        return {"has_power_pair": False}

    verb = words[0].lower().rstrip(".,;:")
    noun = words[1].lower().rstrip(".,;:")

    is_strong_verb = verb in STRONG_ACTION_VERBS
    is_weak_noun = noun in WEAK_NOUNS
    is_power_noun = noun in POWER_NOUNS

    return {
        "has_power_pair": is_strong_verb and is_power_noun,
        "has_weak_noun": is_weak_noun,
        "recommendation": f"Replace '{noun}' with more specific noun" if is_weak_noun else "",
    }


def detect_passive_voice(achievements: list[str]) -> dict:
    """
    Detect and flag passive voice constructions.
    """
    import re

    passive_patterns = [
        r"\b(?:was|were|been|being)\s+\w+ed\b",  # was managed, were implemented
        r"\b(?:was|were)\s+responsible\s+for\b",
        r"\b(?:was|were)\s+tasked\s+with\b",
    ]

    passive_bullets = []
    for achievement in achievements:
        for pattern in passive_patterns:
            if re.search(pattern, achievement, re.IGNORECASE):
                passive_bullets.append(achievement)
                break

    return {
        "passive_count": len(passive_bullets),
        "passive_bullets": passive_bullets,
        "is_acceptable": len(passive_bullets) == 0,
    }


def check_specificity(achievement: str) -> dict:
    """
    Check if bullet is specific enough to be valuable.
    """
    import re

    issues = []

    # Check for generic phrases
    generic_found = [phrase for phrase in GENERIC_PHRASES if phrase in achievement.lower()]
    if generic_found:
        issues.append(f"Generic phrases: {generic_found}")

    # Check for specific technologies (simple heuristic: Capitalized words or known tech)
    has_technology = bool(
        re.search(
            r"\b[A-Z][a-z]+(?:JS|SQL|DB|API)\b|Python|Java|AWS|Azure|GCP|React|Node|Docker|Kubernetes",
            achievement,
        )
    )
    if not has_technology:
        issues.append("No specific technology mentioned")

    # Check for numbers
    has_numbers = bool(re.search(r"\d", achievement))
    if not has_numbers:
        issues.append("No quantifiable metrics")

    is_specific = len(issues) == 0

    return {
        "is_specific": is_specific,
        "specificity_score": 100 - (len(issues) * 33),
        "issues": issues,
    }


def detect_multi_accomplishment_bullets(achievements: list[str]) -> dict:
    """
    Flag bullets that try to cram multiple accomplishments.
    """
    multi_bullets = []

    for achievement in achievements:
        # Count "and" occurrences
        and_count = achievement.lower().count(" and ")

        # Count strong verbs (if more than 1, might be multi-accomplishment)
        verbs_found = sum(
            1 for verb in STRONG_ACTION_VERBS if f" {verb} " in f" {achievement.lower()} "
        )

        if and_count >= 2 or verbs_found >= 2:
            multi_bullets.append(
                {
                    "bullet": achievement,
                    "and_count": and_count,
                    "verb_count": verbs_found,
                    "recommendation": "Split into separate bullets for clarity",
                }
            )

    return {
        "multi_bullet_count": len(multi_bullets),
        "flagged_bullets": multi_bullets,
        "is_acceptable": len(multi_bullets) <= 1,  # Allow one multi-bullet max
    }


def has_scope_indicator(achievement: str) -> bool:
    """Check if bullet includes scale/scope context."""
    import re

    return any(re.search(pattern, achievement, re.IGNORECASE) for pattern in SCOPE_PATTERNS)


def reorder_bullets_by_relevance(
    achievements: list[str], strategy: AlignmentStrategy, required_keywords: list[str]
) -> list[str]:
    """
    Reorder bullets within a single job entry by relevance to target role.
    """
    import re

    scored_bullets = []

    for achievement in achievements:
        score = 0

        # Keyword matching
        keywords_in_bullet = sum(1 for kw in required_keywords if kw.lower() in achievement.lower())
        score += min(keywords_in_bullet * 10, 40)

        # Impact magnitude (has big numbers?)
        if re.search(r"(\$\d+[MB]|\d+%|\d+[MB]\+?\s+users)", achievement):
            score += 30
        elif re.search(r"\d+", achievement):
            score += 15

        # Leadership indicators
        leadership_words = ["led", "managed", "directed", "spearheaded", "team of"]
        if any(word in achievement.lower() for word in leadership_words):
            score += 20

        # Technical depth (mentions specific technologies)
        # Using strategy.identified_matches if available
        if strategy.identified_matches:
            tech_mentions = sum(
                1
                for skill in strategy.identified_matches
                if skill.resume_skill.lower() in achievement.lower()
            )
            score += min(tech_mentions * 5, 10)

        scored_bullets.append((score, achievement))

    # Sort by score descending
    scored_bullets.sort(reverse=True, key=lambda x: x[0])

    return [bullet for score, bullet in scored_bullets]


# ==============================================================================
# STAGE 3: Evaluation Framework (Deterministic Quality Scoring)
# ==============================================================================
# PURPOSE: Provide objective, consistent bullet quality assessment
# WHAT: Pure functions that score bullets without LLM calls
# WHY: Deterministic evaluation ensures consistent feedback across iterations
#
# DESIGN PHILOSOPHY:
# - NO LLM CALLS: Fast, deterministic, cost-effective
# - MULTI-DIMENSIONAL: Checks 6+ quality dimensions
# - ACTIONABLE: Returns specific issues, not just scores
# - COMPOSABLE: Each check is independent function
#
# QUALITY DIMENSIONS CHECKED:
# 1. Structure: XYZ/STAR formula (Metric + Method + Verb)
# 2. Voice: Active vs Passive detection
# 3. Specificity: Generic phrases vs Concrete details
# 4. Impact: Business outcomes vs Activity descriptions
# 5. Scope: Scale indicators (team size, user count, etc.)
# 6. Power Pairs: Strong verb + Impactful noun combinations
# ==============================================================================


def evaluate_single_bullet(
    bullet: str, strategy: AlignmentStrategy, required_keywords: list[str]
) -> dict:
    """
    STAGE 3A: Single Bullet Evaluation - Comprehensive Quality Assessment

    Evaluate a single bullet point against all quality criteria.

    This function consolidates all validation checks into a single
    comprehensive evaluation with scoring and issue tracking.

    SCORING METHODOLOGY:
    - Start at 100 (perfect score)
    - Deduct points for each failed check:
      • Missing structure: -15 points
      • Passive voice: -20 points
      • Too generic: -15 points
      • Activity-focused (no impact): -20 points
      • No scope indicator: -10 points
      • Weak noun pairing: -5 points
    - Minimum score: 0

    EXECUTION SEQUENCE:
    [3A.1] Initialize scoring (score=100, issues=[])
    [3A.2] Check structure (XYZ/STAR formula)
    [3A.3] Check passive voice
    [3A.4] Check specificity (generic phrases, tech mentions)
    [3A.5] Check impact level (activity vs outcome)
    [3A.6] Check scope indicators (team size, scale)
    [3A.7] Check verb-noun power pairing
    [3A.8] Return aggregated results

    Args:
        bullet: The bullet point text to evaluate
        strategy: The alignment strategy for context
        required_keywords: Keywords that should be present

    Returns:
        Dictionary with:
            - score: int (0-100) overall quality score
            - issues: list[str] specific problems found
            - passed_checks: dict[str, bool] individual check results

    Design Note:
        This is a deterministic function (no LLM calls). It uses our
        existing validation functions to provide objective quality metrics.
    """
    # ---------------------------------------------------------------------
    # [SUB-STAGE 3A.1] Initialize Scoring
    # ---------------------------------------------------------------------
    # WHAT: Start with perfect score, deduct for each issue
    # WHY: Negative scoring is intuitive ("what's wrong") vs additive
    score = 100
    issues = []

    # ---------------------------------------------------------------------
    # [SUB-STAGE 3A.2] Check Structure (XYZ/STAR Formula)
    # ---------------------------------------------------------------------
    # WHAT: Validate Metric + Method + Strong Verb presence
    # WHY: Foundation of compelling achievement statements
    # PENALTY: -15 points (critical structural element)
    structure = validate_bullet_structure(bullet)
    if not structure["is_valid"]:
        score -= 15
        missing = []
        if not structure["has_metric"]:
            missing.append("metric")
        if not structure["has_method"]:
            missing.append("method")
        if not structure["has_strong_verb"]:
            missing.append("strong verb")
        issues.append(f"Missing structure: {', '.join(missing)}")

    # ---------------------------------------------------------------------
    # [SUB-STAGE 3A.3] Check Passive Voice
    # ---------------------------------------------------------------------
    # WHAT: Detect "was/were" constructions indicating passive voice
    # WHY: Active voice is more compelling and shows ownership
    # PENALTY: -20 points (major readability/impact issue)
    passive = detect_passive_voice([bullet])
    if not passive["is_acceptable"]:
        score -= 20
        issues.append("Uses passive voice")

    # ---------------------------------------------------------------------
    # [SUB-STAGE 3A.4] Check Specificity
    # ---------------------------------------------------------------------
    # WHAT: Identify generic phrases and missing concrete details
    # WHY: Specific bullets are more credible and memorable
    # PENALTY: -15 points (moderate issue, affects differentiation)
    spec = check_specificity(bullet)
    if not spec["is_specific"]:
        score -= 15
        # Only show first 2 specificity issues to keep critique concise
        spec_issues = spec.get("issues", [])[:2]
        issues.append(f"Too generic: {', '.join(spec_issues)}")

    # ---------------------------------------------------------------------
    # [SUB-STAGE 3A.5] Check Impact Level
    # ---------------------------------------------------------------------
    # WHAT: Assess if bullet shows business outcomes vs just activities
    # WHY: Impact-focused bullets answer "So what?" question
    # PENALTY: -20 for activity-only, -10 for output-only
    impact = assess_impact_level(bullet)
    if impact["tier"] == "activity":
        score -= 20
        issues.append("Activity-focused (missing business impact)")
    elif impact["tier"] == "output":
        score -= 10
        issues.append("Output-focused (could emphasize business impact more)")

    # ---------------------------------------------------------------------
    # [SUB-STAGE 3A.6] Check Scope Indicators
    # ---------------------------------------------------------------------
    # WHAT: Look for scale/scope context (team size, user count, etc.)
    # WHY: Scope provides context for magnitude of achievement
    # PENALTY: -10 points (nice-to-have, not critical)
    if not has_scope_indicator(bullet):
        score -= 10
        issues.append("No scope/scale indicator (e.g., team size, user count)")

    # ---------------------------------------------------------------------
    # [SUB-STAGE 3A.7] Check Power Word Pairing
    # ---------------------------------------------------------------------
    # WHAT: Ensure strong verbs paired with impactful nouns
    # WHY: Weak nouns dilute strong verbs ("Led tasks" vs "Led initiative")
    # PENALTY: -5 points (minor polish issue)
    pair = analyze_verb_noun_pairs(bullet)
    if pair.get("has_weak_noun"):
        score -= 5
        issues.append(pair.get("recommendation", "Weak noun pairing"))

    # ---------------------------------------------------------------------
    # [SUB-STAGE 3A.8] Return Aggregated Results
    # ---------------------------------------------------------------------
    # WHAT: Package score, issues, and check results
    # WHY: Structured output allows both human/agent interpretation
    return {
        "score": max(0, score),  # Floor at 0, no negative scores
        "issues": issues,  # List of specific problems
        "passed_checks": {  # Boolean status per dimension
            "structure": structure["is_valid"],
            "passive_voice": passive["is_acceptable"],
            "specificity": spec["is_specific"],
            "impact": impact["is_impact_focused"],
            "scope": has_scope_indicator(bullet),
        },
    }


def generate_bullet_critique(bullet: str, evaluation: dict, strategy: AlignmentStrategy) -> str:
    """
    STAGE 3B: Critique Generation - Actionable Improvement Instructions

    Generate actionable critique for improving a bullet point.

    This is a deterministic function (no LLM) that translates
    evaluation results into specific improvement instructions.

    CRITIQUE PHILOSOPHY:
    - SPECIFIC: Not "improve this" but "add metrics to show impact"
    - ACTIONABLE: Each suggestion is implementable by agent
    - PRIORITIZED: Most critical issues mentioned first
    - CONCISE: Agent context window is limited

    EXECUTION FLOW:
    [3B.1] Check if bullet already meets threshold (early return)
    [3B.2] Build critique for each failed check dimension
    [3B.3] Concatenate suggestions with proper spacing
    [3B.4] Return complete actionable critique string

    Args:
        bullet: The bullet text being critiqued
        evaluation: Result from evaluate_single_bullet
        strategy: Alignment strategy for context

    Returns:
        String with specific, actionable improvement suggestions

    Design Note:
        By making critique generation deterministic, we avoid extra
        LLM calls and ensure consistent, objective feedback.
    """
    # ---------------------------------------------------------------------
    # [SUB-STAGE 3B.1] Early Return for High-Quality Bullets
    # ---------------------------------------------------------------------
    # WHAT: Skip critique if already meets threshold
    # WHY: Save processing and provide positive reinforcement
    if evaluation["score"] >= QUALITY_THRESHOLD:
        return "Bullet meets quality standards."

    # ---------------------------------------------------------------------
    # [SUB-STAGE 3B.2] Build Critique Per Failed Check
    # ---------------------------------------------------------------------
    # WHAT: Generate specific instructions for each failed dimension
    # WHY: Agent needs concrete guidance on what to fix
    critique_parts = []

    # Structure issues (most critical - addressed first)
    if not evaluation["passed_checks"]["structure"]:
        critique_parts.append(
            "Add measurable outcome (number/percentage), "
            "specify method/technology used, and ensure strong action verb."
        )

    # Passive voice
    if not evaluation["passed_checks"]["passive_voice"]:
        critique_parts.append(
            "Convert to active voice. Remove 'was/were responsible for' constructions."
        )

    # Specificity
    if not evaluation["passed_checks"]["specificity"]:
        critique_parts.append("Add specific technologies/tools used and quantifiable metrics.")

    # Impact
    if not evaluation["passed_checks"]["impact"]:
        critique_parts.append(
            "Add business impact: revenue/cost savings, efficiency gains, "
            "scale improvements, or quality metrics."
        )

    # Scope
    if not evaluation["passed_checks"]["scope"]:
        critique_parts.append(
            "Include scope indicators: team size, user count, data volume, "
            "budget, or geographic reach."
        )

    # ---------------------------------------------------------------------
    # [SUB-STAGE 3B.3 & 3B.4] Concatenate and Return
    # ---------------------------------------------------------------------
    # WHAT: Join all critique parts with spaces
    # WHY: Single coherent instruction string for agent
    return " ".join(critique_parts)


# ==============================================================================
# STAGE 4: Iterative Optimization Loop (Framework for Agent-Driven Improvement)
# ==============================================================================
# PURPOSE: Provide evaluation/critique loop for agent to self-improve
# WHAT: Framework that tracks iterations, scores, and critique history
# WHY: Enables true agentic behavior - agent improves its own output
#
# CRITICAL DESIGN DECISION:
# This function does NOT regenerate bullets (that's the LLM agent's job).
# It ONLY provides:
# 1. Evaluation of current bullets
# 2. Quality scoring
# 3. Actionable critique
# 4. Iteration history tracking
# 5. Termination condition logic
#
# AGENTIC WORKFLOW INTEGRATION:
# - Agent generates initial bullets (LLM reasoning)
# - Agent calls this function OR uses evaluate_experience_bullets tool
# - Function returns score + critique
# - Agent regenerates bullets addressing critique
# - Loop continues until threshold met
#
# WHY SEPARATE EVALUATION FROM REGENERATION:
# [OK] Deterministic evaluation (consistent, fast, cheap)
# [OK] Agent controls regeneration strategy (creative freedom)
# [OK] Clear separation of concerns (easier to debug)
# [OK] Evaluation can be unit tested without LLM
# ==============================================================================


def optimize_bullets_iteratively(
    experience: Experience,
    strategy: AlignmentStrategy,
    required_keywords: list[str],
    max_iterations: int = MAX_IMPROVEMENT_ITERATIONS,
    quality_threshold: int = QUALITY_THRESHOLD,
) -> dict:
    """
    STAGE 4: Iterative Optimization Loop - Self-Improvement Framework

    Iteratively improve experience bullets through self-critique.

    This is the CORE iterative improvement function. It provides the
    evaluation loop and critique generation framework. The actual
    bullet regeneration happens via the LLM agent.

    EXECUTION SEQUENCE:
    [4.1] Initialize: Set current_bullets from experience
    [4.2] Iteration Loop (max MAX_IMPROVEMENT_ITERATIONS):
          [4.2.1] Evaluate: Score all current bullets
          [4.2.2] Aggregate: Calculate average score
          [4.2.3] Collect Issues: Gather all problems found
          [4.2.4] Generate Critique: Create improvement instructions
          [4.2.5] Record Iteration: Save to history as BulletDraft
          [4.2.6] Check Termination: If score >= threshold, stop
          [4.2.7] Log Progress: Track iteration state
    [4.3] Return Results: Final bullets + iteration history + metrics

    IMPORTANT: Bullets DON'T change in this function!
    This function only evaluates and critiques. The LLM agent
    must regenerate bullets based on the critique returned.

    Args:
        experience: The experience entry to optimize
        strategy: Alignment strategy for guidance
        required_keywords: Keywords to integrate
        max_iterations: Maximum improvement iterations (default: 3)
        quality_threshold: Score threshold to stop iterating (default: 85)

    Returns:
        Dictionary with:
            - improved_bullets: list[str] final bullet texts
            - iteration_history: list[BulletDraft] history of iterations
            - final_score: int final average quality score
            - iterations_used: int number of iterations performed

    Design Note:
        This function does NOT call the LLM. It provides the evaluation
        framework that the agent uses to guide its iterative improvement.
        The agent is responsible for regenerating bullets based on critique.
    """
    # ---------------------------------------------------------------------
    # [SUB-STAGE 4.1] Initialize Iteration State
    # ---------------------------------------------------------------------
    # WHAT: Set starting bullets and empty history
    # WHY: Establish baseline for iteration loop
    current_bullets = experience.achievements
    iteration_history = []

    logger.info(
        f"[STAGE 4] Starting iterative optimization for {experience.company_name} "
        f"(max_iterations={max_iterations}, threshold={quality_threshold})"
    )

    # ---------------------------------------------------------------------
    # [SUB-STAGE 4.2] Iteration Loop
    # ---------------------------------------------------------------------
    # WHAT: Repeatedly evaluate, critique, record until threshold or max
    # WHY: This is the heart of the agentic improvement mechanism
    for iteration in range(max_iterations):
        # -----------------------------------------------------------------
        # [SUB-STAGE 4.2.1] Evaluate All Bullets
        # -----------------------------------------------------------------
        # WHAT: Score each bullet using comprehensive quality checks
        # WHY: Need per-bullet feedback to identify weak spots
        # CALLS: evaluate_single_bullet() for each bullet (STAGE 3A)
        bullet_evaluations = [
            evaluate_single_bullet(b, strategy, required_keywords) for b in current_bullets
        ]

        # -----------------------------------------------------------------
        # [SUB-STAGE 4.2.2] Aggregate Scores
        # -----------------------------------------------------------------
        # WHAT: Calculate overall quality as average of bullet scores
        # WHY: Single metric for termination condition
        avg_score = sum(e["score"] for e in bullet_evaluations) / len(bullet_evaluations)

        # -----------------------------------------------------------------
        # [SUB-STAGE 4.2.3] Collect All Issues
        # -----------------------------------------------------------------
        # WHAT: Gather issues from all bullets for tracking
        # WHY: Provides transparency into what needs fixing
        all_issues = []
        for i, eval_result in enumerate(bullet_evaluations):
            if eval_result["issues"]:
                all_issues.append(f"Bullet {i + 1}: {'; '.join(eval_result['issues'])}")

        # -----------------------------------------------------------------
        # [SUB-STAGE 4.2.4] Generate Critique
        # -----------------------------------------------------------------
        # WHAT: Create actionable improvement instructions per bullet
        # WHY: Agent needs specific guidance on what to fix
        # CALLS: generate_bullet_critique() for low-scoring bullets (STAGE 3B)
        if avg_score < quality_threshold:
            critique_parts = []
            for i, (bullet, eval_result) in enumerate(
                zip(current_bullets, bullet_evaluations, strict=True)
            ):
                if eval_result["score"] < quality_threshold:
                    bullet_critique = generate_bullet_critique(bullet, eval_result, strategy)
                    critique_parts.append(f"Bullet {i + 1}: {bullet_critique}")
            overall_critique = " | ".join(critique_parts)
        else:
            overall_critique = "All bullets meet quality standards."

        # -----------------------------------------------------------------
        # [SUB-STAGE 4.2.5] Record Iteration
        # -----------------------------------------------------------------
        # WHAT: Save snapshot of bullets + score + critique to history
        # WHY: Enables debugging and progress tracking
        # CREATES: BulletDraft model instance for this iteration
        iteration_history.append(
            BulletDraft(
                iteration=iteration + 1,
                content="\n".join(current_bullets),
                quality_score=int(avg_score),
                issues_found=all_issues,
                critique=overall_critique,
            )
        )

        # -----------------------------------------------------------------
        # [SUB-STAGE 4.2.7] Log Progress
        # -----------------------------------------------------------------
        # WHAT: Track iteration state for monitoring
        # WHY: Debugging and observability
        logger.info(
            f"[STAGE 4.2] Iteration {iteration + 1}: Score {avg_score:.1f}/100, "
            f"Issues: {len(all_issues)}"
        )

        # -----------------------------------------------------------------
        # [SUB-STAGE 4.2.6] Check Termination Condition
        # -----------------------------------------------------------------
        # WHAT: Decide whether to continue or stop iterating
        # WHY: This is the CRITICAL decision point for agentic loop
        # CONDITION 1: Quality threshold met -> STOP (success)
        # CONDITION 2: Max iterations reached -> STOP (via loop exit)
        if avg_score >= quality_threshold:
            logger.info(
                f"[STAGE 4.2.6] Quality threshold met at iteration {iteration + 1} - STOPPING"
            )
            break

        # If not last iteration, log that we'll continue
        if iteration < max_iterations - 1:
            logger.info(
                f"[STAGE 4.2.6] Below threshold - Continuing to iteration {iteration + 2}..."
            )

    # ---------------------------------------------------------------------
    # [SUB-STAGE 4.3] Return Results
    # ---------------------------------------------------------------------
    # WHAT: Package final state + iteration history + metrics
    # WHY: Provide complete picture of optimization journey
    final_score = int(avg_score)

    logger.info(
        f"[STAGE 4.3] Optimization complete: {len(iteration_history)} iterations, "
        f"final score: {final_score}/100"
    )

    # NOTE: current_bullets haven't changed in this function
    # This function provides evaluation framework only
    # Agent must regenerate bullets based on critique
    return {
        "improved_bullets": current_bullets,  # Same as input (no regeneration here)
        "iteration_history": iteration_history,  # Full improvement journey
        "final_score": final_score,  # Final quality metric
        "iterations_used": len(iteration_history),  # Count of iterations performed
    }


def check_keyword_integration(experience: Experience, required_keywords: list[str]) -> dict:
    """
    Check how well keywords are integrated into an experience entry.

    This function verifies that required keywords appear naturally in the
    description and achievements of the experience entry.

    Args:
        experience: The Experience object to check
        required_keywords: List of keywords that should be integrated

    Returns:
        Dictionary with analysis results including:
        - keywords_found: List of keywords present
        - keywords_missing: List of keywords missing
        - integration_rate: Percentage (0-1)

    Design Note:
        Natural keyword integration is critical for ATS optimization while
        maintaining readability for human reviewers.
    """
    if not required_keywords:
        return {
            "keywords_found": [],
            "keywords_missing": [],
            "integration_rate": 0.0,
        }

    # Combine description and achievements into searchable text
    searchable_text = experience.description.lower()
    for achievement in experience.achievements:
        searchable_text += " " + achievement.lower()

    keywords_found = []
    keywords_missing = []

    for keyword in required_keywords:
        if keyword.lower() in searchable_text:
            keywords_found.append(keyword)
        else:
            keywords_missing.append(keyword)

    integration_rate = len(keywords_found) / len(required_keywords) if required_keywords else 0.0

    return {
        "keywords_found": keywords_found,
        "keywords_missing": keywords_missing,
        "integration_rate": round(integration_rate, 2),
    }


def calculate_relevance_score(
    experience: Experience, strategy: AlignmentStrategy, required_keywords: list[str]
) -> float:
    """
    Calculate a relevance score for an experience entry.

    This function assesses how relevant an experience entry is to the target
    job based on keyword integration, skill matches, and strategic alignment.

    Args:
        experience: The Experience object to score
        strategy: The alignment strategy with guidance
        required_keywords: List of critical keywords

    Returns:
        Relevance score (0-100)

    Scoring Factors:
        - Keyword integration (40%)
        - Skill matches (30%)
        - Action verb strength (20%)
        - Quantification (10%)

    Design Note:
        This score helps prioritize which experiences to emphasize and
        validates that the most relevant entries are positioned prominently.
    """
    score = 0.0

    # Factor 1: Keyword integration (40 points)
    keyword_analysis = check_keyword_integration(experience, required_keywords)
    score += keyword_analysis["integration_rate"] * 40

    # Factor 2: Skill matches (30 points)
    matched_skills = set(experience.skills_used) & {
        m.resume_skill for m in strategy.identified_matches
    }
    if strategy.identified_matches:
        skill_match_rate = len(matched_skills) / min(
            len(strategy.identified_matches), len(experience.skills_used) or 1
        )
        score += min(skill_match_rate, 1.0) * 30

    # Factor 3: Action verb strength (20 points)
    verb_analysis = analyze_action_verbs(experience.achievements)
    score += (verb_analysis["strength_score"] / 100) * 20

    # Factor 4: Quantification (10 points)
    quant_analysis = count_quantified_achievements(experience.achievements)
    score += quant_analysis["quantification_rate"] * 10

    return round(score, 1)


# ==============================================================================
# Content Quality Checks
# ==============================================================================


def check_experience_quality(
    section: OptimizedExperienceSection, strategy: AlignmentStrategy
) -> dict:
    """
    Perform comprehensive quality checks on the optimized experience section.

    This function validates that the experience section is well-optimized,
    strategically aligned, and ready for both ATS and human review.

    Args:
        section: The OptimizedExperienceSection object
        strategy: The AlignmentStrategy that guided the optimization

    Returns:
        Dictionary with quality check results and recommendations

    Quality Checks:
        - Action verb strength per entry
        - Quantification rate per entry
        - Keyword integration across section
        - Relevance ordering validation
        - Bullet count compliance (3-6 per entry)
        - Overall section quality score
        - NEW: Bullet structure (XYZ/STAR)
        - NEW: Passive voice detection
        - NEW: Specificity check
        - NEW: Impact level assessment
        - NEW: Scope indicator check
    """
    issues = []
    warnings = []
    overall_score = 100

    # Required keywords (top 5 from strategy)
    required_keywords = strategy.keywords_to_integrate[:5] if strategy.keywords_to_integrate else []

    # Per-entry analysis
    entry_analyses = []

    for exp in section.optimized_experiences:
        entry_issues = []
        entry_warnings = []
        entry_score = 100

        # Check 1: Bullet count (3-6 optimal)
        bullet_count = len(exp.achievements)
        if bullet_count < 3:
            entry_issues.append(f"Too few bullets ({bullet_count}). Aim for 3-6.")
            entry_score -= 20
        elif bullet_count > 6:
            entry_warnings.append(f"Many bullets ({bullet_count}). Consider condensing to 3-6.")
            entry_score -= 10

        # Check 2: Action verb strength
        verb_analysis = analyze_action_verbs(exp.achievements)
        if verb_analysis["strength_score"] < 70:
            entry_warnings.append(
                f"Weak action verbs ({verb_analysis['strength_score']}/100). "
                f"Improve: {', '.join(verb_analysis['weak_bullets'][:2])}"
            )
            entry_score -= 15

        # Check 3: Quantification rate
        quant_analysis = count_quantified_achievements(exp.achievements)
        if quant_analysis["quantification_rate"] < 0.5:
            entry_warnings.append(
                f"Low quantification rate ({quant_analysis['quantification_rate'] * 100:.0f}%). "
                f"Add metrics to: {', '.join(quant_analysis['unquantified_bullets'][:2])}"
            )
            entry_score -= 15

        # Check 4: Keyword integration
        keyword_analysis = check_keyword_integration(exp, required_keywords)
        if keyword_analysis["integration_rate"] < 0.4 and required_keywords:
            entry_warnings.append(
                f"Low keyword integration ({keyword_analysis['integration_rate'] * 100:.0f}%). "
                f"Missing: {', '.join(keyword_analysis['keywords_missing'][:3])}"
            )
            entry_score -= 10

        # Check 5: Description quality
        if not exp.description or len(exp.description) < 20:
            entry_issues.append("Description too brief or missing")
            entry_score -= 15

        # NEW Check 6: Bullet structure (XYZ/STAR)
        structure_issues = 0
        for achievement in exp.achievements:
            structure_check = validate_bullet_structure(achievement)
            if not structure_check["is_valid"]:
                structure_issues += 1

        if structure_issues > 0:
            entry_warnings.append(
                f"{structure_issues} bullets lack strong structure (Metric + Method + Verb)"
            )
            entry_score -= 5 * structure_issues

        # NEW Check 7: Passive voice detection
        passive_check = detect_passive_voice(exp.achievements)
        if not passive_check["is_acceptable"]:
            entry_issues.append(f"Passive voice in {passive_check['passive_count']} bullets")
            entry_score -= 10 * passive_check["passive_count"]

        # NEW Check 8: Specificity check
        generic_count = 0
        for achievement in exp.achievements:
            spec_check = check_specificity(achievement)
            if not spec_check["is_specific"]:
                generic_count += 1

        if generic_count > 0:
            entry_warnings.append(f"{generic_count} bullets are too generic")
            entry_score -= 5 * generic_count

        # NEW Check 9: Impact level assessment
        low_impact_count = 0
        for achievement in exp.achievements:
            impact = assess_impact_level(achievement)
            if impact["tier"] == "activity":
                low_impact_count += 1

        if low_impact_count > len(exp.achievements) / 2:
            entry_issues.append("Too many activity-focused bullets (missing business impact)")
            entry_score -= 15

        # NEW Check 10: Scope indicator check
        bullets_with_scope = sum(1 for a in exp.achievements if has_scope_indicator(a))
        scope_rate = bullets_with_scope / len(exp.achievements) if exp.achievements else 0
        if scope_rate < 0.3:  # At least 30% should have scope
            entry_warnings.append("Few bullets indicate scope/scale")
            entry_score -= 5

        # NEW Check 11: Power word pairing
        weak_pairs = 0
        for achievement in exp.achievements:
            pair_check = analyze_verb_noun_pairs(achievement)
            if pair_check.get("has_weak_noun"):
                weak_pairs += 1

        if weak_pairs > 0:
            entry_warnings.append(f"{weak_pairs} bullets use weak nouns with verbs")
            entry_score -= 2 * weak_pairs

        # NEW Check 12: Multi-accomplishment detection
        multi_check = detect_multi_accomplishment_bullets(exp.achievements)
        if not multi_check["is_acceptable"]:
            entry_warnings.append(
                f"{multi_check['multi_bullet_count']} bullets cram multiple accomplishments"
            )
            entry_score -= 5

        entry_analyses.append(
            {
                "company": exp.company_name,
                "entry_score": max(0, entry_score),
                "bullet_count": bullet_count,
                "action_verb_strength": verb_analysis["strength_score"],
                "quantification_rate": quant_analysis["quantification_rate"],
                "keyword_integration": keyword_analysis["integration_rate"],
                "issues": entry_issues,
                "warnings": entry_warnings,
            }
        )

        # Add to overall issues/warnings
        if entry_issues:
            issues.append(f"{exp.company_name}: {'; '.join(entry_issues)}")
        if entry_warnings:
            warnings.append(f"{exp.company_name}: {'; '.join(entry_warnings)}")

    # Overall section checks
    avg_verb_strength = (
        sum(e["action_verb_strength"] for e in entry_analyses) / len(entry_analyses)
        if entry_analyses
        else 0
    )
    avg_quant_rate = (
        sum(e["quantification_rate"] for e in entry_analyses) / len(entry_analyses)
        if entry_analyses
        else 0
    )
    avg_keyword_rate = (
        sum(e["keyword_integration"] for e in entry_analyses) / len(entry_analyses)
        if entry_analyses
        else 0
    )
    avg_entry_score = (
        sum(e["entry_score"] for e in entry_analyses) / len(entry_analyses) if entry_analyses else 0
    )

    # Penalize overall score based on averages
    if avg_verb_strength < 70:
        overall_score -= 10
    if avg_quant_rate < 0.5:
        overall_score -= 10
    if avg_keyword_rate < 0.4:
        overall_score -= 15

    # Use average entry score as base for overall score
    overall_score = min(overall_score, avg_entry_score)

    # Check if experience entries are ordered by relevance
    if len(section.optimized_experiences) > 1 and section.relevance_scores:
        scores_in_order = [
            section.relevance_scores.get(exp.company_name, 0)
            for exp in section.optimized_experiences
        ]
        if scores_in_order != sorted(scores_in_order, reverse=True):
            warnings.append("Experiences may not be ordered by relevance to target job")
            overall_score -= 5

    # Determine quality level
    if overall_score >= 90:
        quality = "excellent"
    elif overall_score >= 75:
        quality = "good"
    elif overall_score >= 60:
        quality = "fair"
    else:
        quality = "poor"

    result = {
        "quality": quality,
        "overall_score": max(0, int(overall_score)),
        "issues": issues,
        "warnings": warnings,
        "is_acceptable": overall_score >= 60,
        "entry_analyses": entry_analyses,
        "section_metrics": {
            "avg_action_verb_strength": round(avg_verb_strength, 1),
            "avg_quantification_rate": round(avg_quant_rate, 2),
            "avg_keyword_integration": round(avg_keyword_rate, 2),
            "total_entries": len(section.optimized_experiences),
        },
    }

    # Log the quality check results
    if issues:
        logger.warning(f"Experience section quality issues found: {issues}")
    if warnings:
        logger.info(f"Experience section quality warnings: {warnings}")

    logger.info(f"Experience section quality check: {quality} (score: {overall_score}/100)")

    # Add iterative improvement status for backward compatibility tracking
    result["iterative_improvement_enabled"] = ENABLE_ITERATIVE_IMPROVEMENT

    return result


# ==============================================================================
# Iteration Progress Logging
# ==============================================================================


def log_iteration_progress(iteration_history: dict[str, list[BulletDraft]]) -> None:
    """
    Log detailed iteration progress for each experience entry.

    This function provides visibility into the iterative improvement process,
    showing how bullets evolved across iterations and whether quality thresholds
    were met.

    Args:
        iteration_history: Dictionary mapping company names to lists of BulletDraft objects

    Example Output:
        === Iteration Progress for Tech Corp ===
        Iteration 1: Score 55/100, Issues: 3
        Critique: Bullet 1: Missing structure (metric, method)...
        Iteration 2: Score 72/100, Issues: 2
        Critique: Bullet 2: Too generic (no technology)...
        Iteration 3: Score 88/100, Issues: 0
        [OK] Quality threshold met: 88/100

    Design Note:
        Detailed logging helps debug issues and provides transparency into
        the improvement process for monitoring and analysis.
    """
    if not iteration_history:
        logger.warning("No iteration history to log")
        return

    logger.info("\n" + "=" * 70)
    logger.info("ITERATIVE IMPROVEMENT SUMMARY")
    logger.info("=" * 70)

    total_iterations = 0
    total_entries = len(iteration_history)

    for company, drafts in iteration_history.items():
        logger.info(f"\n--- Iteration Progress for {company} ---")

        for draft in drafts:
            logger.info(
                f"  Iteration {draft.iteration}: "
                f"Score {draft.quality_score}/100, "
                f"Issues: {len(draft.issues_found)}"
            )

            # Log critique if quality threshold not met
            if draft.quality_score < QUALITY_THRESHOLD:
                logger.info(f"    Critique: {draft.critique[:200]}...")

            total_iterations += 1

        # Log final status
        final_draft = drafts[-1]
        if final_draft.quality_score >= QUALITY_THRESHOLD:
            logger.info(f"  [OK] Quality threshold met: {final_draft.quality_score}/100")
        else:
            logger.warning(
                f"  [WARNING] Max iterations reached. Final score: {final_draft.quality_score}/100"
            )

    # Overall summary
    logger.info("\n" + "-" * 70)
    logger.info(f"Total entries processed: {total_entries}")
    logger.info(f"Total iterations performed: {total_iterations}")
    logger.info(f"Average iterations per entry: {total_iterations / total_entries:.1f}")
    logger.info("=" * 70 + "\n")


# ==============================================================================
# Utility Functions
# ==============================================================================


def get_agent_info() -> dict:
    """
    Get information about this agent for debugging or monitoring.

    Returns:
        Dictionary with agent metadata

    Example:
        >>> info = get_agent_info()
        >>> print(info["name"])
        'Experience Section Optimizer'
    """
    config = _load_agent_config()
    return {
        "name": "Experience Section Optimizer",
        "role": config.get("role", "Unknown"),
        "llm": config.get("llm", "Unknown"),
        "tools": [],
        "output_model": "OptimizedExperienceSection",
        "content_type": "work_experience",
    }


# ==============================================================================
# Testing Block
# ==============================================================================

if __name__ == "__main__":
    """
    Test the agent creation and configuration loading.
    Run this script directly to verify the agent can be created.
    """
    print("=" * 70)
    print("Experience Section Optimizer Agent - Test")
    print("=" * 70)

    # Test configuration loading
    print("\n--- Testing Configuration Loading ---")
    config = _load_agent_config()
    print(f"Role: {config.get('role', 'N/A')}")
    print(f"LLM: {config.get('llm', 'N/A')}")
    print(f"Temperature: {config.get('temperature', 'N/A')}")

    # Test agent creation
    print("\n--- Testing Agent Creation ---")
    try:
        agent = create_experience_optimizer_agent()
        print("SUCCESS: Agent created successfully")
        print(f"Agent role: {agent.role}")
        print(f"Tools assigned: {len(agent.tools)}")
    except Exception as e:
        print(f"FAILED: {str(e)}")

    # Display agent info
    print("\n--- Agent Information ---")
    info = get_agent_info()
    for key, value in info.items():
        print(f"{key}: {value}")

    # Test helper functions
    print("\n--- Testing Helper Functions ---")

    # Test action verb analysis
    print("\nAction Verb Analysis:")
    test_achievements = [
        "Led a team of 5 engineers to deliver project ahead of schedule",
        "Managed daily operations and coordinated with stakeholders",
        "Responsible for maintaining system uptime",
    ]
    verb_analysis = analyze_action_verbs(test_achievements)
    print(f"  Strong verbs: {verb_analysis['strong_verb_count']}/{len(test_achievements)}")
    print(f"  Strength score: {verb_analysis['strength_score']}/100")
    if verb_analysis["weak_bullets"]:
        print(f"  Weak bullets: {verb_analysis['weak_bullets']}")

    # Test quantification analysis
    print("\nQuantification Analysis:")
    test_achievements2 = [
        "Reduced infrastructure costs by 40% through optimization",
        "Improved system performance significantly",
        "Led team of 8 engineers to complete $2M project",
    ]
    quant_analysis = count_quantified_achievements(test_achievements2)
    print(f"  Quantified: {quant_analysis['quantified_count']}/{len(test_achievements2)}")
    print(f"  Rate: {quant_analysis['quantification_rate'] * 100:.0f}%")
    if quant_analysis["unquantified_bullets"]:
        print(f"  Needs metrics: {quant_analysis['unquantified_bullets']}")

    # Test quality check function with mock data
    print("\n--- Testing Quality Check Function ---")
    try:
        from src.data_models.strategy import AlignmentStrategy, SkillGap, SkillMatch

        # Create mock strategy
        mock_strategy = AlignmentStrategy(
            overall_fit_score=85.5,
            summary_of_strategy="Emphasize cloud and Python experience",
            identified_matches=[
                SkillMatch(
                    resume_skill="Python",
                    job_requirement="Python development",
                    match_score=95.0,
                    justification="Direct match",
                )
            ],
            identified_gaps=[
                SkillGap(
                    missing_skill="Kubernetes",
                    importance="nice_to_have",
                    suggestion="Mention if available",
                )
            ],
            keywords_to_integrate=["Python", "AWS", "Microservices", "Cloud", "Docker"],
            professional_summary_guidance="Emphasize cloud architecture",
            experience_guidance="Highlight AWS projects",
            skills_guidance="Prioritize cloud technologies",
        )

        # Create mock experience section
        mock_section = OptimizedExperienceSection(
            optimized_experiences=[
                Experience(
                    job_title="Senior Software Engineer",
                    company_name="Tech Corp",
                    start_date=date(2020, 1, 15),
                    end_date=None,
                    is_current_position=True,
                    location="San Francisco, CA",
                    description="Led development of cloud-native microservices platform using Python and AWS",
                    achievements=[
                        "Architected scalable microservices infrastructure using Python and AWS, reducing deployment time by 65%",
                        "Led cross-functional team of 8 engineers to deliver $2M cost-saving automation initiative",
                        "Implemented CI/CD pipeline with Docker and Kubernetes, improving release frequency by 300%",
                    ],
                    skills_used=["Python", "AWS", "Docker", "Kubernetes"],
                )
            ],
            optimization_notes="Emphasized cloud and Python experience",
            keywords_integrated=["Python", "AWS", "Docker", "Kubernetes"],
            relevance_scores={"Tech Corp": 95.0},
        )

        quality_result = check_experience_quality(mock_section, mock_strategy)
        print(f"Quality: {quality_result['quality']}")
        print(f"Overall Score: {quality_result['overall_score']}/100")
        print(f"Acceptable: {quality_result['is_acceptable']}")
        print("\nSection Metrics:")
        for key, value in quality_result["section_metrics"].items():
            print(f"  {key}: {value}")
        if quality_result["issues"]:
            print(f"\nIssues: {quality_result['issues']}")
        if quality_result["warnings"]:
            print(f"\nWarnings: {quality_result['warnings']}")

    except Exception as e:
        print(f"Quality check test failed: {str(e)}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 70)
