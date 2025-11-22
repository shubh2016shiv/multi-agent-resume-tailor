"""
Example: Gap Analysis Specialist Agent
=======================================

OBJECTIVE:
----------
This example demonstrates how the Gap Analysis Agent compares a candidate's
resume against a job description to identify skill matches, gaps, and provide
strategic recommendations for resume tailoring.

WHAT THIS AGENT DOES:
---------------------
1. Receives structured Resume and JobDescription data
2. Compares candidate skills against job requirements
3. Identifies:
   - Skill matches (what the candidate has that the job needs)
   - Critical gaps (must-have skills the candidate is missing)
   - Enhancement opportunities (nice-to-have skills to add)
4. Calculates:
   - Overall fit score (0-100)
   - Skill coverage percentage
   - Match confidence scores
5. Provides strategic guidance for resume optimization
6. Returns an AlignmentStrategy with actionable recommendations

INPUT:
------
- Sample resume markdown (from common.py)
- Sample job description markdown (from common.py)
  Note: In a real workflow, these would be the structured outputs from
  the Resume Extractor and Job Analyzer agents.

EXPECTED OUTPUT:
----------------
- A JSON object that matches the AlignmentStrategy model structure:
  {
    "overall_fit_score": 85,
    "summary_of_strategy": "...",
    "identified_matches": [
      {
        "resume_skill": "Python",
        "job_requirement": "Python programming",
        "match_score": 95,
        "justification": "..."
      }
    ],
    "identified_gaps": [
      {
        "missing_skill": "Kubernetes",
        "importance": "should_have",
        "suggestion": "..."
      }
    ],
    "keywords_to_integrate": ["AWS", "Microservices", "CI/CD"],
    "professional_summary_guidance": "...",
    "experience_guidance": "...",
    "skills_guidance": "..."
  }

STEP-BY-STEP PROCESS:
---------------------
Step 1: Load sample resume and job description data
Step 2: Create the Gap Analysis Agent
Step 3: Define the analysis task with both inputs
Step 4: Execute the agent (calls LLM to perform gap analysis)
Step 5: Access validated output via result.pydantic
Step 6: Display the analysis results and recommendations

WHY THIS MATTERS:
-----------------
This is the strategic planning step. The gap analysis:
- Shows how well the candidate matches the job
- Identifies what to emphasize in the resume
- Highlights what skills might need to be added or inferred
- Provides guidance for content optimization
- Calculates a fit score to assess competitiveness

STRUCTURED OUTPUT ENFORCEMENT:
------------------------------
This example uses CrewAI's `output_pydantic` parameter to enforce structured outputs.
This is the RECOMMENDED approach for ensuring LLM outputs match Pydantic schemas.
- No manual JSON parsing required
- No manual validation logic needed
- Automatic retry on validation failures
- Type-safe access to structured data
"""

import io
import sys
from pathlib import Path

# Fix for Windows console encoding issues
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from crewai import Crew, Process, Task  # noqa: E402

from examples.agents.common import get_job_desc_md, get_resume_md  # noqa: E402
from src.agents.gap_analysis_agent import create_gap_analysis_agent  # noqa: E402
from src.core.config import get_tasks_config  # noqa: E402
from src.core.logger import get_logger  # noqa: E402
from src.data_models.strategy import AlignmentStrategy  # noqa: E402

logger = get_logger(__name__)


def print_section(title: str, content: str, max_length: int = 500):
    """Helper to print formatted sections with truncation."""
    print(f"\n{'=' * 80}")
    print(f"{title}")
    print(f"{'=' * 80}")
    if len(content) > max_length:
        print(content[:max_length] + "...\n[Truncated for display]")
    else:
        print(content)


def main():
    print("\n" + "=" * 80)
    print("GAP ANALYSIS SPECIALIST AGENT - EDUCATIONAL EXAMPLE")
    print("=" * 80)
    print("\nThis example shows how the AI compares a resume against a job description")
    print("to identify matches, gaps, and provide strategic recommendations.\n")

    # ========================================================================
    # STEP 1: PREPARE INPUT DATA
    # ========================================================================
    print_section("STEP 1: INPUT DATA", "Loading sample resume and job description...")

    resume_md = get_resume_md()
    job_md = get_job_desc_md()

    print("\n[INPUT] Sample Resume:")
    print("-" * 40)
    print(resume_md[:200] + "..." if len(resume_md) > 200 else resume_md)
    print("-" * 40)

    print("\n[INPUT] Sample Job Description:")
    print("-" * 40)
    print(job_md[:200] + "..." if len(job_md) > 200 else job_md)
    print("-" * 40)

    print("\n[INFO] In a real workflow, these would be structured JSON objects")
    print("       from the Resume Extractor and Job Analyzer agents.")

    # ========================================================================
    # STEP 2: CREATE THE AGENT
    # ========================================================================
    print_section("STEP 2: AGENT CREATION", "Initializing the Gap Analysis Agent...")

    try:
        agent = create_gap_analysis_agent()
        print("\n[SUCCESS] Agent created successfully!")
        print(f"  Role: {agent.role}")
        print(
            f"  Goal: {agent.goal[:100]}..." if len(agent.goal) > 100 else f"  Goal: {agent.goal}"
        )
        print("\n[INFO] This agent is specialized in:")
        print("  - Comparing resumes against job requirements")
        print("  - Identifying skill matches and gaps")
        print("  - Calculating fit scores and coverage metrics")
        print("  - Providing strategic recommendations")
    except Exception as e:
        logger.error(f"Failed to create agent: {e}", exc_info=True)
        print(f"\n[ERROR] Failed to create agent: {e}")
        return

    # ========================================================================
    # STEP 3: DEFINE THE TASK
    # ========================================================================
    print_section("STEP 3: TASK DEFINITION", "Setting up the gap analysis task...")

    # Load task configuration from YAML
    try:
        tasks_config = get_tasks_config()
        task_config = tasks_config.get("create_alignment_strategy_task")

        if not task_config:
            raise ValueError("create_alignment_strategy_task not found in tasks.yaml")

        print("\n[SUCCESS] Task configuration loaded from src/config/tasks.yaml")

    except Exception as e:
        logger.error(f"Failed to load task config: {e}", exc_info=True)
        print(f"\n[ERROR] Failed to load task config: {e}")
        return

    # Combine the YAML description with the actual content
    # We prepend the content so the agent has the context immediately
    task_description = (
        f"RESUME CONTENT:\n{resume_md}\n\n"
        f"JOB DESCRIPTION CONTENT:\n{job_md}\n\n"
        f"INSTRUCTIONS:\n{task_config['description']}"
    )

    task = Task(
        description=task_description,
        expected_output=task_config["expected_output"],
        agent=agent,
        output_pydantic=AlignmentStrategy,  # ⭐ Structured output enforcement
    )

    print("\n[INFO] Task configured with:")
    print("  - Description from tasks.yaml")
    print("  - Expected output from tasks.yaml")
    print("  - output_pydantic=AlignmentStrategy (Structured Output Enforcement)")

    # ========================================================================
    # STEP 4: EXECUTE THE AGENT
    # ========================================================================
    print_section("STEP 4: AGENT EXECUTION", "Running the agent (this calls the LLM)...")

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )

    print("\n[INFO] The agent will now:")
    print("  1. Compare candidate skills against job requirements")
    print("  2. Identify matches and calculate match scores")
    print("  3. Identify gaps (missing skills)")
    print("  4. Calculate overall fit score")
    print("  5. Generate strategic recommendations")
    print("\n[WAIT] This may take 30-60 seconds as the LLM performs the analysis...\n")

    try:
        result = crew.kickoff()
    except Exception as e:
        logger.error(f"Error during agent execution: {e}", exc_info=True)
        print(f"\n[ERROR] Agent execution failed: {e}")
        return

    # ========================================================================
    # STEP 5: PROCESS AND VALIDATE RESULTS
    # ========================================================================
    print_section("STEP 5: OUTPUT PROCESSING", "Accessing the structured data...")

    try:
        # Access the validated Pydantic object directly
        strategy = result.pydantic
        print("\n[SUCCESS] Structured output received from agent")

        if strategy:
            print("\n" + "=" * 80)
            print("GAP ANALYSIS SUCCESSFUL!")
            print("=" * 80)
            print("\n[VALIDATED] Output automatically validated by CrewAI")

            # Display key analysis results
            print("\n" + "-" * 80)
            print("GAP ANALYSIS RESULTS")
            print("-" * 80)

            print(f"\n  Overall Fit Score: {strategy.overall_fit_score}/100")
            if strategy.overall_fit_score >= 80:
                print("    Status: Strong Match ✓")
            elif strategy.overall_fit_score >= 60:
                print("    Status: Good Match")
            else:
                print("    Status: Needs Improvement")

            print("\n  Strategy Summary:")
            summary = strategy.summary_of_strategy or "Not provided"
            print(f"    {summary[:300]}..." if len(summary) > 300 else f"    {summary}")

            print(f"\n  Identified Matches: {len(strategy.identified_matches)} match(es)")
            if strategy.identified_matches:
                print("    Top Matches:")
                for i, match in enumerate(strategy.identified_matches[:5], 1):
                    print(f"      {i}. {match.resume_skill} → {match.job_requirement}")
                    print(f"         Match Score: {match.match_score}/100")
                    justification = (
                        match.justification[:100] + "..."
                        if len(match.justification) > 100
                        else match.justification
                    )
                    print(f"         Justification: {justification}")

            print(f"\n  Identified Gaps: {len(strategy.identified_gaps)} gap(s)")
            if strategy.identified_gaps:
                must_have_gaps = [
                    g for g in strategy.identified_gaps if g.importance == "must_have"
                ]
                should_have_gaps = [
                    g for g in strategy.identified_gaps if g.importance == "should_have"
                ]
                nice_to_have_gaps = [
                    g for g in strategy.identified_gaps if g.importance == "nice_to_have"
                ]

                if must_have_gaps:
                    print("    Critical Gaps (Must-Have):")
                    for gap in must_have_gaps[:3]:
                        print(f"      ⚠ {gap.missing_skill}")
                        suggestion = (
                            gap.suggestion[:100] + "..."
                            if len(gap.suggestion) > 100
                            else gap.suggestion
                        )
                        print(f"         Suggestion: {suggestion}")

                if should_have_gaps:
                    print("    Important Gaps (Should-Have):")
                    for gap in should_have_gaps[:3]:
                        print(f"      • {gap.missing_skill}")

                if nice_to_have_gaps:
                    print(f"    Enhancement Opportunities (Nice-to-Have): {len(nice_to_have_gaps)}")

            print(f"\n  Keywords to Integrate: {len(strategy.keywords_to_integrate)} keyword(s)")
            if strategy.keywords_to_integrate:
                print(f"    {', '.join(strategy.keywords_to_integrate[:10])}")
                if len(strategy.keywords_to_integrate) > 10:
                    print(f"    ... and {len(strategy.keywords_to_integrate) - 10} more")

            print("\n  Professional Summary Guidance:")
            summary_guidance = strategy.professional_summary_guidance or "Not provided"
            print(
                f"    {summary_guidance[:200]}..."
                if len(summary_guidance) > 200
                else f"    {summary_guidance}"
            )

            print("\n  Experience Guidance:")
            exp_guidance = strategy.experience_guidance or "Not provided"
            print(
                f"    {exp_guidance[:200]}..." if len(exp_guidance) > 200 else f"    {exp_guidance}"
            )

            print("\n  Skills Guidance:")
            skills_guidance = strategy.skills_guidance or "Not provided"
            print(
                f"    {skills_guidance[:200]}..."
                if len(skills_guidance) > 200
                else f"    {skills_guidance}"
            )

            print("\n" + "-" * 80)
            print("\n[SUCCESS] Gap analysis completed and validated!")
            print("\n[INFO] This strategy can now be used to:")
            print("  - Generate tailored professional summary")
            print("  - Optimize experience section bullets")
            print("  - Prioritize and add skills")
            print("  - Integrate ATS keywords")

        else:
            print("\n[ERROR] Validation failed. Result.pydantic is None.")

    except Exception as e:
        logger.error(f"Error processing output: {e}", exc_info=True)
        print(f"\n[ERROR] Could not process output: {e}")

    print("\n" + "=" * 80)
    print("EXAMPLE COMPLETE")
    print("=" * 80)
    print("\nThis example demonstrated how the Gap Analysis Agent:")
    print("  1. Compares resume skills against job requirements")
    print("  2. Identifies matches and calculates scores")
    print("  3. Identifies gaps (missing skills)")
    print("  4. Provides strategic recommendations")
    print("  5. Generates an alignment strategy for optimization")
    print("\nNext steps: Use this strategy with content generation agents:")
    print("  - Professional Summary Writer")
    print("  - Experience Section Optimizer")
    print("  - Skills Section Optimizer")


if __name__ == "__main__":
    main()
