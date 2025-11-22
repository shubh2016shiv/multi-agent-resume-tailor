"""
Example: Professional Summary Writer Agent
==========================================

OBJECTIVE:
----------
This example demonstrates how the Professional Summary Writer Agent generates
a tailored professional summary based on a resume, job description, and
alignment strategy. The agent creates multiple draft versions and selects
the best one.

WHAT THIS AGENT DOES:
---------------------
1. Receives:
   - Structured Resume data
   - Structured JobDescription data
   - AlignmentStrategy with guidance
2. Generates multiple professional summary drafts using different strategies:
   - Hook-Value-Future format
   - Story Spine format
   - ATS-Optimized format
   - Achievement-focused format
3. Evaluates each draft and assigns scores
4. Selects the best version as the recommendation
5. Integrates ATS keywords naturally
6. Ensures the summary aligns with job requirements

INPUT:
------
- Sample resume markdown (from common.py)
- Sample job description markdown (from common.py)
- Mock Alignment Strategy (simplified for example)
  In a real workflow, this would come from the Gap Analysis Agent.

EXPECTED OUTPUT:
----------------
- A JSON object that matches the ProfessionalSummary model structure:
  {
    "recommended_version": "Hook-Value-Future",
    "writing_notes": "...",
    "drafts": [
      {
        "version_name": "Hook-Value-Future",
        "strategy_used": "...",
        "content": "Senior Software Engineer with 5+ years...",
        "critique": "...",
        "score": 92
      },
      ...
    ]
  }

STEP-BY-STEP PROCESS:
---------------------
Step 1: Load sample resume, job description, and mock strategy
Step 2: Create the Professional Summary Writer Agent
Step 3: Define the writing task with all inputs
Step 4: Execute the agent (calls LLM to generate summaries)
Step 5: Parse and validate the output
Step 6: Display the generated drafts and recommendation

WHY THIS MATTERS:
-----------------
The professional summary is the first thing recruiters see. A well-crafted
summary that:
- Grabs attention immediately
- Highlights relevant experience
- Integrates ATS keywords naturally
- Shows value proposition
can significantly improve resume visibility and interview chances.
"""

import io
import json
import sys
from pathlib import Path

# Fix Windows console encoding for UTF-8 characters (emojis, etc.)
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from crewai import Crew, Process, Task
from src.agents.summary_writer_agent import create_summary_writer_agent, ProfessionalSummary
from src.core.logger import get_logger
from examples.agents.common import get_resume_md, get_job_desc_md

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
    print("PROFESSIONAL SUMMARY WRITER AGENT - EDUCATIONAL EXAMPLE")
    print("=" * 80)
    print("\nThis example shows how the AI generates tailored professional summaries")
    print("with multiple draft versions and selects the best one.\n")
    
    # ========================================================================
    # STEP 1: PREPARE INPUT DATA
    # ========================================================================
    print_section("STEP 1: INPUT DATA", "Loading sample resume, job description, and strategy...")
    
    resume_md = get_resume_md()
    job_md = get_job_desc_md()
    
    # Mock strategy for demonstration
    # In a real workflow, this would come from the Gap Analysis Agent
    strategy_md = """
    ALIGNMENT STRATEGY:
    - Overall Fit Score: 85/100
    - Key Matches: Python, AWS, Backend Development, Microservices
    - Key Gaps: Go (missing), Kubernetes (missing)
    - Summary Guidance: Emphasize strong backend experience with Python and AWS. 
      Mention ability to learn new languages (like Go) quickly. Highlight cloud 
      infrastructure skills and microservices architecture experience.
    - Keywords to Integrate: Python, AWS, Microservices, Cloud, CI/CD, DynamoDB, Lambda
    """
    
    print("\n[INPUT] Sample Resume:")
    print("-" * 40)
    print(resume_md[:150] + "..." if len(resume_md) > 150 else resume_md)
    print("-" * 40)
    
    print("\n[INPUT] Sample Job Description:")
    print("-" * 40)
    print(job_md[:150] + "..." if len(job_md) > 150 else job_md)
    print("-" * 40)
    
    print("\n[INPUT] Alignment Strategy (Mock):")
    print("-" * 40)
    print(strategy_md)
    print("-" * 40)
    
    print("\n[INFO] In a real workflow, the strategy would be the structured output")
    print("       from the Gap Analysis Agent.")
    
    # ========================================================================
    # STEP 2: CREATE THE AGENT
    # ========================================================================
    print_section("STEP 2: AGENT CREATION", "Initializing the Professional Summary Writer Agent...")
    
    try:
        agent = create_summary_writer_agent()
        print(f"\n[SUCCESS] Agent created successfully!")
        print(f"  Role: {agent.role}")
        print(f"  Goal: {agent.goal[:100]}..." if len(agent.goal) > 100 else f"  Goal: {agent.goal}")
        print("\n[INFO] This agent is specialized in:")
        print("  - Writing compelling professional summaries")
        print("  - Generating multiple draft versions")
        print("  - Integrating ATS keywords naturally")
        print("  - Evaluating and selecting the best version")
    except Exception as e:
        logger.error(f"Failed to create agent: {e}", exc_info=True)
        print(f"\n[ERROR] Failed to create agent: {e}")
        return
    
    # ========================================================================
    # STEP 3: DEFINE THE TASK
    # ========================================================================
    print_section("STEP 3: TASK DEFINITION", "Loading task configuration from tasks.yaml...")
    
    # Load task configuration from YAML
    try:
        from src.core.config import get_tasks_config
        tasks_config = get_tasks_config()
        task_config = tasks_config.get("write_professional_summary_task")
        
        if not task_config:
            raise ValueError("write_professional_summary_task not found in tasks.yaml")
            
        print("\n[SUCCESS] Task configuration loaded from src/config/tasks.yaml")
        
    except Exception as e:
        logger.error(f"Failed to load task config: {e}", exc_info=True)
        print(f"\n[ERROR] Failed to load task config: {e}")
        return
    
    # Combine the YAML description with the actual content
    # We prepend the content so the agent has the context immediately
    task_description = (
        f"RESUME:\n{resume_md}\n\n"
        f"JOB DESCRIPTION:\n{job_md}\n\n"
        f"ALIGNMENT STRATEGY:\n{strategy_md}\n\n"
        f"INSTRUCTIONS:\n{task_config['description']}"
    )
    
    task = Task(
        description=task_description,
        expected_output=task_config['expected_output'],
        agent=agent,
        output_pydantic=ProfessionalSummary,  # ⭐ Structured output enforcement
    )
    
    print("\n[INFO] Task configured with:")
    print("  - Resume and job description inputs")
    print("  - Alignment strategy guidance")
    print("  - Centralized task definition from tasks.yaml")
    print("  - Multiple writing strategy requirements")
    print("  - Evaluation and selection criteria")
    
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
    print("  1. Analyze the resume and job requirements")
    print("  2. Generate multiple summary drafts using different strategies")
    print("  3. Evaluate each draft for quality and effectiveness")
    print("  4. Select the best version")
    print("\n[WAIT] This may take 45-90 seconds as the LLM generates multiple drafts...\n")
    
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
    
    print("\n[SUCCESS] Structured output received from agent")
    
    # Access the validated Pydantic object directly
    summary_obj = result.pydantic
    
    print("\n" + "=" * 80)
    print("SUMMARY GENERATION SUCCESSFUL!")
    print("=" * 80)
    print("\n[VALIDATED] Output automatically validated by CrewAI")
    
    # Display key results
    print("\n" + "-" * 80)
    print("PROFESSIONAL SUMMARY RESULTS")
    print("-" * 80)
    
    print(f"\n  Recommended Version: {summary_obj.recommended_version}")
    print(f"  Writing Notes: {summary_obj.writing_notes or 'Not provided'}")
    
    print(f"\n  Drafts Generated: {len(summary_obj.drafts)} version(s)")
    print("\n" + "-" * 40)
    
    for i, draft in enumerate(summary_obj.drafts, 1):
        print(f"\n  DRAFT {i}: {draft.version_name}")
        print(f"    Strategy: {draft.strategy_used}")
        print(f"    Score: {draft.score}/100")
        print(f"    Content:")
        # Format the content with proper indentation
        content_lines = draft.content.split('\n')
        for line in content_lines:
            print(f"      {line}")
        if draft.critique:
            print(f"    Critique: {draft.critique[:150]}..." if len(draft.critique) > 150 else f"    Critique: {draft.critique}")
        if draft.version_name == summary_obj.recommended_version:
            print("    ⭐ RECOMMENDED VERSION")
        print("-" * 40)
    
    print("\n" + "-" * 80)
    print("\n[SUCCESS] Professional summary generated and validated!")
    print("\n[INFO] The recommended summary:")
    print("  - Is tailored to the job requirements")
    print("  - Integrates ATS keywords naturally")
    print("  - Highlights relevant experience")
    print("  - Is optimized for both ATS and human reviewers")
    
    print("\n" + "=" * 80)
    print("EXAMPLE COMPLETE")
    print("=" * 80)
    print("\nThis example demonstrated how the Professional Summary Writer Agent:")
    print("  1. Takes resume, job description, and strategy as input")
    print("  2. Generates multiple summary drafts using different strategies")
    print("  3. Evaluates and scores each draft")
    print("  4. Selects the best version as recommendation")
    print("  5. Integrates ATS keywords naturally")
    print("\nNext steps: Use this summary with other agents for:")
    print("  - Experience section optimization")
    print("  - Skills section optimization")
    print("  - Final ATS optimization and assembly")


if __name__ == "__main__":
    main()
