"""
Example: Job Analyzer Agent
===========================

OBJECTIVE:
----------
This example demonstrates how the Job Analyzer Agent works. The agent takes
a raw job description (as text) and extracts structured information including
requirements, skills, ATS keywords, and job level classification.

WHAT THIS AGENT DOES:
---------------------
1. Receives a job description in text format
2. Uses an AI language model to intelligently analyze and extract:
   - Job title and company name
   - Job level (entry, junior, mid-level, senior, lead, etc.)
   - Location and work type (remote, hybrid, on-site)
   - Job summary (1-3 sentence overview)
   - Requirements (must-have, should-have, nice-to-have)
   - ATS keywords (important terms for applicant tracking systems)
   - Years of experience required
3. Structures all extracted data into a validated JSON format
4. Categorizes requirements by importance level

INPUT:
------
- Sample job description markdown (from common.py)
  Example: A job posting for a Senior Backend Cloud Engineer position

EXPECTED OUTPUT:
----------------
- A JSON object that matches the JobDescription model structure:
  {
    "job_title": "Senior Backend Cloud Engineer",
    "company_name": "CloudScale Solutions",
    "job_level": "senior",
    "location": "Remote / New York, NY",
    "summary": "...",
    "full_text": "...",
    "requirements": [
      {
        "requirement": "5+ years of experience in backend development",
        "importance": "must_have",
        "years_required": 5
      },
      ...
    ],
    "ats_keywords": ["Python", "AWS", "Lambda", "DynamoDB", ...]
  }

STEP-BY-STEP PROCESS:
---------------------
Step 1: Load sample job description data
Step 2: Create the Job Analyzer Agent
Step 3: Define the analysis task with clear instructions
Step 4: Execute the agent (calls LLM to extract structured data)
Step 5: Parse and validate the output
Step 6: Display the analyzed information

WHY THIS MATTERS:
-----------------
This is a critical step in resume tailoring. By understanding job requirements
in a structured way, we can:
- Compare candidate skills against requirements
- Identify missing skills (gaps)
- Prioritize which skills to emphasize
- Extract ATS keywords for optimization
"""

import json
import sys
import io
from pathlib import Path

# Fix for Windows console encoding issues
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from crewai import Crew, Process, Task
from src.agents.job_analyzer_agent import (
    create_job_analyzer_agent,
    validate_job_output,
)
from src.core.logger import get_logger
from examples.agents.common import get_job_desc_md, parse_json_output

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
    print("JOB ANALYZER AGENT - EDUCATIONAL EXAMPLE")
    print("=" * 80)
    print("\nThis example shows how the AI analyzes a job description and extracts")
    print("structured requirements, skills, and ATS keywords.\n")
    
    # ========================================================================
    # STEP 1: PREPARE INPUT DATA
    # ========================================================================
    print_section("STEP 1: INPUT DATA", "Loading sample job description...")
    
    job_md = get_job_desc_md()
    print("\n[INPUT] Sample Job Description:")
    print("-" * 40)
    print(job_md[:400] + "..." if len(job_md) > 400 else job_md)
    print("-" * 40)
    print(f"\n[INFO] Job description length: {len(job_md)} characters")
    
    # ========================================================================
    # STEP 2: CREATE THE AGENT
    # ========================================================================
    print_section("STEP 2: AGENT CREATION", "Initializing the Job Analyzer Agent...")
    
    try:
        agent = create_job_analyzer_agent()
        print(f"\n[SUCCESS] Agent created successfully!")
        print(f"  Role: {agent.role}")
        print(f"  Goal: {agent.goal[:100]}..." if len(agent.goal) > 100 else f"  Goal: {agent.goal}")
        print("\n[INFO] This agent is specialized in:")
        print("  - Parsing job descriptions of various formats")
        print("  - Identifying requirements and categorizing by importance")
        print("  - Extracting ATS keywords for optimization")
        print("  - Classifying job level (entry, mid-level, senior, etc.)")
    except Exception as e:
        logger.error(f"Failed to create agent: {e}", exc_info=True)
        print(f"\n[ERROR] Failed to create agent: {e}")
        return
    
    # ========================================================================
    # STEP 3: DEFINE THE TASK
    # ========================================================================
    print_section("STEP 3: TASK DEFINITION", "Setting up the analysis task...")
    
    # CRITICAL FIX: Disable tools for this example since we are providing text directly
    # This prevents the agent from trying to use the parse_job_description tool on a non-existent file
    agent.tools = []
    print("\n[INFO] Tools disabled for this example (using direct text input)")
    
    # Load task configuration from YAML
    try:
        from src.core.config import get_tasks_config
        tasks_config = get_tasks_config()
        task_config = tasks_config.get("analyze_job_description_task")
        
        if not task_config:
            raise ValueError("analyze_job_description_task not found in tasks.yaml")
            
        print("\n[SUCCESS] Task configuration loaded from src/config/tasks.yaml")
        
    except Exception as e:
        logger.error(f"Failed to load task config: {e}", exc_info=True)
        print(f"\n[ERROR] Failed to load task config: {e}")
        return

    # Combine the YAML description with the actual content
    # We prepend the content so the agent has the context immediately
    task_description = (
        f"JOB DESCRIPTION CONTENT:\n{job_md}\n\n"
        f"INSTRUCTIONS:\n{task_config['description']}"
    )
    
    from src.data_models.job import JobDescription
    
    task = Task(
        description=task_description,
        expected_output=task_config['expected_output'],
        agent=agent,
        output_pydantic=JobDescription,  # ⭐ Structured output enforcement
    )
    
    print("\n[INFO] Task configured with:")
    print("  - Description from tasks.yaml")
    print("  - Expected output from tasks.yaml")
    print("  - output_pydantic=JobDescription (Structured Output Enforcement)")
    
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
    print("  1. Analyze the job description using AI")
    print("  2. Identify and categorize all requirements")
    print("  3. Extract ATS keywords and technical terms")
    print("  4. Classify the job level")
    print("  5. Structure the data into JSON format")
    print("\n[WAIT] This may take 30-60 seconds as the LLM processes the job description...\n")
    
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
        validated_job = result.pydantic
        print("\n[SUCCESS] Structured output received from agent")
        
        if validated_job:
            print("\n" + "=" * 80)
            print("ANALYSIS SUCCESSFUL!")
            print("=" * 80)
            print("\n[VALIDATED] Output automatically validated by CrewAI")
            
            # Display key analyzed information
            print("\n" + "-" * 80)
            print("JOB ANALYSIS SUMMARY")
            print("-" * 80)
            print(f"  Job Title: {validated_job.job_title}")
            print(f"  Company: {validated_job.company_name}")
            print(f"  Job Level: {validated_job.job_level.value}")
            print(f"  Location: {validated_job.location or 'Not specified'}")
            
            print(f"\n  Job Summary:")
            summary = validated_job.summary or "Not provided"
            print(f"    {summary[:200]}..." if len(summary) > 200 else f"    {summary}")
            
            print(f"\n  Requirements: {len(validated_job.requirements)} total")
            must_have = [r for r in validated_job.requirements if r.importance.value == "must_have"]
            should_have = [r for r in validated_job.requirements if r.importance.value == "should_have"]
            nice_to_have = [r for r in validated_job.requirements if r.importance.value == "nice_to_have"]
            
            print(f"    Must-Have: {len(must_have)} requirements")
            for req in must_have[:3]:
                years = f" ({req.years_required} years)" if req.years_required else ""
                print(f"      • {req.requirement}{years}")
            
            print(f"    Should-Have: {len(should_have)} requirements")
            for req in should_have[:2]:
                print(f"      • {req.requirement}")
            
            print(f"    Nice-to-Have: {len(nice_to_have)} requirements")
            
            print(f"\n  ATS Keywords: {len(validated_job.ats_keywords)} keywords")
            keywords_sample = validated_job.ats_keywords[:15]
            print(f"    Sample: {', '.join(keywords_sample)}")
            if len(validated_job.ats_keywords) > 15:
                print(f"    ... and {len(validated_job.ats_keywords) - 15} more")
            
            print(f"\n  Must-Have Skills: {len(validated_job.must_have_skills)} skills")
            if validated_job.must_have_skills:
                print(f"    {', '.join(validated_job.must_have_skills[:10])}")
            
            print("\n" + "-" * 80)
            print("\n[SUCCESS] Job description successfully analyzed and validated!")
            print("\n[INFO] This structured data can now be used for:")
            print("  - Comparing against candidate resumes")
            print("  - Identifying skill gaps and matches")
            print("  - Generating alignment strategies")
            print("  - Optimizing resumes with ATS keywords")
            
        else:
            print("\n[ERROR] Validation failed. Result.pydantic is None.")
            
    except Exception as e:
        logger.error(f"Error processing output: {e}", exc_info=True)
        print(f"\n[ERROR] Could not process output: {e}")
    
    print("\n" + "=" * 80)
    print("EXAMPLE COMPLETE")
    print("=" * 80)
    print("\nThis example demonstrated how the Job Analyzer Agent:")
    print("  1. Takes unstructured job description text as input")
    print("  2. Uses AI to intelligently extract requirements and keywords")
    print("  3. Categorizes requirements by importance level")
    print("  4. Structures data into validated JSON format")
    print("\nNext steps: Use this analyzed data with other agents for:")
    print("  - Gap analysis (comparing resume vs job requirements)")
    print("  - Resume content optimization")
    print("  - ATS keyword integration")


if __name__ == "__main__":
    main()
