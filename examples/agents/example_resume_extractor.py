"""
Example: Resume Extractor Agent
================================

OBJECTIVE:
----------
This example demonstrates how the Resume Extractor Agent works. The agent takes
a raw resume (as markdown text) and extracts structured information into a 
validated JSON format that matches our Resume data model.

WHAT THIS AGENT DOES:
---------------------
1. Receives a resume in markdown format (could be from PDF, DOCX, or text file)
2. Uses an AI language model to intelligently parse and extract:
   - Personal information (name, email, phone, location)
   - Professional summary
   - Work experience (job titles, companies, dates, achievements)
   - Education (degrees, institutions, graduation years)
   - Skills (technical and soft skills)
   - Certifications and languages
3. Structures all extracted data into a validated JSON format
4. Ensures data quality and completeness

INPUT:
------
- Sample resume markdown (from common.py)
  Example: A resume with basic information about a software engineer

EXPECTED OUTPUT:
----------------
- A JSON object that matches the Resume model structure:
  {
    "full_name": "John Doe",
    "email": "john.doe@email.com",
    "phone_number": "123-456-7890",
    "location": "New York",
    "professional_summary": "...",
    "work_experience": [
      {
        "job_title": "Software Engineer",
        "company_name": "Tech Corp",
        "start_date": "2020-01-01",
        "end_date": null,
        "is_current_position": true,
        "description": "...",
        "achievements": ["...", "..."],
        "skills_used": ["Python", "AWS", ...]
      }
    ],
    "education": [...],
    "skills": [...],
    "certifications": [...],
    "languages": [...]
  }

STEP-BY-STEP PROCESS:
---------------------
Step 1: Load sample resume data
Step 2: Create the Resume Extractor Agent
Step 3: Define the extraction task with clear instructions
Step 4: Execute the agent (calls LLM to extract structured data)
Step 5: Parse and validate the output
Step 6: Display the extracted information

WHY THIS MATTERS:
-----------------
This is the first step in the resume tailoring process. Without structured data,
we cannot compare resumes to job descriptions or generate tailored content.
The agent ensures we have clean, validated data to work with.
"""

import json
import sys
from pathlib import Path

# Add project root to path to ensure imports work
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from crewai import Crew, Process, Task
from src.agents.resume_extractor_agent import create_resume_extractor_agent
from src.core.config import get_tasks_config
from src.core.logger import get_logger
from src.data_models.resume import Resume
from examples.agents.common import get_resume_md

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
    print("RESUME EXTRACTOR AGENT - EDUCATIONAL EXAMPLE")
    print("=" * 80)
    print("\nThis example shows how the AI extracts structured data from a resume.")
    print("Follow along to see each step of the process.\n")
    
    # ========================================================================
    # STEP 1: PREPARE INPUT DATA
    # ========================================================================
    print_section("STEP 1: INPUT DATA", "Loading sample resume markdown...")
    
    resume_md = get_resume_md()
    print("\n[INPUT] Sample Resume Markdown:")
    print("-" * 40)
    print(resume_md[:300] + "..." if len(resume_md) > 300 else resume_md)
    print("-" * 40)
    print(f"\n[INFO] Resume length: {len(resume_md)} characters")
    
    # ========================================================================
    # STEP 2: CREATE THE AGENT
    # ========================================================================
    print_section("STEP 2: AGENT CREATION", "Initializing the Resume Extractor Agent...")
    
    try:
        agent = create_resume_extractor_agent()
        print(f"\n[SUCCESS] Agent created successfully!")
        print(f"  Role: {agent.role}")
        print(f"  Goal: {agent.goal[:100]}..." if len(agent.goal) > 100 else f"  Goal: {agent.goal}")
        print("\n[INFO] This agent is specialized in:")
        print("  - Parsing various resume formats")
        print("  - Extracting structured information")
        print("  - Validating data completeness")
    except Exception as e:
        logger.error(f"Failed to create agent: {e}", exc_info=True)
        print(f"\n[ERROR] Failed to create agent: {e}")
        return
    
    # ========================================================================
    # STEP 3: DEFINE THE TASK
    # ========================================================================
    print_section("STEP 3: TASK DEFINITION", "Setting up the extraction task...")
    
    # Load task configuration from tasks.yaml (same as real application)
    try:
        tasks_config = get_tasks_config()
        task_config = tasks_config.get("extract_resume_content_task", {})
        
        if not task_config:
            raise ValueError("extract_resume_content_task not found in tasks.yaml")
        
        # Get the base task description and expected_output from config
        base_description = task_config.get("description", "")
        base_expected_output = task_config.get("expected_output", "")
        
        print("\n[INFO] Loaded task configuration from tasks.yaml")
        print(f"  Task: extract_resume_content_task")
        print(f"  Agent: {task_config.get('agent', 'N/A')}")
        
    except Exception as e:
        logger.error(f"Failed to load task config: {e}", exc_info=True)
        print(f"\n[ERROR] Failed to load task configuration: {e}")
        return
    
    # Adapt the task description for the example context
    # The real task expects a file path, but we're providing markdown content directly
    # 
    # IMPORTANT: We use output_pydantic=Resume to enforce structured output.
    # This is the industry-standard approach in CrewAI for ensuring LLM outputs
    # match our Pydantic model schema. It eliminates the need for manual JSON
    # parsing and validation.
    task_description = (
        f"IMPORTANT: The resume content is provided below as Markdown text. "
        f"DO NOT use the Resume Parser Tool. Work directly with the content provided in this task.\n\n"
        f"RESUME CONTENT:\n{resume_md}\n\n"
        f"{base_description}\n\n"
        f"CRITICAL INSTRUCTIONS:\n"
        f"- Extract ALL information accurately from the resume\n"
        f"- For dates: Use ISO format YYYY-MM-DD (e.g., '2020-01-01'). If only year is available, use January 1st of that year\n"
        f"- For current positions: Set end_date to null and is_current_position to true\n"
        f"- Ensure all required fields are populated at the correct schema level\n"
        f"- Use empty lists [] for certifications and languages if none are found\n"
    )
    
    task = Task(
        description=task_description,
        expected_output=base_expected_output,
        agent=agent,
        output_pydantic=Resume,  # ⭐ STRUCTURED OUTPUT ENFORCEMENT
    )
    
    print("\n[INFO] Task configured with:")
    print("  - Real task description from tasks.yaml")
    print("  - Real expected_output from tasks.yaml")
    print("  - Adapted for example context (markdown content provided directly)")
    print("  - Schema requirements to match Resume model")
    
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
    print("  1. Analyze the resume content using AI")
    print("  2. Identify and extract all relevant information")
    print("  3. Structure the data into JSON format")
    print("  4. Validate the output against the Resume model")
    print("\n[WAIT] This may take 30-60 seconds as the LLM processes the resume...\n")
    
    try:
        result = crew.kickoff()
    except Exception as e:
        logger.error(f"Error during agent execution: {e}", exc_info=True)
        print(f"\n[ERROR] Agent execution failed: {e}")
        return
    
    # ========================================================================
    # STEP 5: ACCESS STRUCTURED OUTPUT
    # ========================================================================
    print_section("STEP 5: ACCESSING STRUCTURED OUTPUT", "Retrieving validated Resume object...")
    
    print("\n[INFO] With output_pydantic, the result is automatically validated!")
    print("  - No manual JSON parsing needed")
    print("  - No manual validation needed")
    print("  - Direct access to typed Pydantic object")
    
    try:
        # With output_pydantic, we can access the validated Resume object directly
        # CrewAI has already validated it against the Resume model schema
        validated_resume = result.pydantic
        
        if validated_resume:
            print("\n" + "=" * 80)
            print("EXTRACTION SUCCESSFUL!")
            print("=" * 80)
            print("\n[VALIDATED] Output automatically validated by CrewAI")
            
            # Display key extracted information
            print("\n" + "-" * 80)
            print("EXTRACTED INFORMATION SUMMARY")
            print("-" * 80)
            print(f"  Name: {validated_resume.full_name}")
            print(f"  Email: {validated_resume.email}")
            print(f"  Phone: {validated_resume.phone_number or 'Not provided'}")
            print(f"  Location: {validated_resume.location or 'Not provided'}")
            print(f"\n  Professional Summary:")
            summary = validated_resume.professional_summary or "Not provided"
            print(f"    {summary[:150]}..." if len(summary) > 150 else f"    {summary}")
            
            print(f"\n  Work Experience: {len(validated_resume.work_experience)} role(s)")
            for i, exp in enumerate(validated_resume.work_experience[:3], 1):
                print(f"    {i}. {exp.job_title} at {exp.company_name}")
                print(f"       Period: {exp.start_date} to {exp.end_date or 'Present'}")
                print(f"       Achievements: {len(exp.achievements)} items")
            
            print(f"\n  Education: {len(validated_resume.education)} entry/entries")
            for edu in validated_resume.education:
                print(f"    - {edu.degree} from {edu.institution_name} ({edu.graduation_year})")
            
            print(f"\n  Skills: {len(validated_resume.skills)} skill(s)")
            skill_names = [s.skill_name for s in validated_resume.skills[:10]]
            print(f"    Sample: {', '.join(skill_names)}")
            if len(validated_resume.skills) > 10:
                print(f"    ... and {len(validated_resume.skills) - 10} more")
            
            print(f"\n  Certifications: {len(validated_resume.certifications)} certification(s)")
            print(f"  Languages: {len(validated_resume.languages)} language(s)")
            print(f"  Total Years of Experience: {validated_resume.total_years_of_experience}")
            
            print("\n" + "-" * 80)
            print("\n[SUCCESS] Resume data successfully extracted and validated!")
            print("\n[INFO] This structured data can now be used for:")
            print("  - Comparing against job descriptions")
            print("  - Generating tailored resume content")
            print("  - Identifying skill gaps and matches")
            
        else:
            print("\n[ERROR] No output received from agent.")
            
    except AttributeError as e:
        logger.error(f"Error accessing structured output: {e}", exc_info=True)
        print(f"\n[ERROR] Could not access structured output: {e}")
        print("\n[DEBUG] This might indicate the task didn't use output_pydantic parameter")
    except Exception as e:
        logger.error(f"Unexpected error processing output: {e}", exc_info=True)
        print(f"\n[ERROR] Unexpected error: {e}")
    
    print("\n" + "=" * 80)
    print("EXAMPLE COMPLETE")
    print("=" * 80)
    print("\nThis example demonstrated how the Resume Extractor Agent:")
    print("  1. Takes unstructured resume text as input")
    print("  2. Uses AI to intelligently extract information")
    print("  3. Structures data into validated JSON format")
    print("  4. Ensures data quality and completeness")
    print("\nNext steps: Use this extracted data with other agents for:")
    print("  - Job description analysis")
    print("  - Gap analysis and alignment strategy")
    print("  - Resume content optimization")


if __name__ == "__main__":
    main()
