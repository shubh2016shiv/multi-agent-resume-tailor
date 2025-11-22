"""
Example: Experience Section Optimizer Agent
===========================================

OBJECTIVE:
----------
This example demonstrates how the Experience Optimizer Agent rewrites work
experience entries to align with job requirements, quantify achievements,
and integrate keywords naturally.

WHAT THIS AGENT DOES:
---------------------
1. Receives:
   - Structured Resume data (with work experience)
   - Structured JobDescription data
   - AlignmentStrategy with guidance
2. For each work experience entry:
   - Rewrites achievement bullets to be more impactful
   - Quantifies results (adds metrics if missing)
   - Integrates relevant keywords naturally
   - Emphasizes skills and achievements that match job requirements
   - Uses strong action verbs
3. Ensures consistency and professional tone
4. Returns optimized experience entries with metadata

INPUT:
------
- Sample resume markdown (from common.py)
- Sample job description markdown (from common.py)
- Mock Alignment Strategy (simplified for example)
  In a real workflow, this would come from the Gap Analysis Agent.

EXPECTED OUTPUT:
----------------
- A JSON object that matches the OptimizedExperienceSection model structure:
  {
    "optimization_notes": "...",
    "keywords_integrated": ["Python", "AWS", "Microservices", ...],
    "optimized_experiences": [
      {
        "job_title": "Software Engineer",
        "company_name": "Tech Corp",
        "start_date": "2020-01-01",
        "end_date": null,
        "is_current_position": true,
        "achievements": [
          "Architected and deployed microservices on AWS, reducing latency by 40%",
          "Optimized database queries, improving response time by 30%",
          ...
        ],
        "skills_used": ["Python", "AWS", "Docker", ...]
      },
      ...
    ]
  }

STEP-BY-STEP PROCESS:
---------------------
Step 1: Load sample resume, job description, and mock strategy
Step 2: Create the Experience Optimizer Agent
Step 3: Define the optimization task with all inputs
Step 4: Execute the agent (calls LLM to optimize experience entries)
Step 5: Parse and validate the output
Step 6: Display the optimized experience entries

WHY THIS MATTERS:
-----------------
The experience section is the core of a resume. Optimized experience bullets:
- Use strong action verbs (Architected, Deployed, Optimized)
- Quantify achievements with metrics
- Integrate ATS keywords naturally
- Highlight relevant skills and accomplishments
- Show impact and value delivered
This significantly improves resume visibility and demonstrates value to employers.
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from crewai import Crew, Process, Task
from src.agents.experience_optimizer_agent import (
    create_experience_optimizer_agent,
    validate_experience_output,
    OptimizedExperienceSection,
)
from src.core.logger import get_logger
from src.data_models.strategy import AlignmentStrategy
from examples.agents.common import get_resume_md, get_job_desc_md, parse_json_output

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
    print("EXPERIENCE OPTIMIZER AGENT - EDUCATIONAL EXAMPLE")
    print("=" * 80)
    print("\nThis example shows how the AI optimizes work experience entries to")
    print("align with job requirements, quantify achievements, and integrate keywords.\n")
    
    # ========================================================================
    # STEP 1: PREPARE INPUT DATA
    # ========================================================================
    print_section("STEP 1: INPUT DATA", "Loading sample resume, job description, and strategy...")
    
    resume_md = get_resume_md()
    job_md = get_job_desc_md()
    
    # Create proper AlignmentStrategy object (required for agent's tool)
    strategy = AlignmentStrategy(
        overall_fit_score=0.8,
        summary_of_strategy="Focus on backend scalability and cloud infrastructure. Emphasize Python and AWS experience.",
        identified_matches=[],
        identified_gaps=[],
        keywords_to_integrate=["Python", "AWS", "Lambda", "DynamoDB", "CI/CD", "Terraform", "Microservices"],
        professional_summary_guidance="Emphasize backend development and cloud experience",
        experience_guidance=(
            "Rewrite bullets to emphasize backend scalability and cloud infrastructure. "
            "Quantify results where possible (e.g., 'reduced latency by 20%'). "
            "Use strong action verbs like 'Architected', 'Deployed', 'Optimized'. "
            "Highlight microservices experience and AWS services usage."
        ),
        skills_guidance="Prioritize cloud technologies and backend frameworks"
    )
    
    print("\n[INPUT] Sample Resume:")
    print("-" * 40)
    print(resume_md[:200] + "..." if len(resume_md) > 200 else resume_md)
    print("-" * 40)
    
    print("\n[INPUT] Sample Job Description:")
    print("-" * 40)
    print(job_md[:200] + "..." if len(job_md) > 200 else job_md)
    print("-" * 40)
    
    print("\n[INPUT] Alignment Strategy:")
    print("-" * 40)
    print(f"  Overall Fit Score: {strategy.overall_fit_score}")
    print(f"  Summary: {strategy.summary_of_strategy}")
    print(f"  Keywords to Integrate: {', '.join(strategy.keywords_to_integrate)}")
    print(f"  Experience Guidance: {strategy.experience_guidance[:100]}...")
    print("-" * 40)
    
    print("\n[INFO] In a real workflow, the strategy would be the structured output")
    print("       from the Gap Analysis Agent.")
    
    # ========================================================================
    # STEP 2: CREATE THE AGENT
    # ========================================================================
    print_section("STEP 2: AGENT CREATION", "Initializing the Experience Optimizer Agent...")
    
    try:
        agent = create_experience_optimizer_agent()
        print(f"\n[SUCCESS] Agent created successfully!")
        print(f"  Role: {agent.role}")
        print(f"  Goal: {agent.goal[:100]}..." if len(agent.goal) > 100 else f"  Goal: {agent.goal}")
        print("\n[INFO] This agent is specialized in:")
        print("  - Rewriting experience bullets for impact")
        print("  - Quantifying achievements with metrics")
        print("  - Integrating ATS keywords naturally")
        print("  - Using strong action verbs")
        print("  - Aligning content with job requirements")
    except Exception as e:
        logger.error(f"Failed to create agent: {e}", exc_info=True)
        print(f"\n[ERROR] Failed to create agent: {e}")
        return
    
    # ========================================================================
    # STEP 3: DEFINE THE TASK
    # ========================================================================
    print_section("STEP 3: TASK DEFINITION", "Setting up the experience optimization task...")
    
    # Load task configuration from tasks.yaml (same as real application)
    try:
        from src.core.config import get_tasks_config
        tasks_config = get_tasks_config()
        task_config = tasks_config.get("optimize_experience_section_task", {})
        
        if not task_config:
            raise ValueError("optimize_experience_section_task not found in tasks.yaml")
        
        # Get the base task description and expected_output from config
        base_description = task_config.get("description", "")
        base_expected_output = task_config.get("expected_output", "")
        
        print("\n[INFO] Loaded task configuration from tasks.yaml")
        print(f"  Task: optimize_experience_section_task")
        print(f"  Agent: {task_config.get('agent', 'N/A')}")
        
    except Exception as e:
        logger.error(f"Failed to load task config: {e}", exc_info=True)
        print(f"\n[ERROR] Failed to load task configuration: {e}")
        return
    
    # Adapt the task description for the example context
    # The real task expects structured data from previous agents, but we're providing markdown directly
    # 
    # IMPORTANT: We use output_pydantic=OptimizedExperienceSection to enforce structured output.
    # This is the industry-standard approach in CrewAI for ensuring LLM outputs
    # match our Pydantic model schema. It eliminates the need for manual JSON
    # parsing and validation.
    task_description = (
        f"IMPORTANT: The input data is provided below as Markdown text. "
        f"Work directly with this data to optimize the experience section.\n\n"
        f"RESUME:\n{resume_md}\n\n"
        f"JOB DESCRIPTION:\n{job_md}\n\n"
        f"ALIGNMENT STRATEGY:\n{json.dumps(strategy.model_dump(mode='json'), indent=2)}\n\n"
        f"{base_description}\n\n"
        f"CRITICAL INSTRUCTIONS:\n"
        f"- Extract work experience entries from the resume markdown\n"
        f"- Use the alignment strategy guidance to optimize each entry\n"
        f"- Call the evaluate_experience_bullets tool for iterative improvement\n"
        f"- Ensure output conforms exactly to the OptimizedExperienceSection model schema\n"
    )
    
    task = Task(
        description=task_description,
        expected_output=base_expected_output,
        agent=agent,
        output_pydantic=OptimizedExperienceSection,  # ⭐ STRUCTURED OUTPUT ENFORCEMENT
    )
    
    print("\n[INFO] Task configured with:")
    print("  - Real task description from tasks.yaml")
    print("  - Real expected_output from tasks.yaml")
    print("  - Adapted for example context (markdown content provided directly)")
    print("  - Schema requirements to match OptimizedExperienceSection model")
    
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
    print("  1. Analyze each work experience entry")
    print("  2. Rewrite achievement bullets for maximum impact")
    print("  3. Add quantified metrics where appropriate")
    print("  4. Integrate keywords naturally")
    print("  5. Ensure alignment with job requirements")
    print("\n[WAIT] This may take 60-120 seconds as the LLM optimizes each entry...\n")
    
    try:
        result = crew.kickoff()
    except Exception as e:
        logger.error(f"Error during agent execution: {e}", exc_info=True)
        print(f"\n[ERROR] Agent execution failed: {e}")
        return
    
    # ========================================================================
    # STEP 5: ACCESS STRUCTURED OUTPUT
    # ========================================================================
    print_section("STEP 5: ACCESSING STRUCTURED OUTPUT", "Retrieving validated OptimizedExperienceSection object...")
    
    print("\n[INFO] With output_pydantic, the result is automatically validated!")
    print("  - No manual JSON parsing needed")
    print("  - No manual validation needed")
    print("  - Direct access to typed Pydantic object")
    
    try:
        # Access the validated Pydantic object directly
        optimized_section = result.pydantic
        
        if not optimized_section:
            print("\n[ERROR] No output received from agent.")
            return
            
    except AttributeError as e:
        logger.error(f"Error accessing structured output: {e}", exc_info=True)
        print(f"\n[ERROR] Could not access structured output: {e}")
        print("\n[DEBUG] This might indicate the task didn't use output_pydantic parameter")
        return
    except Exception as e:
        logger.error(f"Unexpected error processing output: {e}", exc_info=True)
        print(f"\n[ERROR] Unexpected error: {e}")
        return
    
    # ========================================================================
    # STEP 6: DISPLAY AND VALIDATE RESULTS
    # ========================================================================
    print_section("STEP 6: OUTPUT PROCESSING", "Validating and displaying results...")
    
    try:
        # Validate against Pydantic model (additional validation for robustness)
        print("\n[INFO] Validating against OptimizedExperienceSection Pydantic model...")
        validated_section = validate_experience_output(optimized_section.model_dump(mode='json'))
        
        if not validated_section:
            # Fallback: use the pydantic result directly if validation fails
            validated_section = optimized_section
        
        if validated_section:
            print("\n" + "=" * 80)
            print("EXPERIENCE OPTIMIZATION SUCCESSFUL!")
            print("=" * 80)
            print("\n[VALIDATED] Output automatically validated by CrewAI")
            
            # Display key results
            print("\n" + "-" * 80)
            print("OPTIMIZED EXPERIENCE SECTION")
            print("-" * 80)
            
            print(f"\n  Optimization Notes:")
            notes = validated_section.optimization_notes or "Not provided"
            print(f"    {notes[:300]}..." if len(notes) > 300 else f"    {notes}")
            
            print(f"\n  Keywords Integrated: {len(validated_section.keywords_integrated)} keyword(s)")
            if validated_section.keywords_integrated:
                print(f"    {', '.join(validated_section.keywords_integrated[:10])}")
                if len(validated_section.keywords_integrated) > 10:
                    print(f"    ... and {len(validated_section.keywords_integrated) - 10} more")
            
            print(f"\n  Optimized Experiences: {len(validated_section.optimized_experiences)} role(s)")
            print("\n" + "-" * 40)
            
            for i, exp in enumerate(validated_section.optimized_experiences, 1):
                print(f"\n  EXPERIENCE {i}: {exp.job_title} at {exp.company_name}")
                print(f"    Period: {exp.start_date} to {exp.end_date or 'Present'}")
                if exp.is_current_position:
                    print("    Status: Current Position")
                
                print(f"\n    Optimized Achievements ({len(exp.achievements)} bullets):")
                for j, bullet in enumerate(exp.achievements, 1):
                    print(f"      {j}. {bullet}")
                
                if exp.skills_used:
                    print(f"\n    Skills Used: {', '.join(exp.skills_used[:10])}")
                    if len(exp.skills_used) > 10:
                        print(f"      ... and {len(exp.skills_used) - 10} more")
                
                print("-" * 40)
            
            print("\n" + "-" * 80)
            print("\n[SUCCESS] Experience section optimized and validated!")
            print("\n[INFO] The optimized experiences:")
            print("  - Use strong action verbs")
            print("  - Include quantified achievements")
            print("  - Integrate ATS keywords naturally")
            print("  - Align with job requirements")
            print("  - Show clear impact and value")
            
        else:
            print("\n[ERROR] Validation failed. Output does not match OptimizedExperienceSection model.")
            print("\n[DEBUG] Full output:")
            print(json.dumps(optimized_section.model_dump(mode='json'), indent=2, ensure_ascii=False))
            
    except Exception as e:
        logger.error(f"Error processing output: {e}", exc_info=True)
        print(f"\n[ERROR] Could not process output: {e}")
    
    print("\n" + "=" * 80)
    print("EXAMPLE COMPLETE")
    print("=" * 80)
    print("\nThis example demonstrated how the Experience Optimizer Agent:")
    print("  1. Takes resume, job description, and strategy as input")
    print("  2. Rewrites experience bullets for maximum impact")
    print("  3. Quantifies achievements with metrics")
    print("  4. Integrates ATS keywords naturally")
    print("  5. Aligns content with job requirements")
    print("\nNext steps: Use this optimized experience with other agents for:")
    print("  - Skills section optimization")
    print("  - Final ATS optimization and resume assembly")


if __name__ == "__main__":
    main()
