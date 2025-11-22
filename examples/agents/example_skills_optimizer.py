"""
Example: Skills Section Optimizer Agent
=======================================

OBJECTIVE:
----------
This example demonstrates how the Skills Optimizer Agent infers missing skills
from work experience, validates them against evidence, prioritizes skills based
on job requirements, and categorizes them for optimal ATS performance.

WHAT THIS AGENT DOES:
---------------------
1. Receives:
   - Original skills from resume
   - Work experience entries
   - Job description requirements
   - Alignment strategy
2. Infers missing skills from experience descriptions:
   - Analyzes work experience to identify skills used but not listed
   - Validates each inference against evidence from experience
   - Assigns confidence scores based on evidence strength
3. Prioritizes all skills based on:
   - Job requirement importance (must-have, should-have, nice-to-have)
   - Relevance to job description
   - ATS keyword optimization
4. Categorizes skills into domain-appropriate categories
5. Returns optimized skills section with metadata

INPUT:
------
- Mock Experience data (structured from resume)
- Mock Original Skills (from resume)
- Mock Job Description (structured)
- Mock Alignment Strategy
  Note: In a real workflow, these would come from previous agents.

EXPECTED OUTPUT:
----------------
- A JSON object that matches the OptimizedSkillsSection model structure:
  {
    "optimization_notes": "...",
    "ats_match_score": 85,
    "added_skills": [
      {
        "skill_name": "AWS",
        "category": "Cloud Platforms",
        "confidence_score": 0.9,
        "justification": "...",
        "evidence": ["..."]
      }
    ],
    "optimized_skills": ["Python", "AWS", "Docker", ...],
    "skill_categories": {
      "Programming Languages": ["Python", "Go"],
      "Cloud Platforms": ["AWS", "Lambda"],
      ...
    }
  }

STEP-BY-STEP PROCESS:
---------------------
Step 1: Prepare mock structured data (Experience, Skills, JobDescription, Strategy)
Step 2: Create the Skills Optimizer Agent
Step 3: Call the optimize_skills_section function
Step 4: Process and validate the output
Step 5: Display the optimized skills section

WHY THIS MATTERS:
-----------------
The skills section is critical for ATS matching. An optimized skills section:
- Includes all relevant skills (even if not explicitly listed)
- Prioritizes skills that match job requirements
- Is properly categorized for readability
- Has high ATS keyword coverage
- Maintains truthfulness (only skills with evidence)
This significantly improves resume visibility in applicant tracking systems.
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agents.skills_optimizer_agent import (
    create_skills_optimizer_agent,
    validate_skills_output,
)
from src.core.logger import get_logger
from src.data_models.resume import Experience, Skill, OptimizedSkillsSection
from src.data_models.job import JobDescription, JobRequirement, SkillImportance
from src.data_models.strategy import AlignmentStrategy

logger = get_logger(__name__)


def print_section(title: str, content: str):
    """Helper to print formatted sections."""
    print(f"\n{'=' * 80}")
    print(f"{title}")
    print(f"{'=' * 80}")
    print(content)


def main():
    print("\n" + "=" * 80)
    print("SKILLS OPTIMIZER AGENT - EDUCATIONAL EXAMPLE")
    print("=" * 80)
    print("\nThis example shows how the AI infers, validates, prioritizes, and")
    print("categorizes skills based on experience and job requirements.\n")
    
    # ========================================================================
    # STEP 1: PREPARE INPUT DATA
    # ========================================================================
    print_section("STEP 1: INPUT DATA", "Setting up mock structured data...")
    
    # Mock Experience (Subset of what would be parsed from resume)
    experiences = [
        Experience(
            job_title="Senior Software Engineer",
            company_name="TechFlow Solutions",
            start_date="2018-01-01",
            end_date="2021-01-01",
            description="Developed microservices using Python and Flask. Deployed applications to AWS using Docker containers.",
            achievements=[
                "Reduced API latency by 40% through code optimization",
                "Migrated legacy monolith to microservices architecture",
                "Implemented CI/CD pipelines using GitHub Actions"
            ],
            skills_used=["Python", "Flask", "AWS", "Docker"]
        ),
        Experience(
            job_title="Software Developer",
            company_name="DataSystems Inc",
            start_date="2015-06-01",
            end_date="2017-12-31",
            description="Built data processing pipelines using SQL and Python.",
            achievements=[
                "Automated daily reporting saving 10 hours per week"
            ],
            skills_used=["Python", "SQL"]
        )
    ]
    
    # Mock Original Skills
    original_skills = [
        Skill(skill_name="Python", category="Programming"),
        Skill(skill_name="SQL", category="Database"),
        Skill(skill_name="Flask", category="Web Framework"),
        Skill(skill_name="Docker", category="DevOps")
    ]
    
    # Mock Job Description Requirements
    job_desc = JobDescription(
        job_title="Senior Backend Engineer",
        company_name="CloudCorp",
        summary="Seeking a skilled backend engineer to build and maintain our cloud infrastructure and microservices architecture.",
        full_text="We are looking for a Senior Backend Engineer to join our CloudCorp team. The ideal candidate will have strong experience with Python development, AWS cloud services, and microservices architecture. Key responsibilities include building scalable backend systems, implementing CI/CD pipelines, and working with containerization technologies like Docker and Kubernetes.",
        requirements=[
            JobRequirement(requirement="Python", importance=SkillImportance.MUST_HAVE),
            JobRequirement(requirement="AWS", importance=SkillImportance.MUST_HAVE),
            JobRequirement(requirement="Kubernetes", importance=SkillImportance.SHOULD_HAVE),
            JobRequirement(requirement="CI/CD", importance=SkillImportance.SHOULD_HAVE),
            JobRequirement(requirement="Microservices", importance=SkillImportance.MUST_HAVE),
            JobRequirement(requirement="Docker", importance=SkillImportance.MUST_HAVE),
            JobRequirement(requirement="Terraform", importance=SkillImportance.SHOULD_HAVE),
        ],
        ats_keywords=["Python", "AWS", "Kubernetes", "Docker", "CI/CD", "Microservices"]
    )
    
    # Mock Strategy
    strategy = AlignmentStrategy(
        overall_fit_score=0.8,
        summary_of_strategy="Focus on Cloud and Backend skills",
        identified_matches=[],
        identified_gaps=[],
        keywords_to_integrate=["AWS", "Microservices", "CI/CD", "Terraform"],
        professional_summary_guidance="",
        experience_guidance="",
        skills_guidance="Prioritize cloud technologies and microservices experience"
    )
    
    print("\n[INPUT] Mock Experience Data:")
    print("-" * 40)
    for exp in experiences:
        print(f"  • {exp.job_title} at {exp.company_name}")
        print(f"    Description: {exp.description}")
        print(f"    Skills mentioned: {', '.join(exp.skills_used)}")
    print("-" * 40)
    
    print("\n[INPUT] Original Skills:")
    print("-" * 40)
    for skill in original_skills:
        print(f"  • {skill.skill_name} ({skill.category})")
    print("-" * 40)
    
    print("\n[INPUT] Job Requirements:")
    print("-" * 40)
    for req in job_desc.requirements:
        print(f"  • {req.requirement} ({req.importance.value})")
    print("-" * 40)
    
    print("\n[INFO] In a real workflow, this data would come from:")
    print("  - Resume Extractor Agent (experiences, original skills)")
    print("  - Job Analyzer Agent (job description)")
    print("  - Gap Analysis Agent (strategy)")
    
    # ========================================================================
    # STEP 2: CREATE THE AGENT
    # ========================================================================
    print_section("STEP 2: AGENT CREATION", "Initializing the Skills Optimizer Agent...")
    
    try:
        agent = create_skills_optimizer_agent()
        print(f"\n[SUCCESS] Agent created successfully!")
        print(f"  Role: {agent.role}")
        print(f"  Goal: {agent.goal}")
        print("\n[INFO] This agent is specialized in:")
        print("  - Inferring missing skills from experience")
        print("  - Validating skills against evidence")
        print("  - Prioritizing skills by job requirements")
        print("  - Categorizing skills for ATS optimization")
    except Exception as e:
        logger.error(f"Failed to create agent: {e}", exc_info=True)
        print(f"\n[ERROR] Failed to create agent: {e}")
        return
    
    # ========================================================================
    # STEP 3: DEFINE THE TASK
    # ========================================================================
    print_section("STEP 3: TASK DEFINITION", "Setting up the optimization task...")
    
    # Load task configuration from tasks.yaml (same as real application)
    try:
        from src.core.config import get_tasks_config
        tasks_config = get_tasks_config()
        task_config = tasks_config.get("optimize_skills_section_task", {})
        
        if not task_config:
            raise ValueError("optimize_skills_section_task not found in tasks.yaml")
        
        # Get the base task description and expected_output from config
        base_description = task_config.get("description", "")
        base_expected_output = task_config.get("expected_output", "")
        
        print("\n[INFO] Loaded task configuration from tasks.yaml")
        print(f"  Task: optimize_skills_section_task")
        print(f"  Agent: {task_config.get('agent', 'N/A')}")
        
    except Exception as e:
        logger.error(f"Failed to load task config: {e}", exc_info=True)
        print(f"\n[ERROR] Failed to load task configuration: {e}")
        return
    
    # Adapt the task description for the example context
    # The real task expects data from previous agents, but we're providing mock data directly
    # 
    # IMPORTANT: We use output_pydantic=OptimizedSkillsSection to enforce structured output.
    # This is the industry-standard approach in CrewAI for ensuring LLM outputs
    # match our Pydantic model schema. It eliminates the need for manual JSON
    # parsing and validation.
    task_description = (
        f"IMPORTANT: The input data is provided below as structured JSON. "
        f"Work directly with this data to optimize the skills section.\n\n"
        f"ORIGINAL SKILLS:\n{json.dumps([s.model_dump(mode='json') for s in original_skills], indent=2)}\n\n"
        f"WORK EXPERIENCE:\n{json.dumps([e.model_dump(mode='json') for e in experiences], indent=2)}\n\n"
        f"JOB DESCRIPTION:\n{json.dumps(job_desc.model_dump(mode='json'), indent=2)}\n\n"
        f"ALIGNMENT STRATEGY:\n{json.dumps(strategy.model_dump(mode='json'), indent=2)}\n\n"
        f"{base_description}\n\n"
        f"CRITICAL INSTRUCTIONS:\n"
        f"- Analyze the candidate's experience to identify unstated but evident skills\n"
        f"- Validate each inferred skill against evidence from experience\n"
        f"- Prioritize skills based on job requirements (MUST_HAVE first)\n"
        f"- Categorize skills into domain-appropriate categories\n"
        f"- Calculate ATS match score based on keyword coverage\n"
        f"- Provide justification and evidence for all added skills\n"
        f"- Ensure output conforms exactly to the OptimizedSkillsSection model schema\n"
    )
    
    from crewai import Task
    task = Task(
        description=task_description,
        expected_output=base_expected_output,
        agent=agent,
        output_pydantic=OptimizedSkillsSection,  # ⭐ STRUCTURED OUTPUT ENFORCEMENT
    )
    
    print("\n[INFO] Task configured with:")
    print("  - Real task description from tasks.yaml")
    print("  - Real expected_output from tasks.yaml")
    print("  - Adapted for example context (structured data provided directly)")
    print("  - Schema requirements to match OptimizedSkillsSection model")
    
    # ========================================================================
    # STEP 4: EXECUTE THE AGENT
    # ========================================================================
    print_section("STEP 4: AGENT EXECUTION", "Running the agent (this calls the LLM)...")
    
    from crewai import Crew, Process
    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )
    
    print("\n[INFO] The agent will now:")
    print("  1. Infer missing skills from experience descriptions")
    print("  2. Validate each inference against evidence")
    print("  3. Prioritize all skills based on job requirements")
    print("  4. Categorize skills into domain-appropriate categories")
    print("  5. Calculate ATS match score")
    print("\n[WAIT] This may take 60-120 seconds as the LLM processes each step...\n")
    
    try:
        result = crew.kickoff()
    except Exception as e:
        logger.error(f"Error during agent execution: {e}", exc_info=True)
        print(f"\n[ERROR] Agent execution failed: {e}")
        return
    
    # ========================================================================
    # STEP 5: ACCESS STRUCTURED OUTPUT
    # ========================================================================
    print_section("STEP 5: ACCESSING STRUCTURED OUTPUT", "Retrieving validated OptimizedSkillsSection object...")
    
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
        # Validate against Pydantic model
        print("\n[INFO] Validating against OptimizedSkillsSection Pydantic model...")
        validated_section = validate_skills_output(optimized_section.model_dump(mode='json'))
        
        if validated_section:
            print("\n" + "=" * 80)
            print("SKILLS OPTIMIZATION SUCCESSFUL!")
            print("=" * 80)
            print("\n[VALIDATED] Output matches OptimizedSkillsSection model structure")
            
            # Display key results
            print("\n" + "-" * 80)
            print("OPTIMIZED SKILLS SECTION")
            print("-" * 80)
            
            print(f"\n  ATS Match Score: {optimized_section.ats_match_score}/100")
            if optimized_section.ats_match_score >= 80:
                print("    Status: Excellent ATS Match ✓")
            elif optimized_section.ats_match_score >= 60:
                print("    Status: Good ATS Match")
            else:
                print("    Status: Needs Improvement")
            
            print(f"\n  Optimization Notes:")
            notes = optimized_section.optimization_notes or "Not provided"
            print(f"    {notes}")
            
            print(f"\n  Original Skills Count: {len(original_skills)}")
            print(f"  Optimized Skills Count: {len(optimized_section.optimized_skills)}")
            print(f"  Skills Added: {len(optimized_section.added_skills)}")
            
            if optimized_section.added_skills:
                print(f"\n  INFERRED SKILLS (Added by AI):")
                print("-" * 40)
                for skill in optimized_section.added_skills:
                    print(f"\n    + {skill.skill_name}")
                    print(f"      Category: {skill.category}")
                    print(f"      Confidence: {skill.confidence_score:.2f}/1.0")
                    if skill.justification:
                        print(f"      Justification: {skill.justification}")
                    if skill.evidence:
                        print(f"      Evidence: {len(skill.evidence)} snippet(s)")
                        for i, evidence in enumerate(skill.evidence, 1):
                            print(f"        {i}. {evidence}")
            else:
                print("\n  No skills inferred (all skills were already present)")
            
            print(f"\n  FINAL CATEGORIZED SKILLS:")
            print("-" * 40)
            if optimized_section.skill_categories:
                for category, skills in optimized_section.skill_categories.items():
                    print(f"\n    [{category}]")
                    for skill_name in skills:
                        # Check if this is an added skill
                        is_added = any(s.skill_name == skill_name for s in optimized_section.added_skills)
                        marker = " ⭐ (inferred)" if is_added else ""
                        print(f"      • {skill_name}{marker}")
            else:
                print("    No categories assigned")
            
            print(f"\n  PRIORITIZED SKILLS LIST:")
            print("-" * 40)
            # Extract skill names from Skill objects
            skill_names = [skill.skill_name for skill in optimized_section.optimized_skills]
            print(f"    {', '.join(skill_names)}")
            
            print("\n" + "-" * 80)
            print("\n[SUCCESS] Skills section optimized and validated!")
            print("\n[INFO] The optimized skills section:")
            print("  - Includes inferred skills with evidence")
            print("  - Prioritizes job-relevant skills")
            print("  - Is properly categorized for ATS")
            print("  - Maintains truthfulness (only skills with evidence)")
            print("  - Has high ATS keyword coverage")
            
        else:
            print("\n[ERROR] Validation failed. Output does not match OptimizedSkillsSection model.")
            print("\n[DEBUG] Full output:")
            print(json.dumps(optimized_section.model_dump(mode='json'), indent=2, ensure_ascii=False))
            
    except Exception as e:
        logger.error(f"Error processing output: {e}", exc_info=True)
        print(f"\n[ERROR] Could not process output: {e}")
    
    print("\n" + "=" * 80)
    print("EXAMPLE COMPLETE")
    print("=" * 80)
    print("\nThis example demonstrated how the Skills Optimizer Agent:")
    print("  1. Takes experience, skills, job description, and strategy as input")
    print("  2. Infers missing skills from experience descriptions")
    print("  3. Validates inferences against evidence")
    print("  4. Prioritizes skills based on job requirements")
    print("  5. Categorizes skills for optimal ATS performance")
    print("\nNext steps: Use this optimized skills section with:")
    print("  - Final ATS optimization and resume assembly")


if __name__ == "__main__":
    main()
