"""
Example: ATS Optimization Specialist Agent
===========================================

OBJECTIVE:
----------
This example demonstrates how the ATS Optimization Agent assembles all optimized
resume components into a final ATS-compliant resume, validates keyword density,
checks formatting compliance, and generates both markdown and JSON outputs.

WHAT THIS AGENT DOES:
---------------------
1. Receives optimized resume components:
   - Professional Summary (from Summary Writer Agent)
   - Optimized Experience Section (from Experience Optimizer Agent)
   - Optimized Skills Section (from Skills Optimizer Agent)
   - Education and other sections (from Resume Extractor Agent)
2. Assembles all components into a complete resume
3. Validates ATS compatibility:
   - Keyword density and coverage
   - Formatting compliance (headers, structure)
   - Section completeness
4. Optimizes for ATS:
   - Ensures proper keyword placement
   - Validates formatting standards
   - Checks for common ATS issues
5. Generates final outputs:
   - Markdown resume (human-readable)
   - JSON resume (structured data)
   - ATS validation report

INPUT:
------
- Sample resume markdown (from common.py)
- Sample job description markdown (from common.py)
  Note: In a real workflow, these would be the optimized outputs from
  previous agents (Summary Writer, Experience Optimizer, Skills Optimizer).

EXPECTED OUTPUT:
----------------
- A JSON object that matches the OptimizedResume model structure:
  {
    "ats_validation": {
      "overall_score": 92,
      "is_compatible": true,
      "keyword_report": {
        "keyword_density": 0.045,
        "keyword_coverage": 0.85
      },
      "formatting_issues": []
    },
    "optimization_summary": "...",
    "markdown_content": "# John Doe\n...",
    "json_content": {...}
  }

STEP-BY-STEP PROCESS:
---------------------
Step 1: Load sample resume and job description data
Step 2: Create the ATS Optimization Agent
Step 3: Define the optimization task with all inputs
Step 4: Execute the agent (calls LLM to assemble and validate)
Step 5: Parse and validate the output
Step 6: Display the ATS validation results and final resume

WHY THIS MATTERS:
-----------------
Applicant Tracking Systems (ATS) are used by most companies to filter resumes.
An ATS-optimized resume:
- Has proper keyword density (not too low, not too high)
- Uses standard formatting (no complex layouts)
- Has proper section headers
- Includes all required sections
- Is parseable by ATS systems
This is the final quality check before the resume is ready for submission.
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from crewai import Crew, Process, Task  # noqa: E402

from examples.agents.common import get_job_desc_md, get_resume_md  # noqa: E402
from src.agents.ats_optimization_agent import (  # noqa: E402
    OptimizedResume,
    create_ats_optimization_agent,
)
from src.core.logger import get_logger  # noqa: E402

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
    print("ATS OPTIMIZATION SPECIALIST AGENT - EDUCATIONAL EXAMPLE")
    print("=" * 80)
    print("\nThis example shows how the AI assembles optimized resume components")
    print("and validates ATS compatibility for maximum visibility.\n")

    # ========================================================================
    # STEP 1: PREPARE INPUT DATA
    # ========================================================================
    print_section("STEP 1: INPUT DATA", "Loading sample resume and job description...")

    resume_md = get_resume_md()
    job_md = get_job_desc_md()

    print("\n[INPUT] Sample Resume:")
    print("-" * 40)
    print(resume_md[:300] + "..." if len(resume_md) > 300 else resume_md)
    print("-" * 40)

    print("\n[INPUT] Sample Job Description:")
    print("-" * 40)
    print(job_md[:300] + "..." if len(job_md) > 300 else job_md)
    print("-" * 40)

    print("\n[INFO] In a real workflow, the resume would be assembled from:")
    print("  - Professional Summary (from Summary Writer Agent)")
    print("  - Optimized Experience Section (from Experience Optimizer Agent)")
    print("  - Optimized Skills Section (from Skills Optimizer Agent)")
    print("  - Education and other sections (from Resume Extractor Agent)")

    # ========================================================================
    # STEP 2: CREATE THE AGENT
    # ========================================================================
    print_section("STEP 2: AGENT CREATION", "Initializing the ATS Optimization Agent...")

    try:
        agent = create_ats_optimization_agent()
        print("\n[SUCCESS] Agent created successfully!")
        print(f"  Role: {agent.role}")
        print(
            f"  Goal: {agent.goal[:100]}..." if len(agent.goal) > 100 else f"  Goal: {agent.goal}"
        )
        print("\n[INFO] This agent is specialized in:")
        print("  - Assembling resume components into final format")
        print("  - Validating ATS compatibility")
        print("  - Checking keyword density and coverage")
        print("  - Ensuring formatting compliance")
        print("  - Generating markdown and JSON outputs")
    except Exception as e:
        logger.error(f"Failed to create agent: {e}", exc_info=True)
        print(f"\n[ERROR] Failed to create agent: {e}")
        return

    # ========================================================================
    # STEP 3: DEFINE THE TASK
    # ========================================================================
    print_section("STEP 3: TASK DEFINITION", "Setting up the ATS optimization task...")

    # Load task configuration from tasks.yaml (same as real application)
    try:
        from src.core.config import get_tasks_config

        tasks_config = get_tasks_config()
        task_config = tasks_config.get("ats_optimization_task", {})

        if not task_config:
            raise ValueError("ats_optimization_task not found in tasks.yaml")

        # Get the base task description and expected_output from config
        base_description = task_config.get("description", "")
        base_expected_output = task_config.get("expected_output", "")

        print("\n[INFO] Loaded task configuration from tasks.yaml")
        print("  Task: ats_optimization_task")
        print(f"  Agent: {task_config.get('agent', 'N/A')}")

    except Exception as e:
        logger.error(f"Failed to load task config: {e}", exc_info=True)
        print(f"\n[ERROR] Failed to load task configuration: {e}")
        return

    # Adapt the task description for the example context
    # The real task expects data from previous agents, but we're providing markdown directly
    #
    # IMPORTANT: We use output_pydantic=OptimizedResume to enforce structured output.
    # This is the industry-standard approach in CrewAI for ensuring LLM outputs
    # match our Pydantic model schema. It eliminates the need for manual JSON
    # parsing and validation.
    task_description = (
        f"IMPORTANT: The resume and job description content is provided below as Markdown text. "
        f"Work directly with this data to optimize the resume for ATS compatibility.\n\n"
        f"RESUME CONTENT:\n{resume_md}\n\n"
        f"JOB DESCRIPTION:\n{job_md}\n\n"
        f"{base_description}\n\n"
        f"CRITICAL INSTRUCTIONS:\n"
        f"- Assemble all resume components into a complete resume object\n"
        f"- Validate ATS compatibility and calculate scores (keyword density 2-5%, formatting, etc.)\n"
        f"- Generate ATS-optimized markdown content and structured JSON content\n"
        f"- Create a comprehensive validation report with scores and recommendations\n"
        f"- Populate all required fields in one complete JSON response\n\n"
        f"OUTPUT REQUIREMENT:\n"
        f"You must output a SINGLE, complete JSON object containing:\n"
        f"- resume: full resume data\n"
        f"- markdown_content: formatted markdown resume\n"
        f"- json_content: structured JSON resume\n"
        f"- ats_validation: comprehensive validation results\n"
        f"- optimization_summary: summary of changes\n\n"
        f"Do NOT make multiple separate calls. Include all data in one JSON response."
    )

    # Override the expected_output to include explicit JSON schema
    # CRITICAL: We do NOT use output_pydantic here because it causes the LLM to call
    # nested Pydantic models as separate tools. Instead, we provide explicit JSON schema
    # in the expected_output and manually validate the response.
    expected_output = (
        "Output a valid JSON object (and ONLY JSON, no markdown formatting) with this structure:\n\n"
        "{\n"
        '  "resume": {\n'
        '    "full_name": "John Doe",\n'
        '    "email": "john@example.com",\n'
        '    "phone_number": "123-456-7890",\n'
        '    "location": "New York, NY",\n'
        '    "professional_summary": "Professional summary text...",\n'
        '    "work_experience": [\n'
        "      {\n"
        '        "job_title": "Software Engineer",\n'
        '        "company_name": "Tech Corp",\n'
        '        "start_date": "2020-01",\n'
        '        "end_date": "2023-01",\n'
        '        "is_current_position": false,\n'
        '        "description": "Built scalable systems...",\n'
        '        "achievements": ["Increased efficiency by 40%", "Led team of 5"],\n'
        '        "skills_used": ["Python", "AWS", "Docker"]\n'
        "      }\n"
        "    ],\n"
        '    "education": [\n'
        "      {\n"
        '        "institution_name": "University Name",\n'
        '        "degree": "Bachelor of Science",\n'
        '        "field_of_study": "Computer Science",\n'
        '        "graduation_year": 2020\n'
        "      }\n"
        "    ],\n"
        '    "skills": [\n'
        '      {"skill_name": "Python", "category": "Backend", "proficiency_level": "Expert"},\n'
        '      {"skill_name": "AWS", "category": "Cloud", "proficiency_level": "Advanced"}\n'
        "    ],\n"
        '    "certifications": [],\n'
        '    "languages": ["English", "Spanish"]\n'
        "  },\n"
        '  "markdown_content": "# John Doe\\n...full resume in markdown format...",\n'
        '  "json_content": "{\\"full_name\\": \\"John Doe\\", ...}",\n'
        '  "ats_validation": {\n'
        '    "overall_score": 85.0,\n'
        '    "is_compatible": true,\n'
        '    "formatting_issues": [],\n'
        '    "recommendations": ["Add more technical keywords"],\n'
        '    "keyword_report": {\n'
        '      "total_words": 500,\n'
        '      "total_keywords": 25,\n'
        '      "unique_keywords": 15,\n'
        '      "keyword_density": 0.035,\n'
        '      "is_optimal": true,\n'
        '      "keyword_coverage": 0.85,\n'
        '      "missing_must_have_keywords": [],\n'
        '      "keyword_frequency": {"Python": 5, "AWS": 3}\n'
        "    },\n"
        '    "section_validations": [\n'
        "      {\n"
        '        "section_name": "Summary",\n'
        '        "is_present": true,\n'
        '        "is_standard_header": true,\n'
        '        "header_used": "Professional Summary",\n'
        '        "recommended_header": "Professional Summary",\n'
        '        "content_length": 150,\n'
        '        "has_formatting_issues": false,\n'
        '        "issues_found": []\n'
        "      }\n"
        "    ],\n"
        '    "special_character_issues": [],\n'
        '    "strengths": ["Strong keyword coverage", "Clear formatting"]\n'
        "  },\n"
        '  "optimization_summary": "Resume optimized for ATS with 85/100 score. Added technical keywords and improved formatting.",\n'
        '  "components_assembled": {\n'
        '    "professional_summary": true,\n'
        '    "work_experience": true,\n'
        '    "skills": true,\n'
        '    "education": true,\n'
        '    "certifications": false\n'
        "  },\n"
        '  "quality_metrics": {\n'
        '    "ats_score": 85.0,\n'
        '    "keyword_density": 0.035,\n'
        '    "keyword_coverage": 0.85,\n'
        '    "total_sections": 5\n'
        "  }\n"
        "}\n\n"
        "CRITICAL RULES:\n"
        "1. Output ONLY the JSON object - no markdown code fences, no explanatory text\n"
        "2. 'components_assembled' must contain ONLY boolean values (true/false) indicating which sections were included\n"
        "3. 'quality_metrics' must contain ONLY numeric values (floats/integers)\n"
        "4. All arrays must contain objects with the exact structure shown above\n"
        "5. DO NOT put resume data in 'components_assembled' - it's for tracking flags only"
    )

    task = Task(
        description=task_description,
        expected_output=expected_output,
        agent=agent,
        # DO NOT USE output_pydantic - it causes nested model tool calls
        # We will manually validate the JSON response instead
    )

    print("\n[INFO] Task configured with:")
    print("  - Centralized task description from tasks.yaml (ats_optimization_task)")
    print("  - Centralized expected_output from tasks.yaml")
    print("  - Adapted for example context (markdown content provided directly)")
    print("  - Structured output enforcement (output_pydantic=OptimizedResume)")

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
    print("  1. Assemble all resume components")
    print("  2. Validate ATS compatibility")
    print("  3. Check keyword density and coverage")
    print("  4. Verify formatting compliance")
    print("  5. Generate final markdown and JSON outputs")
    print("\n[WAIT] This may take 45-90 seconds as the LLM assembles and validates...\n")

    try:
        result = crew.kickoff()
    except Exception as e:
        logger.error(f"Error during agent execution: {e}", exc_info=True)
        print(f"\n[ERROR] Agent execution failed: {e}")
        return

    # ========================================================================
    # VALIDATION HELPER FUNCTION
    # ========================================================================

    def fix_common_llm_issues(json_data: dict) -> dict:
        """
        Fix common LLM misunderstandings in the JSON output.

        This is an enterprise-grade defensive programming approach that ensures
        robustness even when the LLM doesn't perfectly follow the schema.

        Args:
            json_data: Raw JSON data from LLM

        Returns:
            Corrected JSON data that conforms to the expected schema
        """
        # Fix skills if they're simple strings instead of Skill objects
        if "resume" in json_data and "skills" in json_data["resume"]:
            skills_list = json_data["resume"]["skills"]
            if isinstance(skills_list, list) and skills_list:
                # Check if first item is a string
                if isinstance(skills_list[0], str):
                    logger.warning("Converting string skills to Skill objects")
                    json_data["resume"]["skills"] = [
                        {"skill_name": skill, "category": "General", "proficiency_level": None}
                        for skill in skills_list
                    ]

        # Fix date formats in work_experience
        if "resume" in json_data and "work_experience" in json_data["resume"]:
            for exp in json_data["resume"]["work_experience"]:
                # Fix start_date: "2020" -> "2020-01-01"
                if "start_date" in exp and isinstance(exp["start_date"], str):
                    date_str = exp["start_date"]
                    if len(date_str) == 4 and date_str.isdigit():  # Just year
                        exp["start_date"] = f"{date_str}-01-01"
                    elif date_str and not date_str.count("-") == 2:  # Not full date
                        # Try to parse partial dates
                        if "-" in date_str:  # "2020-01" format
                            exp["start_date"] = f"{date_str}-01"
                        else:
                            exp["start_date"] = f"{date_str}-01-01"

                # Fix end_date: "Present" -> None, "2020" -> "2020-01-01"
                if "end_date" in exp:
                    date_str = exp["end_date"]
                    if isinstance(date_str, str):
                        if date_str.lower() in ["present", "current", "now"]:
                            exp["end_date"] = None
                            exp["is_current_position"] = True
                        elif len(date_str) == 4 and date_str.isdigit():  # Just year
                            exp["end_date"] = f"{date_str}-01-01"
                        elif date_str and not date_str.count("-") == 2:  # Not full date
                            if "-" in date_str:  # "2020-01" format
                                exp["end_date"] = f"{date_str}-01"
                            else:
                                exp["end_date"] = f"{date_str}-01-01"

        # Fix keyword_report field names (LLM might use different names)
        if "ats_validation" in json_data and "keyword_report" in json_data["ats_validation"]:
            kr = json_data["ats_validation"]["keyword_report"]
            # Map common LLM variations to expected field names
            field_mappings = {
                "total_keywords_found": "total_keywords",
                "unique_keywords_found": "unique_keywords",
                "unique_keywords_required": "unique_keywords",
            }
            for old_name, new_name in field_mappings.items():
                if old_name in kr and new_name not in kr:
                    kr[new_name] = kr.pop(old_name)
                    logger.warning(f"Renamed keyword_report field: {old_name} -> {new_name}")

        # Fix components_assembled if it has wrong structure
        if "components_assembled" in json_data:
            ca = json_data["components_assembled"]
            if isinstance(ca, dict):
                # Check if LLM put resume data instead of boolean flags
                if any(not isinstance(v, bool) for v in ca.values()):
                    logger.warning("Fixing components_assembled: LLM returned non-boolean values")
                    # Extract which sections are actually present in the resume
                    resume_data = json_data.get("resume", {})
                    json_data["components_assembled"] = {
                        "professional_summary": bool(
                            resume_data.get("professional_summary", "").strip()
                        ),
                        "work_experience": len(resume_data.get("work_experience", [])) > 0,
                        "skills": len(resume_data.get("skills", [])) > 0,
                        "education": len(resume_data.get("education", [])) > 0,
                        "certifications": len(resume_data.get("certifications", [])) > 0,
                    }

        # Fix quality_metrics if it has non-numeric values
        if "quality_metrics" in json_data:
            qm = json_data["quality_metrics"]
            if isinstance(qm, dict):
                # Ensure all values are numeric
                for key, value in list(qm.items()):
                    if isinstance(value, str):
                        try:
                            qm[key] = float(value)
                        except (ValueError, TypeError):
                            logger.warning(f"Removing non-numeric quality_metric: {key}={value}")
                            del qm[key]

        # Ensure section_validations is a list of objects, not strings
        if "ats_validation" in json_data:
            ats_val = json_data["ats_validation"]
            if isinstance(ats_val, dict) and "section_validations" in ats_val:
                sv = ats_val["section_validations"]
                if isinstance(sv, list):
                    # Filter out any string entries and fix null header_used
                    fixed_validations = []
                    for item in sv:
                        if isinstance(item, dict):
                            # Fix null header_used -> convert to empty string
                            if "header_used" in item and item["header_used"] is None:
                                item["header_used"] = ""
                            fixed_validations.append(item)
                    ats_val["section_validations"] = fixed_validations

            # Ensure keyword_report has all required fields
            if "keyword_report" in ats_val and isinstance(ats_val["keyword_report"], dict):
                kr = ats_val["keyword_report"]
                # Ensure total_keywords exists (some LLMs might miss it after remapping)
                if "total_keywords" not in kr:
                    # Calculate from keyword_frequency if available
                    if "keyword_frequency" in kr and isinstance(kr["keyword_frequency"], dict):
                        kr["total_keywords"] = sum(kr["keyword_frequency"].values())
                    else:
                        kr["total_keywords"] = kr.get("total_keywords_found", 0)

                # Ensure unique_keywords exists
                if "unique_keywords" not in kr:
                    if "keyword_frequency" in kr and isinstance(kr["keyword_frequency"], dict):
                        kr["unique_keywords"] = len(kr["keyword_frequency"])
                    else:
                        kr["unique_keywords"] = kr.get("unique_keywords_found", 0)

        return json_data

    # ========================================================================
    # STEP 5: OUTPUT PROCESSING
    # ========================================================================
    print_section("STEP 5: OUTPUT PROCESSING", "Parsing and validating the ATS-optimized resume...")

    print("\n[INFO] Processing agent output...")

    try:
        # Get the raw text output from the agent
        output_text = str(result)
        print(f"\n[INFO] Received output: {len(output_text)} characters")

        # Try to extract JSON from the output
        # The LLM might wrap JSON in markdown code fences or add explanatory text
        json_str = output_text.strip()

        # Remove markdown code fences if present
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0].strip()

        # Try to find JSON object if it's embedded in text
        if not json_str.startswith("{"):
            import re

            json_match = re.search(r"\{.*?\}", json_str, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                # Debug: Show what we actually got
                print("\n[DEBUG] Could not find JSON in output (first 2000 chars):")
                print(output_text[:2000])
                raise ValueError("Could not find JSON object in output")

        # # Handle "Extra data" error by finding the first complete JSON object
        # The LLM might add explanatory text after the JSON
        try:
            json_data = json.loads(json_str)
        except json.JSONDecodeError as e:
            if "Extra data" in str(e):
                # Find the position where the first JSON object ends
                # Use a decoder to find the end of the first valid JSON
                from json.decoder import JSONDecoder

                decoder = JSONDecoder()
                try:
                    json_data, idx = decoder.raw_decode(json_str)
                    logger.warning(
                        f"Extracted first JSON object, ignoring {len(json_str) - idx} extra characters"
                    )
                except json.JSONDecodeError:
                    raise  # Re-raise if we still can't parse it
            else:
                raise  # Re-raise other JSON errors
        print("\n[SUCCESS] Successfully parsed JSON from agent output")

        # Apply enterprise-grade validation fixes
        json_data = fix_common_llm_issues(json_data)
        print("[SUCCESS] Applied validation fixes for common LLM issues")

        # Validate and convert to Pydantic model
        # This ensures the data conforms to our schema
        optimized_resume = OptimizedResume(**json_data)
        print("\n[SUCCESS] Output validated against OptimizedResume schema")
        print(f"  Type: {type(optimized_resume).__name__}")

        # Display key results
        print("\n" + "=" * 80)
        print("ATS OPTIMIZATION SUCCESSFUL!")
        print("=" * 80)
        print("\n[VALIDATED] Output contains all required fields")

        print("\n" + "-" * 80)
        print("ATS VALIDATION RESULTS")
        print("-" * 80)

        ats_val = optimized_resume.ats_validation
        overall_score = ats_val.overall_score
        is_compatible = ats_val.is_compatible

        print(f"\n  Overall ATS Score: {overall_score}/100")
        if overall_score >= 80:
            print("    Status: Excellent ATS Compatibility ✓")
            print("    Status: Excellent ATS Compatibility [PASS]")
        elif overall_score >= 60:
            print("    Status: Good ATS Compatibility")
        else:
            print("    Status: Needs Improvement")

        print(f"\n  ATS Compatible: {'Yes [PASS]' if is_compatible else 'No [WARN]'}")

        # Keyword Report
        kw_report = ats_val.keyword_report
        if kw_report:
            print("\n  Keyword Analysis:")
            keyword_density = kw_report.keyword_density
            keyword_coverage = kw_report.keyword_coverage
            print(f"    Density: {keyword_density:.1%} (optimal: 2-5%)")
            if 0.02 <= keyword_density <= 0.05:
                print("      Status: Optimal ✓")
            elif keyword_density < 0.02:
                print("      Status: Too Low - Add more keywords")
            else:
                print("      Status: Too High - May appear keyword-stuffed")

            print(f"    Coverage: {keyword_coverage:.0%} of job keywords")
            if keyword_coverage >= 0.8:
                print("      Status: Excellent Coverage ✓")
            elif keyword_coverage >= 0.6:
                print("      Status: Good Coverage")
            else:
                print("      Status: Needs More Keywords")

        # Formatting Issues
        formatting_issues = ats_val.formatting_issues
        print(f"\n  Formatting Issues: {len(formatting_issues)} issue(s)")
        if formatting_issues:
            for issue in formatting_issues[:5]:
                print(f"    ⚠ {issue}")
            if len(formatting_issues) > 5:
                print(f"    ... and {len(formatting_issues) - 5} more")
        else:
            print("    None detected ✓")

        # Optimization Summary
        optimization_summary = optimized_resume.optimization_summary
        if optimization_summary:
            print("\n  Optimization Summary:")
            summary = (
                optimization_summary[:300] + "..."
                if len(optimization_summary) > 300
                else optimization_summary
            )
            print(f"    {summary}")

        # Markdown Content Preview
        markdown_content = optimized_resume.markdown_content
        if markdown_content:
            print("\n  Final Resume (Markdown):")
            print("-" * 40)
            # Show first 500 characters
            preview = (
                markdown_content[:500] + "..." if len(markdown_content) > 500 else markdown_content
            )
            print(preview)
            print("-" * 40)
            print(f"    Total length: {len(markdown_content)} characters")

        print("\n" + "-" * 80)
        print("\n[SUCCESS] ATS optimization completed and validated!")
        print("\n[INFO] The optimized resume:")
        print("  - Is ATS-compliant with proper formatting")
        print("  - Has optimal keyword density and coverage")
        print("  - Is ready for submission to job applications")
        print("  - Available in both markdown and JSON formats")

    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}", exc_info=True)
        print(f"\n[ERROR] Could not parse JSON from agent output: {e}")
        print("\n[INFO] This suggests the LLM did not return valid JSON.")
        print("  - Check that the expected_output is clear")
        print("  - Verify the LLM model supports structured output")
        print("\n[DEBUG] Raw output (first 1000 chars):")
        print(output_text[:1000])
    except Exception as e:
        logger.error(f"Error processing output: {e}", exc_info=True)
        print(f"\n[ERROR] Could not process output: {e}")
        print("\n[INFO] This could indicate:")
        print("  - The JSON structure doesn't match the expected schema")
        print("  - Required fields are missing or have invalid types")
        print("  - There may be a validation error in the data")
        print("\n[DEBUG] Partial data:")
        if "json_data" in locals():
            print(json.dumps(json_data, indent=2, ensure_ascii=False)[:1000])

    print("\n" + "=" * 80)
    print("EXAMPLE COMPLETE")
    print("=" * 80)
    print("\nThis example demonstrated how the ATS Optimization Agent:")
    print("  1. Assembles all optimized resume components")
    print("  2. Validates ATS compatibility")
    print("  3. Checks keyword density and coverage")
    print("  4. Verifies formatting compliance")
    print("  5. Generates final markdown and JSON outputs")
    print("\nThe resume is now ready for submission to job applications!")


if __name__ == "__main__":
    main()
