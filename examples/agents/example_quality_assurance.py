"""
Example: Quality Assurance Reviewer Agent
==========================================

OBJECTIVE:
----------
This example demonstrates how the Quality Assurance Reviewer Agent performs
comprehensive quality evaluation using quantitative metrics across three weighted
dimensions: Accuracy (40%), Relevance (35%), and ATS Optimization (25%).

WHAT THIS AGENT DOES:
---------------------
1. Receives all optimized resume components:
   - Original Resume (from Resume Extractor Agent)
   - Tailored Resume (assembled from all optimization agents)
   - Job Description (from Job Analyzer Agent)
   - Professional Summary (from Summary Writer Agent)
   - Optimized Experience (from Experience Optimizer Agent)
   - Optimized Skills (from Skills Optimizer Agent)
2. Evaluates Accuracy (40% weight):
   - Compares tailored vs original to detect exaggerations
   - Flags unsupported skills
   - Validates factual consistency
3. Evaluates Relevance (35% weight):
   - Checks alignment with job requirements
   - Calculates must-have skills coverage
   - Identifies missed requirements
4. Evaluates ATS Optimization (25% weight):
   - Validates keyword coverage (target: 60-80%)
   - Checks formatting compliance
   - Verifies standard section headers
5. Calculates weighted overall score (pass threshold: 80/100)
6. Generates actionable feedback if failed

INPUT:
------
- Sample original resume (from common.py)
- Sample tailored resume (mock data representing optimized output)
- Sample job description (from common.py)

EXPECTED OUTPUT:
----------------
- A QualityReport object with comprehensive evaluation:
  {
    "overall_quality_score": 88.5,
    "passed_quality_threshold": true,
    "assessment_summary": "High quality resume, approved for submission",
    "accuracy": {
      "accuracy_score": 95.0,
      "exaggerated_claims": [],
      "unsupported_skills": [],
      "justification": "..."
    },
    "relevance": {
      "relevance_score": 85.0,
      "must_have_skills_coverage": 90.0,
      "missed_requirements": [],
      "justification": "..."
    },
    "ats_optimization": {
      "ats_score": 82.0,
      "keyword_coverage": 75.0,
      "formatting_issues": [],
      "justification": "..."
    },
    "feedback_for_improvement": null
  }

STEP-BY-STEP PROCESS:
---------------------
Step 1: Load sample data (original resume, tailored resume, job description)
Step 2: Create the Quality Assurance Reviewer Agent
Step 3: Define the evaluation task with all inputs
Step 4: Execute the agent (calls LLM to evaluate quality)
Step 5: Parse and validate the output
Step 6: Display the quality report with pass/fail decision

WHY THIS MATTERS:
-----------------
The Quality Assurance Reviewer serves as the final gatekeeper, ensuring:
- No exaggerated or fabricated claims (protects candidate credibility)
- Job requirements are properly addressed (increases interview chances)
- ATS compatibility is validated (passes automated screening)
- Professional standards are met (grammar, formatting, structure)

This is the ethical guardian that prevents dishonest resumes while ensuring
high-quality, truthful resumes are approved for submission.
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from crewai import Crew, Process, Task  # noqa: E402

from src.agents.quality_assurance_agent import (  # noqa: E402
    check_qa_quality,
    create_quality_assurance_agent,
)
from src.core.logger import get_logger  # noqa: E402
from src.data_models.evaluation import QualityReport  # noqa: E402

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


def create_mock_tailored_resume():
    """
    Create mock tailored resume data for the example.

    In a real workflow, this would be the assembled output from:
    - Professional Summary Writer Agent
    - Experience Section Optimizer Agent
    - Skills Section Strategist Agent
    - ATS Optimization Specialist Agent

    This mock data represents a high-quality optimized resume.
    """
    return {
        "personal_info": {
            "full_name": "Sarah Johnson",
            "email": "sarah.johnson@gmail.com",
            "phone": "+1-555-0123",
            "location": "San Francisco, CA",
        },
        "summary": (
            "Senior Software Engineer with 8+ years of experience in Python, JavaScript, "
            "and cloud technologies. Led cross-functional teams to deliver scalable "
            "microservices architectures, reducing infrastructure costs by 40%. Expertise "
            "in AWS, Docker, Kubernetes, and CI/CD automation. Passionate about building "
            "high-performance systems that drive business value."
        ),
        "experience": [
            {
                "job_title": "Senior Software Engineer",
                "company": "TechCorp",
                "start_date": "2020-01",
                "end_date": "Present",
                "descriptions": [
                    "Led team of 5 engineers to architect microservices platform using Python and Kubernetes, reducing deployment time by 70%",
                    "Implemented CI/CD pipeline using Docker and Jenkins, enabling 50+ deployments per week with 99.9% success rate",
                    "Optimized AWS infrastructure (EC2, Lambda, RDS), reducing monthly costs by $15,000 while improving performance by 35%",
                    "Mentored junior developers on best practices for code review, testing, and agile methodologies",
                ],
            },
            {
                "job_title": "Software Engineer",
                "company": "StartupXYZ",
                "start_date": "2017-06",
                "end_date": "2019-12",
                "descriptions": [
                    "Developed RESTful APIs using Python Flask and PostgreSQL, serving 1M+ daily requests with <100ms latency",
                    "Built data processing pipelines using Apache Kafka and Redis, handling 10TB of data daily",
                    "Implemented automated testing suite (pytest, unittest), achieving 90% code coverage and reducing bugs by 60%",
                ],
            },
            {
                "job_title": "Junior Developer",
                "company": "WebSolutions Inc",
                "start_date": "2015-03",
                "end_date": "2017-05",
                "descriptions": [
                    "Developed responsive web applications using JavaScript, React, and Node.js for 20+ clients",
                    "Collaborated with design team to implement pixel-perfect UIs, increasing user engagement by 25%",
                    "Maintained legacy PHP applications and migrated to modern tech stack (React + Node.js)",
                ],
            },
        ],
        "education": [
            {
                "degree": "Bachelor of Science in Computer Science",
                "institution": "Stanford University",
                "graduation_year": 2015,
            }
        ],
        "skills": [
            "Python",
            "JavaScript",
            "AWS",
            "Docker",
            "Kubernetes",
            "React",
            "Node.js",
            "PostgreSQL",
            "Redis",
            "Apache Kafka",
            "CI/CD",
            "Jenkins",
            "Git",
            "Agile",
            "Microservices",
        ],
    }


def create_mock_job_description():
    """
    Create mock job description data for the example.

    This represents the structured output from the Job Analyzer Agent.
    """
    return {
        "job_title": "Senior Software Engineer",
        "company": "Tech Giant Inc",
        "requirements": [
            {"description": "5+ years of Python development", "importance_level": "must-have"},
            {"description": "Experience with AWS cloud services", "importance_level": "must-have"},
            {"description": "Kubernetes and Docker expertise", "importance_level": "must-have"},
            {"description": "CI/CD pipeline experience", "importance_level": "should-have"},
            {"description": "Leadership and mentoring skills", "importance_level": "should-have"},
        ],
        "skills": {
            "must_have": ["Python", "AWS", "Kubernetes", "Docker"],
            "should_have": ["CI/CD", "Microservices", "Agile", "PostgreSQL"],
        },
        "keywords": [
            "Python",
            "AWS",
            "Kubernetes",
            "Docker",
            "CI/CD",
            "Microservices",
            "Leadership",
            "Agile",
            "PostgreSQL",
            "Redis",
            "Team Lead",
        ],
    }


def main():
    print("\n" + "=" * 80)
    print("QUALITY ASSURANCE REVIEWER AGENT - EDUCATIONAL EXAMPLE")
    print("=" * 80)
    print("\nThis example shows how the AI evaluates resume quality using")
    print("quantitative metrics across Accuracy, Relevance, and ATS Optimization.\n")

    # ========================================================================
    # STEP 1: PREPARE INPUT DATA
    # ========================================================================
    print_section("STEP 1: INPUT DATA", "Preparing evaluation inputs...")

    # In a real workflow, the original resume comes from Resume Extractor Agent
    original_resume = create_mock_tailored_resume()  # Using same as mock

    # The tailored resume is the assembled output from all optimization agents
    tailored_resume = create_mock_tailored_resume()

    # Job description from Job Analyzer Agent
    job_description = create_mock_job_description()

    print("\n[INPUT] Original Resume:")
    print(f"  Name: {original_resume['personal_info']['full_name']}")
    print(f"  Experience entries: {len(original_resume['experience'])}")
    print(f"  Skills: {len(original_resume['skills'])}")

    print("\n[INPUT] Tailored Resume:")
    print(f"  Name: {tailored_resume['personal_info']['full_name']}")
    print(f"  Summary: {tailored_resume['summary'][:80]}...")
    print(f"  Experience entries: {len(tailored_resume['experience'])}")
    print(f"  Skills: {len(tailored_resume['skills'])}")

    print("\n[INPUT] Job Description:")
    print(f"  Position: {job_description['job_title']}")
    print(f"  Company: {job_description['company']}")
    print(f"  Must-have skills: {', '.join(job_description['skills']['must_have'])}")

    print("\n[INFO] In a real workflow, these inputs come from:")
    print("  - Original Resume: Resume Extractor Agent")
    print("  - Tailored Resume: Assembled from all optimization agents")
    print("  - Job Description: Job Analyzer Agent")

    # ========================================================================
    # STEP 2: CREATE THE AGENT
    # ========================================================================
    print_section("STEP 2: AGENT CREATION", "Initializing the Quality Assurance Reviewer Agent...")

    try:
        agent = create_quality_assurance_agent()
        print("\n[SUCCESS] Agent created successfully!")
        print(f"  Role: {agent.role}")
        print(
            f"  Goal: {agent.goal[:100]}..." if len(agent.goal) > 100 else f"  Goal: {agent.goal}"
        )
        print("\n[INFO] This agent is specialized in:")
        print("  - Evaluating accuracy (40% weight): Detect exaggerations")
        print("  - Evaluating relevance (35% weight): Check job alignment")
        print("  - Evaluating ATS optimization (25% weight): Validate formatting")
        print("  - Calculating weighted score (pass threshold: 80/100)")
        print("  - Generating actionable feedback for failed reviews")
    except Exception as e:
        logger.error(f"Failed to create agent: {e}", exc_info=True)
        print(f"\n[ERROR] Failed to create agent: {e}")
        return

    # ========================================================================
    # STEP 3: DEFINE THE TASK
    # ========================================================================
    print_section("STEP 3: TASK DEFINITION", "Setting up the quality evaluation task...")

    # Load task configuration from tasks.yaml (same as real application)
    try:
        from src.core.config import get_tasks_config

        tasks_config = get_tasks_config()
        task_config = tasks_config.get("quality_assurance_task", {})

        if not task_config:
            raise ValueError("quality_assurance_task not found in tasks.yaml")

        # Get the base task description and expected_output from config
        base_description = task_config.get("description", "")
        base_expected_output = task_config.get("expected_output", "")

        print("\n[INFO] Loaded task configuration from tasks.yaml")
        print("  Task: quality_assurance_task")
        print(f"  Agent: {task_config.get('agent', 'N/A')}")

    except Exception as e:
        logger.error(f"Failed to load task config: {e}", exc_info=True)
        print(f"\n[ERROR] Failed to load task configuration: {e}")
        return

    # Adapt the task description for the example context
    # The real task expects data from previous agents, but we're providing data directly
    #
    # IMPORTANT: We use output_pydantic=QualityReport to enforce structured output.
    # This is the industry-standard approach in CrewAI for ensuring LLM outputs
    # match our Pydantic model schema.
    task_description = (
        f"IMPORTANT: The resume and job description content is provided below as JSON. "
        f"Work directly with this data to evaluate resume quality.\n\n"
        f"ORIGINAL RESUME (for accuracy comparison):\n{json.dumps(original_resume, indent=2)}\n\n"
        f"TAILORED RESUME (to be evaluated):\n{json.dumps(tailored_resume, indent=2)}\n\n"
        f"JOB DESCRIPTION:\n{json.dumps(job_description, indent=2)}\n\n"
        f"{base_description}\n\n"
        f"CRITICAL INSTRUCTIONS:\n"
        f"- Evaluate ACCURACY (40% weight): Compare tailored vs original, detect exaggerations\n"
        f"- Evaluate RELEVANCE (35% weight): Check job requirement coverage\n"
        f"- Evaluate ATS OPTIMIZATION (25% weight): Validate keyword coverage and formatting\n"
        f"- Calculate weighted overall score: accuracy*0.4 + relevance*0.35 + ats*0.25\n"
        f"- Pass threshold: 80/100 or higher\n"
        f"- If failed, provide specific actionable feedback\n\n"
        f"OUTPUT REQUIREMENT:\n"
        f"You must output a SINGLE, valid JSON object that conforms to the QualityReport schema.\n"
        f"Include all three metric dimensions (accuracy, relevance, ats_optimization) with scores "
        f"and justifications. Calculate the weighted overall score correctly."
    )

    # Override the expected_output to match QualityReport model structure
    expected_output = (
        "Output ONLY a valid JSON object that matches the QualityReport model structure. "
        "Do not include any markdown formatting or explanations. "
        "The JSON must be parseable and conform exactly to the schema."
    )

    task = Task(
        description=task_description,
        expected_output=expected_output,
        agent=agent,
        output_pydantic=QualityReport,  # ⭐ STRUCTURED OUTPUT ENFORCEMENT
    )

    print("\n[INFO] Task configured with:")
    print("  - Centralized task description from tasks.yaml (quality_assurance_task)")
    print("  - Centralized expected_output from tasks.yaml")
    print("  - Adapted for example context (JSON data provided directly)")
    print("  - Structured output enforcement (output_pydantic=QualityReport)")

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
    print("  1. Compare tailored vs original resume for accuracy")
    print("  2. Check alignment with job requirements")
    print("  3. Validate ATS compatibility and formatting")
    print("  4. Calculate weighted overall score")
    print("  5. Determine pass/fail (threshold: 80/100)")
    print("  6. Generate actionable feedback if needed")
    print("\n[WAIT] This may take 45-90 seconds as the LLM evaluates quality...\n")

    try:
        result = crew.kickoff()
    except Exception as e:
        logger.error(f"Error during agent execution: {e}", exc_info=True)
        print(f"\n[ERROR] Agent execution failed: {e}")
        return

    # ========================================================================
    # STEP 5: PROCESS AND VALIDATE RESULTS
    # ========================================================================
    print_section("STEP 5: OUTPUT PROCESSING", "Parsing and validating the quality report...")

    print("\n[INFO] Processing structured output from agent...")

    try:
        # Access the validated Pydantic object directly
        # CrewAI automatically validates and converts the LLM output to our Pydantic model
        if hasattr(result, "pydantic") and result.pydantic is not None:
            quality_report = result.pydantic
            print("\n[SUCCESS] Structured output validated successfully!")
            print(f"  Type: {type(quality_report).__name__}")

            # Convert to dict for easier access
            json_data = quality_report.model_dump(mode="json")
        else:
            # Fallback: try to parse from raw output if pydantic is not available
            logger.warning("result.pydantic not available, attempting manual parsing...")
            output_text = str(result)
            print("\n[WARNING] Falling back to manual JSON parsing")
            print(f"  Output length: {len(output_text)} characters")

            # Try to extract and parse JSON
            import re

            json_match = re.search(r"\{.*\}", output_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                json_data = json.loads(json_str)
                quality_report = QualityReport(**json_data)
            else:
                raise ValueError("Could not find JSON in output")

        # ====================================================================
        # STEP 6: DISPLAY RESULTS
        # ====================================================================
        print("\n" + "=" * 80)
        print("QUALITY EVALUATION COMPLETE!")
        print("=" * 80)

        # Overall Assessment
        print("\n" + "-" * 80)
        print("OVERALL ASSESSMENT")
        print("-" * 80)

        overall_score = quality_report.overall_quality_score
        passed = quality_report.passed_quality_threshold

        print(f"\n  Overall Quality Score: {overall_score:.1f}/100")

        if passed:
            print("  Status: ✓ PASSED (Approved for submission)")
            print(f"  Assessment: {quality_report.assessment_summary}")
        else:
            print("  Status: ✗ FAILED (Corrections required)")
            print(f"  Assessment: {quality_report.assessment_summary}")

        # Component Scores
        print("\n" + "-" * 80)
        print("COMPONENT SCORES (Weighted)")
        print("-" * 80)

        # Accuracy (40% weight)
        accuracy = quality_report.accuracy
        print(f"\n  1. ACCURACY (40% weight): {accuracy.accuracy_score:.1f}/100")
        if accuracy.accuracy_score >= 90:
            print("     Status: Excellent ✓")
        elif accuracy.accuracy_score >= 75:
            print("     Status: Good")
        else:
            print("     Status: Needs Improvement ⚠")

        print(f"     Exaggerated claims: {len(accuracy.exaggerated_claims)}")
        if accuracy.exaggerated_claims:
            for claim in accuracy.exaggerated_claims[:3]:
                print(f"       - {claim}")

        print(f"     Unsupported skills: {len(accuracy.unsupported_skills)}")
        if accuracy.unsupported_skills:
            for skill in accuracy.unsupported_skills[:3]:
                print(f"       - {skill}")

        print(f"     Justification: {accuracy.justification[:150]}...")

        # Relevance (35% weight)
        relevance = quality_report.relevance
        print(f"\n  2. RELEVANCE (35% weight): {relevance.relevance_score:.1f}/100")
        if relevance.relevance_score >= 90:
            print("     Status: Excellent ✓")
        elif relevance.relevance_score >= 75:
            print("     Status: Good")
        else:
            print("     Status: Needs Improvement ⚠")

        print(f"     Must-have coverage: {relevance.must_have_skills_coverage:.0f}%")
        print(f"     Missed requirements: {len(relevance.missed_requirements)}")
        if relevance.missed_requirements:
            for req in relevance.missed_requirements[:3]:
                print(f"       - {req}")

        print(f"     Justification: {relevance.justification[:150]}...")

        # ATS Optimization (25% weight)
        ats = quality_report.ats_optimization
        print(f"\n  3. ATS OPTIMIZATION (25% weight): {ats.ats_score:.1f}/100")
        if ats.ats_score >= 90:
            print("     Status: Excellent ✓")
        elif ats.ats_score >= 75:
            print("     Status: Good")
        else:
            print("     Status: Needs Improvement ⚠")

        print(f"     Keyword coverage: {ats.keyword_coverage:.0f}%")
        if 60 <= ats.keyword_coverage <= 80:
            print("       (Optimal range: 60-80%) ✓")
        elif ats.keyword_coverage < 60:
            print("       (Below target: add more keywords) ⚠")
        else:
            print("       (Above target: possible keyword stuffing) ⚠")

        print(f"     Formatting issues: {len(ats.formatting_issues)}")
        if ats.formatting_issues:
            for issue in ats.formatting_issues[:3]:
                print(f"       - {issue}")

        print(f"     Justification: {ats.justification[:150]}...")

        # Feedback for Improvement (if failed)
        if not passed and quality_report.feedback_for_improvement:
            print("\n" + "-" * 80)
            print("FEEDBACK FOR IMPROVEMENT")
            print("-" * 80)
            print(f"\n{quality_report.feedback_for_improvement}")

        # Meta-Validation
        print("\n" + "-" * 80)
        print("META-VALIDATION (Quality of QA Output)")
        print("-" * 80)

        meta_result = check_qa_quality(quality_report)
        print(f"\n  Meta-Quality Score: {meta_result['meta_quality_score']:.1f}/100")
        print(f"  Is Valid: {meta_result['is_valid']}")
        print(f"  Summary: {meta_result['summary']}")
        if meta_result["issues"]:
            print("  Issues:")
            for issue in meta_result["issues"]:
                print(f"    - {issue}")

        print("\n" + "-" * 80)
        print("\n[SUCCESS] Quality evaluation completed successfully!")
        print("\n[INFO] The quality report:")
        if passed:
            print("  - Resume PASSED quality threshold (80/100)")
            print("  - Approved for submission to job application")
            print("  - All claims are accurate and well-supported")
            print("  - Job requirements are properly addressed")
            print("  - ATS compatibility is validated")
        else:
            print("  - Resume FAILED quality threshold (80/100)")
            print("  - Corrections required before submission")
            print("  - Review feedback for specific improvements")
            print("  - Re-run optimization agents to address issues")

    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}", exc_info=True)
        print(f"\n[ERROR] Could not parse JSON from agent output: {e}")
        print("\n[INFO] This suggests the structured output enforcement may have failed.")
        print("  - Check that output_pydantic=QualityReport is set in the Task")
        print("  - Verify the LLM is returning valid JSON matching the schema")
        if "output_text" in locals():
            print("\n[DEBUG] Raw output (first 500 chars):")
            print(output_text[:500])
    except Exception as e:
        logger.error(f"Error processing output: {e}", exc_info=True)
        print(f"\n[ERROR] Could not process output: {e}")
        print("\n[INFO] This could indicate:")
        print("  - The LLM output doesn't match the QualityReport schema")
        print("  - Required fields are missing or invalid")
        print("  - There may be a validation error in the Pydantic model")

    print("\n" + "=" * 80)
    print("EXAMPLE COMPLETE")
    print("=" * 80)
    print("\nThis example demonstrated how the Quality Assurance Reviewer Agent:")
    print("  1. Evaluates accuracy by comparing tailored vs original resume")
    print("  2. Evaluates relevance by checking job requirement coverage")
    print("  3. Evaluates ATS optimization by validating keywords and formatting")
    print("  4. Calculates weighted overall score (Accuracy 40%, Relevance 35%, ATS 25%)")
    print("  5. Determines pass/fail based on 80/100 threshold")
    print("  6. Generates actionable feedback for failed reviews")
    print("\nThe resume quality is now validated and ready for final decision!")


if __name__ == "__main__":
    main()
