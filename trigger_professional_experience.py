"""Trigger the Professional Experience agent in isolation.

Pipeline:
  1. PDF resume       → extract_structured Resume
  2. Job text         → extract_structured JobDescription
  3. Gap bypass       → AlignmentStrategy (strategic guidance)
  4. Experience node  → role-scoped OptimizedExperienceSection
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# from crewai import Crew, Process, Task

# from src.agents.gap_analysis import create_gap_analysis_agent
from src.agents.job_description_analyser import create_job_analyzer_agent
from src.agents.professional_experience import create_professional_experience_agent
# from src.core.settings import get_tasks_config
from src.data_models.job import JobDescription
from src.data_models.strategy import AlignmentStrategy
# from src.formatters.gap_analysis_formatter import format_gap_analysis_context
from src.orchestration.crew_task_execution import run_agent_task
from src.orchestration.nodes import optimize_experience
from src.tools.document_ingestion import (
    convert_document_to_markdown,
    extract_resume,
)

# ── paths ─────────────────────────────────────────────────────────────────────

PDF_PATH = str(Path("sample_documents/5de9878acaf813644e37cbf8_5de98838caf81311687f9290.pdf").resolve())
JD_PATH = str(Path("sample_documents/sample_job_description.txt").resolve())

# ── step 1: extract Resume ────────────────────────────────────────────────────

print("=== STEP 1: Extract Resume ===")
resume_md = convert_document_to_markdown(PDF_PATH)
resume = extract_resume(resume_md)
print("  Name       :", resume.full_name)
print("  Experience :", len(resume.work_experience), "roles")

# ── step 2: extract JobDescription ────────────────────────────────────────────

print("\n=== STEP 2: Extract JobDescription ===")
jd_markdown = Path(JD_PATH).read_text()
jda_agent = create_job_analyzer_agent()
job = run_agent_task(
    agent=jda_agent,
    task_name="analyze_job_description_task",
    context=f"JOB DESCRIPTION:\n{jd_markdown}",
    output_model=JobDescription,
)
print("  Title      :", job.job_title)
print("  Requirements:", len(job.requirements), "requirements")

# ── step 3: bypass Gap Analysis → AlignmentStrategy ───────────────────────────

print("\n=== STEP 3: Bypass Gap Analysis ===")
# gap_agent = create_gap_analysis_agent()
# gap_task_config = get_tasks_config().get("create_alignment_strategy_task", {})
# gap_context = format_gap_analysis_context(
#     resume=resume,
#     job_description=job,
#     format_type="toon",
# )
#
# gap_result = Crew(
#     agents=[gap_agent],
#     tasks=[
#         Task(
#             description=(
#                 f"{gap_task_config.get('description', '')}\n\n"
#                 f"CONTEXT:\n{gap_context}"
#             ),
#             expected_output=gap_task_config.get(
#                 "expected_output", "A validated AlignmentStrategy object."
#             ),
#             agent=gap_agent,
#             output_pydantic=AlignmentStrategy,
#         )
#     ],
#     process=Process.sequential,
#     verbose=False,
# ).kickoff()
#
# strategy = gap_result.pydantic

strategy = AlignmentStrategy(
    overall_fit_score=72.0,
    summary_of_strategy=(
        "Emphasize backend engineering, distributed systems, cloud/serverless, "
        "CI/CD, and high-scale AI platform work for the Senior Backend Engineer role."
    ),
    identified_matches=[],
    identified_gaps=[],
    keywords_to_integrate=[
        "Python",
        "RESTful APIs",
        "microservices",
        "AWS",
        "AWS Lambda",
        "Docker",
        "Kubernetes",
        "CI/CD",
        "system design",
        "distributed systems",
        "event-driven architecture",
        "database design",
        "Agile",
        "observability",
        "serverless",
    ],
    professional_summary_guidance=(
        "Lead with backend, cloud, and AI platform experience, while keeping the "
        "summary truthful to the candidate's actual work history."
    ),
    experience_guidance=(
        "For each role, rewrite only the provided role evidence. Prioritize backend "
        "architecture, cloud/serverless systems, CI/CD, scale, latency, reliability, "
        "and distributed-system impact where the original role supports it."
    ),
    skills_guidance=(
        "Prioritize Python, AWS, serverless, Docker, CI/CD, distributed systems, "
        "database design, and backend architecture skills that are evidenced in work history."
    ),
)
print("  Fit score  :", strategy.overall_fit_score)
print("  Keywords   :", len(strategy.keywords_to_integrate))

# ── step 4: run Professional Experience agent ─────────────────────────────────

print("\n=== STEP 4: Run Professional Experience Agent ===")

agent = create_professional_experience_agent()
print("  Agent  :", agent.role)
print("  Model  :", agent.llm.model)
print("  Tools  :", [t.name for t in agent.tools])

experience_state = {
    "resume": resume,
    "job_description": job,
    "alignment_strategy": strategy,
}
section = optimize_experience(experience_state)["optimized_experience"]

# ── step 5: print results ─────────────────────────────────────────────────────

print("\n=== OPTIMIZED EXPERIENCE ===")
print("  Entries optimized :", len(section.optimized_experiences))
print("  Keywords integrated:", len(section.keywords_integrated))
print("  Keywords           :", section.keywords_integrated)
print("  Notes              :", section.optimization_notes[:150], "...")
for exp in section.optimized_experiences:
    print(f"\n  ── {exp.company_name} — {exp.job_title} ──")
    for achievement in exp.achievements:
        print(f"    • {achievement[:100]}...")
print("\n  CONTRACT: valid OptimizedExperienceSection ready for assembly")
