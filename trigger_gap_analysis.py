"""Trigger the Gap Analysis agent in isolation.

Pipeline:
  1. PDF resume  → extract_structured Resume
  2. Job text    → extract_structured JobDescription
  3. Run gap analysis agent → AlignmentStrategy
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from crewai import Crew, Process, Task

from src.agents.gap_analysis import create_gap_analysis_agent
from src.agents.job_description_analyser import create_job_analyzer_agent
from src.agents.resume_parser import create_resume_extractor_agent
from src.core.config import get_tasks_config
from src.data_models.job import JobDescription
from src.data_models.resume import Resume
from src.data_models.strategy import AlignmentStrategy
from src.orchestration.crew_task_execution import run_agent_task
from src.tools.document_ingestion import convert_document_to_markdown

# ── paths ─────────────────────────────────────────────────────────────────────

PDF_PATH = str(Path("sample_documents/Shubham_Resume_2026_April_version2.pdf").resolve())
JD_PATH = str(Path("sample_documents/sample_job_description.txt").resolve())

# ── step 1: extract Resume from PDF ───────────────────────────────────────────

print("=== STEP 1: Extract Resume from PDF ===")
resume_md = convert_document_to_markdown(PDF_PATH)

resume_agent = create_resume_extractor_agent()
print("  Agent  :", resume_agent.role)
print("  Model  :", resume_agent.llm.model)

resume_result = Crew(
    agents=[resume_agent],
    tasks=[
        Task(
            description=(f"Resume file path: {PDF_PATH}\n\nRESUME MARKDOWN CONTENT:\n{resume_md}"),
            expected_output="A validated Resume object.",
            agent=resume_agent,
            output_pydantic=Resume,
        )
    ],
    process=Process.sequential,
).kickoff()

resume = resume_result.pydantic
print("  Name       :", resume.full_name)
print("  Email      :", resume.email)
print("  Experience :", len(resume.work_experience), "roles")
print("  Skills     :", len(resume.skills), "skills")
print("  CONTRACT   : valid Resume object")

# ── step 2: extract JobDescription from text ──────────────────────────────────

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
print("  Company    :", job.company_name)
print("  Level      :", job.job_level.value)
print("  Requirements:", len(job.requirements), "requirements")
print("  Keywords   :", len(job.ats_keywords), "ATS keywords")
print("  CONTRACT   : valid JobDescription object")

# ── step 3: run Gap Analysis agent ────────────────────────────────────────────

print("\n=== STEP 3: Run Gap Analysis Agent ===")

agent = create_gap_analysis_agent()
print("  Agent  :", agent.role)
print("  Model  :", agent.llm.model)
print("  Tools  :", [t.name for t in agent.tools])

task_config = get_tasks_config().get("create_alignment_strategy_task", {})

resume_json = resume.model_dump_json()
job_json = job.model_dump_json()

task = Task(
    description=(
        f"RESUME DATA:\n{resume_json}\n\n"
        f"JOB DESCRIPTION DATA:\n{job_json}\n\n"
        f"{task_config.get('description', '')}"
    ),
    expected_output=task_config.get(
        "expected_output",
        "A validated AlignmentStrategy object.",
    ),
    agent=agent,
    output_pydantic=AlignmentStrategy,
)

result = Crew(
    agents=[agent],
    tasks=[task],
    process=Process.sequential,
    verbose=True,
).kickoff()

strategy = result.pydantic

# ── step 4: print results ─────────────────────────────────────────────────────

print("\n=== ALIGNMENT STRATEGY ===")
print("  Fit score         :", strategy.overall_fit_score)
print("  Summary           :", strategy.summary_of_strategy[:120], "...")
print("  Matches           :", len(strategy.identified_matches))
print("  Gaps              :", len(strategy.identified_gaps))
print("  Keywords to add   :", len(strategy.keywords_to_integrate))
print("  Keywords          :", strategy.keywords_to_integrate[:5], "...")
print("  Summary guidance  :", strategy.professional_summary_guidance[:100], "...")
print("  Experience guidance:", strategy.experience_guidance[:100], "...")
print("  Skills guidance   :", strategy.skills_guidance[:100], "...")
print("  CONTRACT          : valid AlignmentStrategy ready for downstream agents")
