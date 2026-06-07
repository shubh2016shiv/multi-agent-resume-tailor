"""Trigger the Professional Summary agent in isolation.

Pipeline:
  1. PDF resume       → extract_structured Resume
  2. Job text         → extract_structured JobDescription
  3. Gap analysis     → AlignmentStrategy (strategic guidance)
  4. Summary writer   → ProfessionalSummary (4 drafts + recommendation)
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from crewai import Crew, Process, Task

from src.agents.gap_analysis import create_gap_analysis_agent
from src.agents.job_description_analyser import create_job_analyzer_agent
from src.agents.professional_summary import create_professional_summary_agent
from src.agents.professional_summary.models import ProfessionalSummary
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

# ── step 1: extract Resume ────────────────────────────────────────────────────

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
print("  Experience :", len(resume.work_experience), "roles")
print("  Skills     :", len(resume.skills), "skills")

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
print("  Company    :", job.company_name)
print("  Requirements:", len(job.requirements), "requirements")
print("  Keywords   :", len(job.ats_keywords), "ATS keywords")

# ── step 3: run Gap Analysis → AlignmentStrategy ──────────────────────────────

print("\n=== STEP 3: Run Gap Analysis ===")
gap_agent = create_gap_analysis_agent()
gap_task_config = get_tasks_config().get("create_alignment_strategy_task", {})

gap_result = Crew(
    agents=[gap_agent],
    tasks=[
        Task(
            description=(
                f"RESUME DATA:\n{resume.model_dump_json()}\n\n"
                f"JOB DESCRIPTION DATA:\n{job.model_dump_json()}\n\n"
                f"{gap_task_config.get('description', '')}"
            ),
            expected_output=gap_task_config.get(
                "expected_output", "A validated AlignmentStrategy object."
            ),
            agent=gap_agent,
            output_pydantic=AlignmentStrategy,
        )
    ],
    process=Process.sequential,
    verbose=False,
).kickoff()

strategy = gap_result.pydantic
print("  Fit score  :", strategy.overall_fit_score)
print("  Matches    :", len(strategy.identified_matches))
print("  Gaps       :", len(strategy.identified_gaps))
print("  Keywords   :", len(strategy.keywords_to_integrate))

# ── step 4: run Professional Summary agent ────────────────────────────────────

print("\n=== STEP 4: Run Professional Summary Agent ===")

agent = create_professional_summary_agent()
print("  Agent  :", agent.role)
print("  Model  :", agent.llm.model)
print("  Tools  :", [t.name for t in agent.tools])

task_config = get_tasks_config().get("write_professional_summary_task", {})

result = Crew(
    agents=[agent],
    tasks=[
        Task(
            description=(
                f"RESUME DATA:\n{resume.model_dump_json()}\n\n"
                f"JOB DESCRIPTION DATA:\n{job.model_dump_json()}\n\n"
                f"ALIGNMENT STRATEGY:\n{strategy.model_dump_json()}\n\n"
                f"{task_config.get('description', '')}"
            ),
            expected_output=task_config.get(
                "expected_output",
                "A validated ProfessionalSummary object.",
            ),
            agent=agent,
            output_pydantic=ProfessionalSummary,
        )
    ],
    process=Process.sequential,
    verbose=True,
).kickoff()

summary = result.pydantic

# ── step 5: print results ─────────────────────────────────────────────────────

print("\n=== PROFESSIONAL SUMMARY ===")
print("  Drafts generated :", len(summary.drafts))
print("  Recommended      :", summary.recommended_version)
for draft in summary.drafts:
    print(f"\n  ── {draft.version_name} (score: {draft.score}) ──")
    print(f"  Strategy: {draft.strategy_used}")
    print(f"  Content : {draft.content}")
    print(f"  Critique: {draft.critique}")
print(f"\n  Writing notes: {summary.writing_notes}")
print("  CONTRACT: valid ProfessionalSummary ready for assembly")
