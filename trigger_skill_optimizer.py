"""Trigger the Skill Optimizer agent in isolation.

Pipeline:
  1. PDF resume       → extract_structured Resume
  2. Job text         → extract_structured JobDescription
  3. Gap analysis     → AlignmentStrategy (strategic guidance)
  4. Skill optimizer  → OptimizedSkillsSection
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from crewai import Crew, Process, Task

from src.agents.gap_analysis import create_gap_analysis_agent
from src.agents.job_description_analyser import create_job_analyzer_agent
from src.agents.resume_parser import create_resume_extractor_agent
from src.agents.skill_optimizer import create_skill_optimizer_agent
from src.core.config import get_tasks_config
from src.data_models.job import JobDescription
from src.data_models.resume import OptimizedSkillsSection, Resume
from src.data_models.strategy import AlignmentStrategy
from src.orchestration.crew_task_execution import run_agent_task
from src.tools.document_ingestion import convert_document_to_markdown

# ── paths ─────────────────────────────────────────────────────────────────────

PDF_PATH = str(Path("sample_documents/Shubham_Resume_2026_April_version2.pdf").resolve())
JD_PATH = str(Path("sample_documents/sample_job_description.txt").resolve())

# ── step 1: extract Resume ────────────────────────────────────────────────────

print("=== STEP 1: Extract Resume ===")
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
print("  Keywords   :", len(strategy.keywords_to_integrate))

# ── step 4: run Skill Optimizer agent ─────────────────────────────────────────

print("\n=== STEP 4: Run Skill Optimizer Agent ===")

agent = create_skill_optimizer_agent()
print("  Agent  :", agent.role)
print("  Model  :", agent.llm.model)
print("  Tools  :", [t.name for t in agent.tools])

task_config = get_tasks_config().get("optimize_skills_section_task", {})

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
                "A validated OptimizedSkillsSection object.",
            ),
            agent=agent,
            output_pydantic=OptimizedSkillsSection,
        )
    ],
    process=Process.sequential,
    verbose=True,
).kickoff()

section = result.pydantic

# ── step 5: print results ─────────────────────────────────────────────────────

print("\n=== OPTIMIZED SKILLS ===")
print("  Skills optimized :", len(section.optimized_skills))
print("  Categories       :", list(section.skill_categories.keys()))
print("  Added            :", len(section.added_skills))
print("  Removed          :", len(section.removed_skills))
print("  ATS match score  :", section.ats_match_score)
print("  Notes            :", section.optimization_notes[:150], "...")
print("\n  Categories:")
for category, skills in section.skill_categories.items():
    print(f"    {category}: {', '.join(skills[:5])}...")
print("\n  CONTRACT: valid OptimizedSkillsSection ready for assembly")
