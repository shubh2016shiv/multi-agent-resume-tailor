"""Trigger the ATS Optimizer agent in isolation.

Pipeline (the ATS optimizer is the LAST content stage, so it needs every upstream
output as a real, typed input):

  1. PDF resume   -> Resume
  2. Job text     -> JobDescription
  3. Gap analysis -> AlignmentStrategy
  4. Summary      -> ProfessionalSummary
  5. Experience   -> OptimizedExperienceSection
  6. Skills       -> OptimizedSkillsSection
  7. ATS optimizer-> AtsOptimizedResume        (the module under test)
  8. Code-owned ATS quality measurement (engines.check_ats_quality)

Each upstream step uses the same production factory + run_agent_task path the
LangGraph nodes use, so this exercises the real contracts end to end.
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from crewai import Crew, Process, Task

from src.agents.ats_optimizer import create_ats_optimizer_agent
from src.agents.ats_optimizer.engines import check_ats_quality
from src.agents.ats_optimizer.models import AtsOptimizedResume
from src.agents.gap_analysis import create_gap_analysis_agent
from src.agents.job_description_analyser import create_job_analyzer_agent
from src.agents.professional_experience import create_professional_experience_agent
from src.agents.professional_experience.models import OptimizedExperienceSection
from src.agents.professional_summary import create_professional_summary_agent
from src.agents.professional_summary.models import ProfessionalSummary
from src.agents.resume_parser import create_resume_extractor_agent
from src.agents.skill_optimizer import create_skill_optimizer_agent
from src.data_models.job import JobDescription
from src.data_models.resume import OptimizedSkillsSection, Resume
from src.data_models.strategy import AlignmentStrategy
from src.formatters.ats_optimization_formatter import format_ats_optimization_context
from src.formatters.experience_optimizer_formatter import format_experience_optimizer_context
from src.formatters.gap_analysis_formatter import format_gap_analysis_context
from src.formatters.professional_summary_formatter import format_professional_summary_context
from src.formatters.skills_optimizer_formatter import format_skills_optimizer_context
from src.orchestration.crew_task_execution import run_agent_task
from src.tools.document_ingestion import convert_document_to_markdown

# ── paths ─────────────────────────────────────────────────────────────────────

PDF_PATH = str(Path("sample_documents/Shubham_Resume_2026_April_version2.pdf").resolve())
JD_PATH = str(Path("sample_documents/sample_job_description.txt").resolve())

# ── step 1: extract Resume ────────────────────────────────────────────────────

print("=== STEP 1: Extract Resume from PDF ===")
resume_md = convert_document_to_markdown(PDF_PATH)
resume_agent = create_resume_extractor_agent()
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
job = run_agent_task(
    agent=create_job_analyzer_agent(),
    task_name="analyze_job_description_task",
    context=f"JOB DESCRIPTION:\n{jd_markdown}",
    output_model=JobDescription,
)
print("  Title      :", job.job_title)
print("  Keywords   :", len(job.ats_keywords), "ATS keywords")

# ── step 3: gap analysis -> AlignmentStrategy ─────────────────────────────────

print("\n=== STEP 3: Gap Analysis ===")
strategy = run_agent_task(
    agent=create_gap_analysis_agent(),
    task_name="create_alignment_strategy_task",
    context=format_gap_analysis_context(resume=resume, job_description=job, format_type="toon"),
    output_model=AlignmentStrategy,
)
print("  Fit score  :", strategy.overall_fit_score)
print("  Keywords   :", len(strategy.keywords_to_integrate))

# ── step 4: professional summary -> ProfessionalSummary ───────────────────────

print("\n=== STEP 4: Professional Summary ===")
summary = run_agent_task(
    agent=create_professional_summary_agent(),
    task_name="write_professional_summary_task",
    context=format_professional_summary_context(
        resume=resume, job_description=job, strategy=strategy, format_type="toon"
    ),
    output_model=ProfessionalSummary,
)
print("  Drafts     :", len(summary.drafts))
print("  Recommended:", summary.recommended_version)

# ── step 5: experience -> OptimizedExperienceSection ──────────────────────────

print("\n=== STEP 5: Experience Optimization ===")
optimized_experience = run_agent_task(
    agent=create_professional_experience_agent(),
    task_name="optimize_experience_section_task",
    context=format_experience_optimizer_context(
        resume=resume, job_description=job, strategy=strategy, format_type="toon"
    ),
    output_model=OptimizedExperienceSection,
)
print("  Entries    :", len(optimized_experience.optimized_experiences))

# ── step 6: skills -> OptimizedSkillsSection ──────────────────────────────────

print("\n=== STEP 6: Skills Optimization ===")
optimized_skills = run_agent_task(
    agent=create_skill_optimizer_agent(),
    task_name="optimize_skills_section_task",
    context=format_skills_optimizer_context(
        resume=resume, job_description=job, strategy=strategy, format_type="toon"
    ),
    output_model=OptimizedSkillsSection,
)
print("  Skills     :", len(optimized_skills.optimized_skills))

# ── step 7: ATS optimizer -> AtsOptimizedResume (module under test) ───────────

print("\n=== STEP 7: Run ATS Optimizer Agent ===")
agent = create_ats_optimizer_agent()
print("  Agent  :", agent.role)
print("  Model  :", agent.llm.model)
print("  Tools  :", [t.name for t in agent.tools])

ats_context = format_ats_optimization_context(
    professional_summary=summary,
    optimized_experience=optimized_experience,
    optimized_skills=optimized_skills,
    original_resume=resume,
    job_description=job,
    format_type="toon",
)
optimized = run_agent_task(
    agent=agent,
    task_name="optimize_ats_resume_task",
    context=ats_context,
    output_model=AtsOptimizedResume,
)

# ── step 8: code-owned ATS quality measurement ────────────────────────────────

print("\n=== ATS OPTIMIZED RESUME ===")
print("  Section order       :", optimized.section_order)
print("  Final skills        :", len(optimized.final_resume.skills))
print("  Final experience    :", len(optimized.final_resume.work_experience))
print("  Optimization summary:", optimized.optimization_summary[:120], "...")
print("  Keyword notes       :", optimized.keyword_integration_notes[:120], "...")
print("  Unresolved issues   :", optimized.unresolved_issues)

print("\n=== CODE-OWNED ATS QUALITY (engines.check_ats_quality) ===")
report = check_ats_quality(optimized, job)
print("  Overall status   :", report["overall_status"])
print("  Keyword coverage  :", report["keyword_coverage"])
print("  Formatting issues :", report["formatting_issues"])
print("  Header issues     :", report["header_issues"])
print("  Serious findings  :", report["serious_findings"])
print("\n  CONTRACT: valid AtsOptimizedResume, measured code-side (not self-scored)")
