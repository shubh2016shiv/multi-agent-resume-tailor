"""Trigger the Quality Assessment agent in isolation.

Quality Assessment is the LAST agent, so it needs the full pipeline ahead of it:

  1. PDF resume   -> Resume
  2. Job text     -> JobDescription
  3. Gap analysis -> AlignmentStrategy
  4. Summary      -> ProfessionalSummary
  5. Experience   -> OptimizedExperienceSection
  6. Skills       -> OptimizedSkillsSection
  7. ATS optimizer-> AtsOptimizedResume
  8. Quality assessment -> QualityReport        (the module under test)
  9. Code-owned gate (engines.apply_quality_gate) -> the render decision

Step 9 is the whole point: it shows the deterministic boolean that would gate PDF
rendering, computed from the score in code -- not by the LLM.
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from crewai import Crew, Process, Task

from src.agents.ats_optimizer import create_ats_optimizer_agent
from src.agents.ats_optimizer.models import AtsOptimizedResume
from src.agents.gap_analysis import create_gap_analysis_agent
from src.agents.job_description_analyser import create_job_analyzer_agent
from src.agents.professional_experience import create_professional_experience_agent
from src.agents.professional_experience.models import OptimizedExperienceSection
from src.agents.professional_summary import create_professional_summary_agent
from src.agents.professional_summary.models import ProfessionalSummary
from src.agents.quality_assessment import create_quality_assessment_agent
from src.agents.quality_assessment.engines import apply_quality_gate, should_render_resume
from src.agents.resume_parser import create_resume_extractor_agent
from src.agents.skill_optimizer import create_skill_optimizer_agent
from src.data_models.evaluation import QualityReport
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

# ── step 3: gap analysis ──────────────────────────────────────────────────────

print("\n=== STEP 3: Gap Analysis ===")
strategy = run_agent_task(
    agent=create_gap_analysis_agent(),
    task_name="create_alignment_strategy_task",
    context=format_gap_analysis_context(resume=resume, job_description=job, format_type="toon"),
    output_model=AlignmentStrategy,
)
print("  Fit score  :", strategy.overall_fit_score)

# ── step 4: professional summary ──────────────────────────────────────────────

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

# ── step 5: experience ────────────────────────────────────────────────────────

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

# ── step 6: skills ────────────────────────────────────────────────────────────

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

# ── step 7: ATS optimizer ─────────────────────────────────────────────────────

print("\n=== STEP 7: ATS Optimizer ===")
optimized = run_agent_task(
    agent=create_ats_optimizer_agent(),
    task_name="optimize_ats_resume_task",
    context=format_ats_optimization_context(
        professional_summary=summary,
        optimized_experience=optimized_experience,
        optimized_skills=optimized_skills,
        original_resume=resume,
        job_description=job,
        format_type="toon",
    ),
    output_model=AtsOptimizedResume,
)
print("  Section order:", optimized.section_order)

# ── step 8: Quality Assessment (module under test) ────────────────────────────

print("\n=== STEP 8: Run Quality Assessment Agent ===")
agent = create_quality_assessment_agent()
print("  Agent  :", agent.role)
print("  Model  :", agent.llm.model)
print("  Tools  :", [t.name for t in agent.tools])

# The production QA formatter is still coupled to the OLD OptimizedResume type, so
# build the context inline from the three things QA fundamentally needs.
qa_context = (
    f"OPTIMIZED_RESUME_JSON:\n{optimized.final_resume.model_dump_json()}\n\n"
    f"ORIGINAL_RESUME_JSON:\n{resume.model_dump_json()}\n\n"
    f"JOB_DESCRIPTION_JSON:\n{job.model_dump_json()}"
)
report: QualityReport = run_agent_task(
    agent=agent,
    task_name="assess_quality_task",
    context=qa_context,
    output_model=QualityReport,
)

# ── step 9: code-owned render gate ────────────────────────────────────────────

print("\n=== QUALITY REPORT (agent output) ===")
print("  Overall score          :", report.overall_quality_score)
print("  Accuracy score         :", report.accuracy.accuracy_score)
print("  Relevance score        :", report.relevance.relevance_score)
print("  ATS score              :", report.ats_optimization.ats_score)
print("  LLM-set passed flag    :", report.passed_quality_threshold)
print("  Feedback               :", (report.feedback_for_improvement or "(none)")[:140])

gated = apply_quality_gate(report)
print("\n=== CODE-OWNED GATE (engines.apply_quality_gate, threshold=80) ===")
print("  Authoritative passed   :", gated.passed_quality_threshold)
print("  RENDER PDF?            ->", should_render_resume(gated))
print("\n  CONTRACT: QualityReport with a code-computed render gate (not LLM whim)")
