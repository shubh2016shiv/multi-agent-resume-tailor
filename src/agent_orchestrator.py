"""
Agent Orchestrator Module
=========================

This module defines the `ResumeTailorOrchestrator` class, which manages the end-to-end
execution of the Resume Tailor agentic workflow. It orchestrates multiple specialized
AI agents to process a resume and job description, optimize content, and generate
a final tailored resume.

KEY FEATURES:
-------------
- **Parallel Execution**: Runs independent agents concurrently to reduce total latency.
- **Sequential Dependencies**: Manages data flow where agents depend on previous outputs.
- **Structured State**: Uses Pydantic models for type-safe data passing between stages.
- **Robust Error Handling**: Catches and reports errors at each stage.
- **Traceability**: Logs execution progress and results.

WORKFLOW STAGES:
----------------
1. **Ingestion & Analysis** (Parallel):
   - Resume Extractor: Parses raw resume text -> `Resume`
   - Job Analyzer: Parses job description -> `JobDescription`
2. **Strategic Planning** (Sequential):
   - Gap Analysis: Compares Resume & Job -> `AlignmentStrategy`
3. **Content Optimization** (Parallel):
   - Summary Writer: Generates professional summary -> `ProfessionalSummary`
   - Experience Optimizer: Tailors work history -> `OptimizedExperienceSection`
   - Skills Optimizer: Selects relevant skills -> `OptimizedSkillsSection`
4. **Final Assembly** (Sequential):
   - ATS Optimization: Assembles final resume -> `OptimizedResume`
5. **Quality Assurance** (Sequential):
   - QA Reviewer: Validates final output -> `QualityReport`
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

from crewai import Crew, Process, Task
from pydantic import BaseModel

from src.agents.ats_optimization_agent import OptimizedResume, create_ats_optimization_agent
from src.agents.experience_optimizer_agent import (
    OptimizedExperienceSection,
    create_experience_optimizer_agent,
)
from src.agents.gap_analysis_agent import create_gap_analysis_agent
from src.agents.job_analyzer_agent import create_job_analyzer_agent
from src.agents.quality_assurance_agent import create_quality_assurance_agent

# Agent Imports & Specific Models
from src.agents.resume_extractor_agent import create_resume_extractor_agent
from src.agents.skills_optimizer_agent import create_skills_optimizer_agent
from src.agents.summary_writer_agent import ProfessionalSummary, create_summary_writer_agent

# Core Imports
from src.core.config import get_config, get_tasks_config
from src.core.logger import get_logger
from src.data_models.evaluation import QualityReport
from src.data_models.job import JobDescription

# Data Models
from src.data_models.resume import OptimizedSkillsSection, Resume
from src.data_models.strategy import AlignmentStrategy
from src.tools.document_converter import convert_document_to_markdown

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class OrchestrationResult(BaseModel):
    """Container for the final output of the orchestration process."""

    original_resume: Resume
    job_description: JobDescription
    strategy: AlignmentStrategy
    optimized_resume: OptimizedResume
    qa_report: QualityReport


class ResumeTailorOrchestrator:
    """
    Orchestrates the execution of the Resume Tailor multi-agent system.
    """

    def __init__(self):
        self.tasks_config = get_tasks_config()
        self.config = get_config()

    def load_files_from_config(self) -> tuple[str, str]:
        """
        Load resume and job description files from configuration.

        Returns:
            Tuple of (resume_text, job_description_text) loaded from configured paths.

        Raises:
            FileNotFoundError: If configured files don't exist.
        """
        from pathlib import Path

        # Get paths from config
        resume_path = self.config.file_paths.resume_path
        jd_path = self.config.file_paths.job_description_path

        # Resolve relative paths from project root
        project_root = Path(__file__).parent.parent
        resume_full_path = (project_root / resume_path).resolve()
        jd_full_path = (project_root / jd_path).resolve()

        logger.info(f"Loading resume from: {resume_full_path}")
        logger.info(f"Loading job description from: {jd_full_path}")

        # Load and convert documents
        resume_text = convert_document_to_markdown(str(resume_full_path))
        jd_text = convert_document_to_markdown(str(jd_full_path))

        logger.info(f"Resume loaded: {len(resume_text)} characters")
        logger.info(f"Job description loaded: {len(jd_text)} characters")

        return resume_text, jd_text

    def run_from_config(self) -> OrchestrationResult:
        """
        Run the orchestration workflow using file paths from configuration.

        Returns:
            OrchestrationResult containing all intermediate and final artifacts.
        """
        resume_text, jd_text = self.load_files_from_config()
        return self.orchestrate(resume_text, jd_text)

    def orchestrate(self, resume_text: str, job_description_text: str) -> OrchestrationResult:
        """
        Main entry point to run the full resume tailoring workflow.

        Args:
            resume_text: Raw text/markdown of the candidate's resume.
            job_description_text: Raw text/markdown of the target job description.

        Returns:
            OrchestrationResult containing all intermediate and final artifacts.
        """
        logger.info("Starting Resume Tailor Orchestration...")

        # ==============================================================================
        # BLOCK 1: INGESTION & ANALYSIS (PARALLEL)
        # ==============================================================================
        # STAGE 1: Extract structured data from raw text inputs.
        # - Resume Extractor and Job Analyzer run in parallel as they are independent.
        logger.info("STAGE 1: Running Ingestion & Analysis (Parallel)...")

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_resume = executor.submit(self._run_resume_extraction, resume_text)
            future_job = executor.submit(self._run_job_analysis, job_description_text)

            resume_data = future_resume.result()
            job_data = future_job.result()

        logger.info("Stage 1 Complete: Resume and Job Description extracted.")

        # ==============================================================================
        # BLOCK 2: STRATEGIC PLANNING (SEQUENTIAL)
        # ==============================================================================
        # STAGE 2: Analyze gaps and form a tailoring strategy.
        # - Depends on both Resume and JobDescription from Stage 1.
        logger.info("STAGE 2: Running Strategic Planning (Sequential)...")

        alignment_strategy = self._run_gap_analysis(resume_data, job_data)

        logger.info(
            f"Stage 2 Complete: Strategy formed (Fit Score: {alignment_strategy.overall_fit_score})."
        )

        # ==============================================================================
        # BLOCK 3: CONTENT OPTIMIZATION (PARALLEL)
        # ==============================================================================
        # STAGE 3: Optimize specific sections of the resume based on the strategy.
        # - Summary, Experience, and Skills agents can run independently.
        logger.info("STAGE 3: Running Content Optimization (Parallel)...")

        with ThreadPoolExecutor(max_workers=3) as executor:
            future_summary = executor.submit(
                self._run_summary_writing, resume_data, job_data, alignment_strategy
            )
            future_experience = executor.submit(
                self._run_experience_optimization, resume_data, job_data, alignment_strategy
            )
            future_skills = executor.submit(
                self._run_skills_optimization, resume_data, job_data, alignment_strategy
            )

            prof_summary = future_summary.result()
            opt_experience = future_experience.result()
            opt_skills = future_skills.result()

        logger.info("Stage 3 Complete: Resume sections optimized.")

        # ==============================================================================
        # BLOCK 4: FINAL ASSEMBLY (SEQUENTIAL)
        # ==============================================================================
        # STAGE 4: Assemble the final resume and ensure ATS compliance.
        # - Depends on all optimized sections from Stage 3 + original data.
        logger.info("STAGE 4: Running Final Assembly (Sequential)...")

        optimized_resume = self._run_ats_assembly(
            prof_summary, opt_experience, opt_skills, resume_data, job_data
        )

        logger.info("Stage 4 Complete: Final resume assembled.")

        # ==============================================================================
        # BLOCK 5: QUALITY ASSURANCE (SEQUENTIAL)
        # ==============================================================================
        # STAGE 5: Final review of the generated resume.
        # - Depends on the final assembled resume.
        logger.info("STAGE 5: Running Quality Assurance (Sequential)...")

        qa_report = self._run_quality_assurance(optimized_resume, job_data)

        logger.info(
            f"Stage 5 Complete: Quality Assurance finished (Score: {qa_report.overall_score})."
        )

        return OrchestrationResult(
            original_resume=resume_data,
            job_description=job_data,
            strategy=alignment_strategy,
            optimized_resume=optimized_resume,
            qa_report=qa_report,
        )

    # ==============================================================================
    # HELPER METHODS: AGENT EXECUTION WRAPPERS
    # ==============================================================================

    def _create_and_run_crew(
        self,
        agent: Any,
        task_name: str,
        description_params: str,
        output_model: type[T],
        expected_output_override: str | None = None,
    ) -> T:
        """
        Generic helper to instantiate a Crew, create a Task, and run it.
        """
        # Get task config
        task_config = self.tasks_config.get(task_name, {})
        base_desc = task_config.get(
            "description", f"Process the following data: {description_params}"
        )
        base_expected = task_config.get("expected_output", "Structured output matching the model.")

        if expected_output_override:
            base_expected = expected_output_override

        # Create Task
        task = Task(
            description=f"{base_desc}\n\nCONTEXT:\n{description_params}",
            expected_output=base_expected,
            agent=agent,
            output_pydantic=output_model,
        )

        # Create Crew
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)

        # Execute
        result = crew.kickoff()

        # Access Typed Output
        if hasattr(result, "pydantic") and result.pydantic:
            return result.pydantic
        else:
            raise ValueError(
                f"Agent {agent.role} did not return a valid {output_model.__name__} object."
            )

    def _run_resume_extraction(self, text: str) -> Resume:
        agent = create_resume_extractor_agent()
        return self._create_and_run_crew(
            agent=agent,
            task_name="extract_resume_content_task",
            description_params=f"RESUME CONTENT:\n{text}",
            output_model=Resume,
        )

    def _run_job_analysis(self, text: str) -> JobDescription:
        agent = create_job_analyzer_agent()
        return self._create_and_run_crew(
            agent=agent,
            task_name="analyze_job_description_task",
            description_params=f"JOB DESCRIPTION:\n{text}",
            output_model=JobDescription,
        )

    def _run_gap_analysis(self, resume: Resume, job: JobDescription) -> AlignmentStrategy:
        agent = create_gap_analysis_agent()
        # Serialize inputs to JSON/Text for the prompt
        context = f"RESUME DATA:\n{resume.model_dump_json()}\n\nJOB DATA:\n{job.model_dump_json()}"
        return self._create_and_run_crew(
            agent=agent,
            task_name="gap_analysis_task",
            description_params=context,
            output_model=AlignmentStrategy,
        )

    def _run_summary_writing(
        self, resume: Resume, job: JobDescription, strategy: AlignmentStrategy
    ) -> ProfessionalSummary:
        agent = create_summary_writer_agent()
        context = (
            f"RESUME DATA:\n{resume.model_dump_json()}\n\n"
            f"JOB DATA:\n{job.model_dump_json()}\n\n"
            f"STRATEGY:\n{strategy.model_dump_json()}"
        )
        return self._create_and_run_crew(
            agent=agent,
            task_name="generate_summary_task",
            description_params=context,
            output_model=ProfessionalSummary,
        )

    def _run_experience_optimization(
        self, resume: Resume, job: JobDescription, strategy: AlignmentStrategy
    ) -> OptimizedExperienceSection:
        agent = create_experience_optimizer_agent()
        context = (
            f"RESUME WORK EXPERIENCE:\n{resume.model_dump_json(include={'work_experience'})}\n\n"
            f"JOB REQUIREMENTS:\n{job.model_dump_json(include={'requirements', 'ats_keywords'})}\n\n"
            f"STRATEGY GAPS & MATCHES:\n{strategy.model_dump_json()}"
        )
        return self._create_and_run_crew(
            agent=agent,
            task_name="optimize_experience_task",
            description_params=context,
            output_model=OptimizedExperienceSection,
        )

    def _run_skills_optimization(
        self, resume: Resume, job: JobDescription, strategy: AlignmentStrategy
    ) -> OptimizedSkillsSection:
        agent = create_skills_optimizer_agent()
        context = (
            f"CURRENT SKILLS:\n{resume.model_dump_json(include={'skills'})}\n\n"
            f"JOB TARGETS:\n{job.model_dump_json(include={'requirements', 'ats_keywords'})}\n\n"
            f"STRATEGY:\n{strategy.model_dump_json()}"
        )
        return self._create_and_run_crew(
            agent=agent,
            task_name="optimize_skills_task",
            description_params=context,
            output_model=OptimizedSkillsSection,
        )

    def _run_ats_assembly(
        self,
        summary: ProfessionalSummary,
        experience: OptimizedExperienceSection,
        skills: OptimizedSkillsSection,
        original_resume: Resume,
        job: JobDescription,
    ) -> OptimizedResume:
        agent = create_ats_optimization_agent()

        # We need to combine all parts.
        # The ATS agent expects specific inputs.
        context = (
            f"OPTIMIZED SUMMARY:\n{summary.model_dump_json()}\n\n"
            f"OPTIMIZED EXPERIENCE:\n{experience.model_dump_json()}\n\n"
            f"OPTIMIZED SKILLS:\n{skills.model_dump_json()}\n\n"
            f"ORIGINAL RESUME (For Education/Contact):\n{original_resume.model_dump_json()}\n\n"
            f"TARGET JOB:\n{job.model_dump_json()}"
        )

        return self._create_and_run_crew(
            agent=agent,
            task_name="compile_resume_task",  # Verify this task name in config if possible, or generic
            description_params=context,
            output_model=OptimizedResume,
        )

    def _run_quality_assurance(self, resume: OptimizedResume, job: JobDescription) -> QualityReport:
        agent = create_quality_assurance_agent()
        context = (
            f"FINAL RESUME:\n{resume.model_dump_json()}\n\n"
            f"JOB DESCRIPTION:\n{job.model_dump_json()}"
        )
        return self._create_and_run_crew(
            agent=agent,
            task_name="quality_assurance_task",
            description_params=context,
            output_model=QualityReport,
        )


# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================
# This block allows running the orchestrator directly for testing and development.
# ==============================================================================

if __name__ == "__main__":
    """
    Main entry point for running the Resume Tailor Orchestrator.

    This will:
    1. Load file paths from settings.yaml
    2. Convert documents to text
    3. Run the full orchestration workflow
    4. Display results
    """
    print("=" * 80)
    print("RESUME TAILOR ORCHESTRATOR - EXECUTION")
    print("=" * 80)
    print()

    try:
        # Create orchestrator instance
        orchestrator = ResumeTailorOrchestrator()

        # Run workflow using config file paths
        print("Starting orchestration workflow...")
        print(f"Resume path: {orchestrator.config.file_paths.resume_path}")
        print(f"Job description path: {orchestrator.config.file_paths.job_description_path}")
        print()

        result = orchestrator.run_from_config()

        # Display results
        print("\n" + "=" * 80)
        print("ORCHESTRATION COMPLETE")
        print("=" * 80)
        print(f"\nOriginal Resume: {result.original_resume.full_name}")
        print(f"Job Title: {result.job_description.job_title}")
        print(f"Company: {result.job_description.company_name}")
        print(f"\nAlignment Strategy Fit Score: {result.strategy.overall_fit_score}/100")
        print(f"Quality Assurance Score: {result.qa_report.overall_score}/100")
        print(
            f"\nOptimized Resume ATS Score: {result.optimized_resume.ats_validation.overall_score}/100"
        )
        print(f"ATS Compatible: {result.optimized_resume.ats_validation.is_compatible}")

        print("\n" + "=" * 80)
        print("SUCCESS: Orchestration completed successfully!")
        print("=" * 80)

    except FileNotFoundError as e:
        print(f"\n[ERROR] File not found: {e}")
        print("Please check the file paths in src/config/settings.yaml")
        exit(1)
    except Exception as e:
        print(f"\n[ERROR] Orchestration failed: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
