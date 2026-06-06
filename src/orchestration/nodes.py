"""
One function per agent call. No orchestration logic here.

Each node reads the fields it needs from ResumeEnhancementPipelineState,
runs exactly one CrewAI agent, and returns a partial state dict.
LangGraph merges that dict back into the shared state.

Precondition for every node: the fields it reads must already be set
(non-None) in the state. The graph topology in graph.py enforces this.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from crewai import Agent, Crew, Process, Task
from pydantic import BaseModel

from src.agents.ats_optimization_agent import OptimizedResume, create_ats_optimization_agent
from src.agents.gap_analysis import create_gap_analysis_agent
from src.agents.job_analyzer_agent import create_job_analyzer_agent
from src.agents.professional_experience import create_professional_experience_agent
from src.agents.professional_experience.models import OptimizedExperienceSection
from src.agents.quality_assurance_agent import create_quality_assurance_agent
# src/agents/resume_parser is the authoritative resume extraction agent.
# It owns all 4 document-ingestion tools (convert -> quality check -> redact -> extract).
# The node passes a file path -- the agent reasons through the steps itself.
from src.agents.resume_parser import create_resume_extractor_agent
from src.agents.skills_optimizer_agent import create_skills_optimizer_agent
from src.agents.professional_summary import create_professional_summary_agent
from src.agents.professional_summary.models import ProfessionalSummary
from src.core.config import get_tasks_config
from src.data_models.evaluation import QualityReport
from src.data_models.job import JobDescription
from src.data_models.resume import Experience, OptimizedSkillsSection, Resume
from src.data_models.strategy import AlignmentStrategy
from src.formatters.ats_optimization_formatter import format_ats_optimization_context
from src.formatters.experience_optimizer_formatter import format_experience_optimizer_context
from src.formatters.gap_analysis_formatter import format_gap_analysis_context
from src.formatters.professional_summary_formatter import format_professional_summary_context
from src.formatters.quality_assurance_formatter import format_quality_assurance_context
from src.formatters.skills_optimizer_formatter import format_skills_optimizer_context
from src.orchestration.state import ResumeEnhancementPipelineState
from src.tools.document_ingestion.document_converter import convert_document_to_markdown
from src.tools.document_ingestion.resume_section_extractor import assign_experience_ids
from src.tools.resume_diagnostics import audit_experience_quality_for_experiences
from src.tools.review_contract.review_models import ReviewResult, Severity


_SERIOUS_EXPERIENCE_AUDIT_SEVERITIES = {Severity.BLOCKER, Severity.MAJOR}


def _run_single_agent_crew(
    agent: Agent,
    task_name: str,
    context: str,
    output_model: type[BaseModel],
) -> Any:
    """Run one CrewAI agent on one task and return its typed Pydantic output.

    Precondition: agent is a configured CrewAI Agent; task_name exists in tasks.yaml.
    Returns: a validated instance of output_model.
    Raises: ValueError if the agent does not produce a valid output_model instance.
    """
    tasks_config = get_tasks_config()
    task_config = tasks_config.get(task_name, {})
    task_description = task_config.get("description", "") + "\n\nCONTEXT:\n" + context
    task_expected_output = task_config.get("expected_output", "Structured output.")

    task = Task(
        description=task_description,
        expected_output=task_expected_output,
        agent=agent,
        output_pydantic=output_model,
    )
    result = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    ).kickoff()

    if hasattr(result, "pydantic") and result.pydantic:
        return result.pydantic
    raise ValueError(f"Agent {agent.role} did not return a valid {output_model.__name__} object.")


def _build_resume_with_single_experience(resume: Resume, experience: Experience) -> Resume:
    """Copy resume context with only one work experience entry.

    Expects a validated Resume and one Experience from resume.work_experience.
    Returns a Resume copy whose work_experience list contains only that entry.
    """
    # Stage 3 / Step 3.2: Build one-role resume context
    # Receives from: Resume.work_experience entry selected by orchestration.
    # Sends to: format_experience_optimizer_context(..., format_type="toon").
    return resume.model_copy(update={"work_experience": [experience]})


def _require_single_optimized_experience(section: OptimizedExperienceSection) -> Experience:
    """Return the single optimized experience expected from a role-scoped call.

    Expects an OptimizedExperienceSection from one role-only CrewAI task.
    Raises ValueError when the LLM returns zero or multiple experiences.
    """
    if len(section.optimized_experiences) != 1:
        raise ValueError("Role-scoped optimization must return exactly one experience.")
    return section.optimized_experiences[0]


def _restore_original_experience_id(
    section: OptimizedExperienceSection,
    original_experience: Experience,
) -> OptimizedExperienceSection:
    """Copy the code-owned experience_id onto the LLM-written experience.

    Expects one optimized experience and the original source Experience.
    Returns a section whose optimized experience keeps the original ID.
    """
    # Stage 3 / Step 3.6: Restore the code-owned experience_id
    # Receives from: original Experience.experience_id and LLM-written Experience.
    # Sends to: merged OptimizedExperienceSection for downstream ATS assembly.
    optimized = _require_single_optimized_experience(section)
    fixed = optimized.model_copy(update={"experience_id": original_experience.experience_id})
    return section.model_copy(update={"optimized_experiences": [fixed]})


def _experience_audit_needs_rewrite(audit_result: ReviewResult) -> bool:
    """Return True when audit comments are serious enough for one rewrite.

    Expects a ReviewResult from experience quality checks.
    Returns True for blocker or major findings only.
    """
    return any(
        comment.severity in _SERIOUS_EXPERIENCE_AUDIT_SEVERITIES
        for comment in audit_result.comments
    )


def _render_experience_audit_feedback(audit_result: ReviewResult) -> str:
    """Render audit comments into compact feedback for one rewrite attempt.

    Expects a ReviewResult from the experience audit helper.
    Returns plain text suitable for a CrewAI task context.
    """
    lines = [audit_result.summary or "Experience audit found serious issues."]
    for comment in audit_result.comments:
        lines.append(f"- {comment.severity.value}: {comment.message}")
        lines.append(f"  advice: {comment.advice}")
    return "\n".join(lines)


def _build_experience_rewrite_context(
    original_context: str,
    section: OptimizedExperienceSection,
    audit_result: ReviewResult,
) -> str:
    """Add previous output and audit feedback to the original role context.

    Expects original TOON context, prior structured output, and audit result.
    Returns context for exactly one rewrite attempt.
    """
    return (
        f"{original_context}\n\n"
        f"PREVIOUS_OPTIMIZED_EXPERIENCE_JSON:\n{section.model_dump_json()}\n\n"
        f"EXPERIENCE_AUDIT_FEEDBACK:\n{_render_experience_audit_feedback(audit_result)}\n\n"
        "Rewrite once to address blocker or major audit findings. "
        "Return only OptimizedExperienceSection JSON."
    )


def _write_experience_section(context: str) -> OptimizedExperienceSection:
    """Ask the professional experience agent to write one role.

    Expects TOON context for a single role.
    Returns an OptimizedExperienceSection validated by CrewAI.
    """
    # Stage 3 / Step 3.3: Ask the LLM to write one role
    # Receives from: TOON context produced by format_experience_optimizer_context.
    # Sends to: OptimizedExperienceSection validated by CrewAI output_pydantic.
    return _run_single_agent_crew(
        agent=create_professional_experience_agent(),
        task_name="optimize_experience_section_task",
        context=context,
        output_model=OptimizedExperienceSection,
    )


def _audit_experience_section(section: OptimizedExperienceSection) -> ReviewResult:
    """Run code-owned quality checks on optimized experience output.

    Expects a role-scoped OptimizedExperienceSection.
    Returns the merged ReviewResult from the diagnostics layer.
    """
    # Stage 3 / Step 3.4: Check the written role in code
    # Receives from: OptimizedExperienceSection.optimized_experiences.
    # Sends to: audit_experience_quality_for_experiences(...).
    return audit_experience_quality_for_experiences(section.optimized_experiences)


def _rewrite_experience_section_once(
    context: str,
    section: OptimizedExperienceSection,
    audit_result: ReviewResult,
) -> OptimizedExperienceSection:
    """Ask for one rewrite using the previous output and audit feedback.

    Expects a serious audit result for the first optimized section.
    Returns one rewritten OptimizedExperienceSection.
    """
    rewrite_context = _build_experience_rewrite_context(context, section, audit_result)
    return _write_experience_section(rewrite_context)


def _run_single_experience_optimization(
    resume: Resume,
    job_description: JobDescription,
    strategy: AlignmentStrategy,
    experience: Experience,
) -> OptimizedExperienceSection:
    """Optimize one experience entry with a role-scoped CrewAI call.

    Expects job_description and strategy to be present in pipeline state.
    Returns an OptimizedExperienceSection containing the optimized role entry.
    """
    role_resume = _build_resume_with_single_experience(resume, experience)
    context = format_experience_optimizer_context(
        resume=role_resume,
        job_description=job_description,
        strategy=strategy,
        format_type="toon",
    )
    optimized_section = _write_experience_section(context)
    optimized_section = _restore_original_experience_id(optimized_section, experience)
    audit_result = _audit_experience_section(optimized_section)

    # Stage 3 / Step 3.5: Decide whether one rewrite is needed
    # Receives from: ReviewResult comments from the experience quality checks.
    # Sends to: either accepted OptimizedExperienceSection or one repair CrewAI task.
    if not _experience_audit_needs_rewrite(audit_result):
        return optimized_section

    rewritten_section = _rewrite_experience_section_once(context, optimized_section, audit_result)
    return _restore_original_experience_id(rewritten_section, experience)


def _merge_optimized_experience_sections(
    sections: list[OptimizedExperienceSection],
) -> OptimizedExperienceSection:
    """Merge role-scoped optimization results into one section.

    Expects each section to contain at least one optimized experience.
    Returns one OptimizedExperienceSection for downstream ATS assembly.
    """
    optimized_experiences = []
    optimization_notes = []
    keywords_integrated = []
    relevance_scores = {}

    for section in sections:
        optimized_experiences.extend(section.optimized_experiences)
        if section.optimization_notes:
            optimization_notes.append(section.optimization_notes)
        keywords_integrated.extend(section.keywords_integrated)
        relevance_scores.update(section.relevance_scores)

    return OptimizedExperienceSection(
        optimized_experiences=optimized_experiences,
        optimization_notes="\n".join(optimization_notes),
        keywords_integrated=list(dict.fromkeys(keywords_integrated)),
        relevance_scores=relevance_scores,
    )


def _run_experience_optimization_workers(
    resume: Resume,
    job_description: JobDescription,
    strategy: AlignmentStrategy,
    experiences: list[Experience],
) -> list[OptimizedExperienceSection]:
    """Run role-scoped experience optimization calls in parallel.

    Expects a non-empty experiences list from resume.work_experience.
    Returns one OptimizedExperienceSection per input experience.
    """
    max_workers = min(len(experiences), 4)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(
            executor.map(
                lambda experience: _run_single_experience_optimization(
                    resume,
                    job_description,
                    strategy,
                    experience,
                ),
                experiences,
            )
        )


# ---------------------------------------------------------------------------
# Stage 1 -- parallel ingestion
# ---------------------------------------------------------------------------


def extract_resume(state: ResumeEnhancementPipelineState) -> dict:
    """Hand the resume file path to the resume_parser agent and receive a structured Resume.

    Reads:  resume_path (a file path to a PDF or DOCX)
    Writes: resume

    WHY the node only passes a path (not pre-converted Markdown):
      The resume_parser agent owns all 4 document-ingestion tools:
        convert_resume_document_to_markdown  -> Markdown
        check_resume_markdown_quality        -> quality gate
        redact_pii_from_resume_markdown      -> PII-safe Markdown
        extract_structured_resume_from_markdown -> Resume
      The agent reasons through them in order. The orchestration node's job
      is to hand over the file path and collect the Resume output -- nothing else.
      Pre-converting here would bypass the agent's quality gate and PII redaction.

    NOTE: tasks.yaml 'extract_resume_content_task' description must say
      "You are given a file path. Use your tools to convert, check quality,
       redact PII, and extract the resume." -- NOT "content is already provided".
    # TODO: update extract_resume_content_task in tasks.yaml to match this contract
    #       Proposed: remove the "DO NOT use file-parsing tools" instruction, replace
    #                 with "use your tools in order: convert -> quality check -> redact -> extract"
    #       Deferred: tasks.yaml edit is a separate change; behavior is correct with current agent
    """
    agent = create_resume_extractor_agent()
    resume = _run_single_agent_crew(
        agent=agent,
        task_name="extract_resume_content_task",
        # The agent receives the file path. It calls convert_resume_document_to_markdown
        # first, then decides whether the result is clean enough to extract from.
        context=f"RESUME FILE PATH: {state['resume_path']}",
        output_model=Resume,
    )
    return {"resume": assign_experience_ids(resume)}


def analyze_job(state: ResumeEnhancementPipelineState) -> dict:
    """Convert the JD file to Markdown and extract a structured JobDescription.

    Reads:  jd_path (a file path to a PDF, DOCX, or plain text file)
    Writes: job_description

    WHY this node pre-converts (unlike extract_resume):
      The Job Analyzer agent has no document-ingestion tools -- it only reasons
      over text. The orchestration node does the one mechanical step (PDF -> Markdown)
      and passes the result to the agent. This is correct because there is no quality
      gate or PII redaction required for a job description.
    """
    jd_markdown = convert_document_to_markdown(state["jd_path"])
    agent = create_job_analyzer_agent()
    job_description = _run_single_agent_crew(
        agent=agent,
        task_name="analyze_job_description_task",
        context=f"JOB DESCRIPTION:\n{jd_markdown}",
        output_model=JobDescription,
    )
    return {"job_description": job_description}


# ---------------------------------------------------------------------------
# Stage 2 -- sequential gap analysis
# ---------------------------------------------------------------------------


def run_gap_analysis(state: ResumeEnhancementPipelineState) -> dict:
    """Identify gaps between the resume and the job, producing a tailoring strategy.

    Reads:  resume, job_description   (set by Stage 1 -- both must be non-None)
    Writes: alignment_strategy        (fit score, matched/missing requirements,
                                       keyword targets, per-section guidance)

    How context reaches the agent:
      format_gap_analysis_context() converts the typed Resume + JobDescription
      objects into a TOON-formatted string (token-optimised Markdown table).
      That string becomes the CONTEXT block appended to the task description.
      The LLM reads it as part of the prompt -- it is NOT a tool call argument.

    How the tool fits in:
      The agent has one tool: match_job_requirements.
      During reasoning the agent can call it to get a structured comparison
      (requirement coverage %, keyword matches, missing must-haves) as a
      formatted report string. The agent uses that report alongside the CONTEXT
      to build the AlignmentStrategy -- it is a reasoning aid, not a data pipe.

    What comes out:
      AlignmentStrategy -- a typed Pydantic object with per-section guidance
      that the three Stage 3 agents (summary, experience, skills) each read
      from state to know what to optimise.
    """
    context = format_gap_analysis_context(
        resume=state["resume"],
        job_description=state["job_description"],
        format_type="toon",
    )
    agent = create_gap_analysis_agent()
    alignment_strategy = _run_single_agent_crew(
        agent=agent,
        task_name="create_alignment_strategy_task",
        context=context,
        output_model=AlignmentStrategy,
    )
    return {"alignment_strategy": alignment_strategy}


# ---------------------------------------------------------------------------
# Stage 3 -- parallel content generation
# ---------------------------------------------------------------------------


def write_professional_summary(state: ResumeEnhancementPipelineState) -> dict:
    """Generate a professional summary tailored to the job description.

    Reads:  resume, job_description, alignment_strategy  (all set by Stages 1-2)
    Writes: professional_summary

    How context reaches the agent:
      format_professional_summary_context() pulls two key fields from
      alignment_strategy: professional_summary_guidance (what angle to lead with)
      and keywords_to_integrate (ATS keywords the summary must include).
      These become part of the TOON-formatted CONTEXT block in the prompt,
      so the agent already knows the strategic direction before it writes a word.

    How the tool fits in:
      The agent has one tool: audit_summary.
      The agent's internal loop is: draft -> call audit_summary -> read the
      critique (length, first-person voice, boilerplate flags, missing value prop)
      -> refine. It runs this loop for multiple drafts at temperature=0.7 so
      each draft explores a different narrative frame.

    What comes out:
      ProfessionalSummary -- the agent's recommended draft plus the rejected
      alternatives. Only the recommended draft flows into Stage 4 (ATS assembly).
      The rejected drafts are preserved in the model for inspection/debugging.
    """
    context = format_professional_summary_context(
        resume=state["resume"],
        job_description=state["job_description"],
        strategy=state["alignment_strategy"],
        format_type="toon",
    )
    agent = create_professional_summary_agent()
    professional_summary = _run_single_agent_crew(
        agent=agent,
        task_name="write_professional_summary_task",
        context=context,
        output_model=ProfessionalSummary,
    )
    return {"professional_summary": professional_summary}


def optimize_experience(state: ResumeEnhancementPipelineState) -> dict:
    """Rewrite work experience bullets with one role-scoped agent call per entry.

    Reads: resume, job_description, alignment_strategy
    Writes: optimized_experience
    Precondition: resume, job_description, alignment_strategy are non-None.
    """
    resume = state["resume"]
    job_description = state["job_description"]
    strategy = state["alignment_strategy"]
    if resume is None or job_description is None or strategy is None:
        raise ValueError("resume, job_description, and alignment_strategy must be set.")

    experiences = resume.work_experience
    if not experiences:
        raise ValueError("resume.work_experience must contain at least one entry.")

    sections = _run_experience_optimization_workers(
        resume=resume,
        job_description=job_description,
        strategy=strategy,
        experiences=experiences,
    )
    optimized_experience = _merge_optimized_experience_sections(sections)
    return {"optimized_experience": optimized_experience}


def optimize_skills(state: ResumeEnhancementPipelineState) -> dict:
    """Match and optimize the skills section for ATS keyword coverage.

    Reads: resume, job_description, alignment_strategy
    Writes: optimized_skills
    Precondition: resume, job_description, alignment_strategy are non-None.
    """
    context = format_skills_optimizer_context(
        resume=state["resume"],
        job_description=state["job_description"],
        strategy=state["alignment_strategy"],
        format_type="toon",
    )
    agent = create_skills_optimizer_agent()
    optimized_skills = _run_single_agent_crew(
        agent=agent,
        task_name="optimize_skills_task",
        context=context,
        output_model=OptimizedSkillsSection,
    )
    return {"optimized_skills": optimized_skills}


# ---------------------------------------------------------------------------
# Stage 4 -- sequential ATS assembly
# ---------------------------------------------------------------------------


def assemble_ats_resume(state: ResumeEnhancementPipelineState) -> dict:
    """Assemble all optimized sections into a single ATS-compliant resume.

    Reads: professional_summary, optimized_experience, optimized_skills,
           resume, job_description
    Writes: optimized_resume
    Precondition: all Stage 3 outputs and original resume + job are non-None.
    """
    context = format_ats_optimization_context(
        professional_summary=state["professional_summary"],
        optimized_experience=state["optimized_experience"],
        optimized_skills=state["optimized_skills"],
        original_resume=state["resume"],
        job_description=state["job_description"],
        format_type="toon",
    )
    agent = create_ats_optimization_agent()
    optimized_resume = _run_single_agent_crew(
        agent=agent,
        task_name="compile_resume_task",
        context=context,
        output_model=OptimizedResume,
    )
    return {"optimized_resume": optimized_resume}


# ---------------------------------------------------------------------------
# Stage 5 -- sequential quality assurance
# ---------------------------------------------------------------------------


def run_quality_assurance(state: ResumeEnhancementPipelineState) -> dict:
    """Validate the optimized resume for quality and consistency.

    Reads: optimized_resume, resume, job_description
    Writes: qa_report
    Precondition: optimized_resume, resume, and job_description are non-None.
    """
    context = format_quality_assurance_context(
        optimized_resume=state["optimized_resume"],
        original_resume=state["resume"],
        job=state["job_description"],
        format_type="toon",
    )
    agent = create_quality_assurance_agent()
    qa_report = _run_single_agent_crew(
        agent=agent,
        task_name="quality_assurance_task",
        context=context,
        output_model=QualityReport,
    )
    return {"qa_report": qa_report}
