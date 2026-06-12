# ATS Optimizer — Architecture

```
================================================================================
                         INTERNAL ARCHITECTURE
================================================================================

                      +------------------------------------------+
                      |          ATS OPTIMIZER MODULE            |
                      |  src/agents/ats_optimizer/               |
                      |                                          |
  agents.yaml --------+  agent.py                                |
  (ats_optimization_  |  +------------------------------------+  |
   specialist:        |  |  create_ats_optimizer_agent()      |  |
   role, goal,        |  |                                    |  |
   backstory, llm)    |  |  1. _load_agent_config()           |  |
                      |  |     validates role, goal,          |  |
                      |  |     backstory, llm MUST exist      |  |
                      |  |     Raises RuntimeError if broken  |  |
                      |  |                                    |  |
                      |  |  2. builds Agent with:             |  |
                      |  |     - LLM instance                 |  |
                      |  |     - _ATS_TOOLS (2 read-only      |  |
                      |  |       audit tools)                 |  |
                      |  |     - resilience defaults          |  |
                      |  |                                    |  |
                      |  |  3. returns Agent                  |  |
                      |  +------------------------------------+  |
                      |                                          |
                      |  models.py    -> AtsOptimizedResume      |
                      |  engines.py   -> check_ats_quality()     |
                      |  ats_optimizer_architecture.md (this)    |
                      +------------------------------------------+

  _ATS_TOOLS (consulted WHILE reasoning, never mutate the resume):
    - validate_ats_compliance      (ats_formatting + section_headers, mechanical)
    - analyze_jd_keyword_coverage  (keyword coverage + density, mechanical)


================================================================================
                          END-TO-END DATA FLOW
================================================================================

  UPSTREAM OUTPUTS              FORMATTER                       MODULE
  ===============              =========                       ======

  ProfessionalSummary  ─┐
  OptimizedExperience  ─┤
  OptimizedSkills      ─┼──► format_ats_optimization_context()  (code, no LLM)
  Resume (original)    ─┤      filters each input to the
  JobDescription       ─┘      assembly-required fields, TOON
                                       │
                                       ▼
                              run_agent_task(
                                agent,
                                "optimize_ats_resume_task",
                                context=ats_context,
                                output_model=AtsOptimizedResume
                              )
                                       │
                                       ▼
                       +-------------------------------------+
                       |  CrewAI Agent (LLM)                 |
                       |  TOOLS: validate_ats_compliance,    |
                       |         analyze_jd_keyword_coverage |
                       |  ASSEMBLES one Resume from sections |
                       |  STANDARDIZES headers, orders them  |
                       |  VERIFIES keywords via tools        |
                       |  Produces:                          |
                       |    AtsOptimizedResume               |
                       |    (final_resume + decision notes)  |
                       +-------------------------------------+
                                       │
                                       ▼
                       +-------------------------------------+
                       |  engines.check_ats_quality(         |
                       |    optimized, job)  (CODE, no LLM)  |
                       |  renders final_resume -> text,      |
                       |  runs formatting + header + keyword |
                       |  engines, returns a scored report   |
                       +-------------------------------------+

  KEY SPLIT: the agent REASONS (assemble, standardize, order, verify).
             code MEASURES (render, score, flag). The LLM never emits the
             rendered markdown/JSON or self-grades a validation report.


================================================================================
                      MODULE STRUCTURE
================================================================================

  src/agents/ats_optimizer/
  ├── agent.py                       <- factory only: create_ats_optimizer_agent()
  ├── models.py                      <- output contract: AtsOptimizedResume
  ├── engines.py                     <- post-hoc validator: check_ats_quality()
  ├── __init__.py                    <- exports: create_ats_optimizer_agent
  └── ats_optimizer_architecture.md  <- this file

  Crew/Task execution lives in:
    src/orchestration/crew_task_execution.py -> run_agent_task()
    src/orchestration/nodes/assembly.py      -> assemble_ats_resume() node
                                                (still wired to the OLD monolith;
                                                 see "Wiring" below)


================================================================================
                       IMPORT DEPENDENCY GRAPH
================================================================================

  agent.py
     ├── from crewai import LLM, Agent
     ├── from src.core.settings import get_agents_config, get_config
     ├── from src.core.logger import get_logger
     └── from src.tools.agent_facing_tools import
              validate_ats_compliance, analyze_jd_keyword_coverage

  models.py
     └── from src.data_models.resume import Resume

  engines.py
     ├── from src.agents.ats_optimizer.models import AtsOptimizedResume
     ├── from src.data_models.job import JobDescription
     ├── from src.tools.ats_compliance import audit_ats_formatting, audit_section_headers
     ├── from src.tools.job_matching import analyze_keyword_coverage
     ├── from src.tools.review_contract.review_models import ReviewResult, Severity
     └── from src.tools.shared.resume_rendering import render_resume

  RULE: agent.py imports config + framework + agent-facing tools only.
        engines.py imports the mechanical tool engines for code-owned scoring.
        Neither imports orchestration.


================================================================================
                       DESIGN DECISIONS
================================================================================

  1. SLIM OUTPUT — AGENT REASONS, CODE MEASURES
     AtsOptimizedResume carries the assembled final_resume plus decision notes
     (section_order, optimization_summary, keyword_integration_notes,
     unresolved_issues). It does NOT carry rendered markdown, rendered JSON, or
     a self-scored ATS report. Rendering and scoring are mechanical and
     deterministic — they belong in code (engines.check_ats_quality), not in an
     LLM that would self-grade. This is the opposite of the old monolith's
     OptimizedResume, which asked the LLM to emit all three.

  2. TWO READ-ONLY AUDIT TOOLS
     The agent is handed validate_ats_compliance and analyze_jd_keyword_coverage
     so it can check its own assembly against the real ATS rules WHILE reasoning.
     These tools only measure; they never rewrite the resume. The tool catalog
     (tool_documentations/README.md) maps both to "ATS Optimization".

  3. NO FABRICATION
     The agent only reorders, standardizes, and surfaces content the optimized
     sections already contain. A required keyword with no truthful support goes
     into unresolved_issues — it is never invented into the resume.

  4. CONFIG FAIL-FAST
     _load_agent_config validates role, goal, backstory, llm at creation time.
     Missing config raises RuntimeError immediately, not three steps later.


================================================================================
                  WIRING (NOT DONE — written plan only)
================================================================================

  This module proves itself in isolation via trigger_ats_optimizer.py. It is NOT
  yet wired into the LangGraph pipeline. To wire it in later (separate task):

    1. src/orchestration/nodes/assembly.py
         - import create_ats_optimizer_agent from src.agents.ats_optimizer
         - import AtsOptimizedResume from src.agents.ats_optimizer.models
         - call run_agent_task(..., "optimize_ats_resume_task",
                               output_model=AtsOptimizedResume)
         - fix the task name: assembly.py currently passes "compile_resume_task",
           which does not exist in tasks.yaml. Use "optimize_ats_resume_task".
         - (optional) run engines.check_ats_quality post-hoc, mirroring the
           code-owned audit in nodes/skills.py.

    2. src/orchestration/state.py
         - change optimized_resume to AtsOptimizedResume from the new module.

    3. Delete the old monolith src/agents/ats_optimization_agent.py once nothing
       imports it.

  Until then, the old monolith stays untouched and the pipeline keeps using it.
```
