# Quality Assessment — Architecture

```
================================================================================
                         INTERNAL ARCHITECTURE
================================================================================

                      +------------------------------------------+
                      |        QUALITY ASSESSMENT MODULE         |
                      |  src/agents/quality_assessment/          |
                      |                                          |
  agents.yaml --------+  agent.py                                |
  (quality_assurance_ |  +------------------------------------+  |
   reviewer:          |  |  create_quality_assessment_agent() |  |
   role, goal,        |  |                                    |  |
   backstory, llm)    |  |  1. _load_agent_config()           |  |
                      |  |     validates role, goal,          |  |
                      |  |     backstory, llm MUST exist      |  |
                      |  |                                    |  |
                      |  |  2. builds Agent with:             |  |
                      |  |     - LLM instance                 |  |
                      |  |     - _QA_TOOLS (3 read-only       |  |
                      |  |       audit tools, 1 per dimension)|  |
                      |  |     - resilience defaults          |  |
                      |  |                                    |  |
                      |  |  3. returns Agent                  |  |
                      |  +------------------------------------+  |
                      |                                          |
                      |  engines.py -> apply_quality_gate()      |
                      |              -> should_render_resume()   |
                      |  quality_assessment_architecture.md (this)|
                      +------------------------------------------+

  _QA_TOOLS  (consulted WHILE reasoning; each grounds one report dimension):
    - audit_truthfulness          -> accuracy   (original vs revised resume)
    - validate_ats_compliance     -> ats        (formatting + section headers)
    - analyze_jd_keyword_coverage -> relevance  (job keyword coverage + density)

  OUTPUT CONTRACT: QualityReport  (REUSED from src/data_models/evaluation.py L187).
  No models.py in this module -- the contract already existed; we do not redefine it.


================================================================================
                          END-TO-END DATA FLOW
================================================================================

  UPSTREAM                       CONTEXT                         MODULE
  ========                       =======                         ======

  AtsOptimizedResume.final_resume ─┐
  Resume (original)               ─┼──► QA context (optimized + original + job)
  JobDescription                  ─┘
                                       │
                                       ▼
                              run_agent_task(
                                agent,
                                "assess_quality_task",
                                context=qa_context,
                                output_model=QualityReport
                              )
                                       │
                                       ▼
                       +-------------------------------------+
                       |  CrewAI Agent (LLM)                 |
                       |  TOOLS: audit_truthfulness,         |
                       |         validate_ats_compliance,    |
                       |         analyze_jd_keyword_coverage |
                       |  Scores accuracy / relevance / ats, |
                       |  blends -> overall_quality_score    |
                       |  Produces: QualityReport            |
                       +-------------------------------------+
                                       │
                                       ▼
                       +-------------------------------------+
                       |  engines.apply_quality_gate(report) |
                       |  (CODE, deterministic, no LLM)      |
                       |  passed_quality_threshold =         |
                       |    overall_quality_score >= 80      |
                       +-------------------------------------+
                                       │
                                       ▼
                            authoritative QualityReport


================================================================================
                  THE RENDER GATE (why this module exists last)
================================================================================

  Quality Assessment is the LAST agent in the pipeline. Its code-owned boolean is
  the single trigger for PDF rendering -- not a heuristic, one comparison:

      run_quality_assessment
            │
            ▼
      engines.apply_quality_gate   (passed = score >= 80)
            │
            ├── should_render_resume(report) == True  ─► render_final_resume ─► END
            │                                            (plain code node:
            │                                             render_resume_document(
            │                                               final_resume, out.pdf))
            │
            └── should_render_resume(report) == False ─► END (no PDF)
                                                         surface feedback_for_improvement

  The render node is PLAIN CODE, not a CrewAI agent -- rendering (Resume -> LaTeX ->
  PDF via src/tools/document_rendering) is mechanical, with no judgment to make.


================================================================================
                      MODULE STRUCTURE
================================================================================

  src/agents/quality_assessment/
  ├── agent.py                            <- factory only: create_quality_assessment_agent()
  ├── engines.py                          <- code-owned render gate (apply_quality_gate, should_render_resume)
  ├── __init__.py                         <- exports: create_quality_assessment_agent
  └── quality_assessment_architecture.md  <- this file

  NO models.py: the output contract QualityReport is reused from
  src/data_models/evaluation.py (AccuracyMetrics, RelevanceMetrics, ATSMetrics).


================================================================================
                       IMPORT DEPENDENCY GRAPH
================================================================================

  agent.py
     ├── from crewai import LLM, Agent
     ├── from src.core.settings import get_agents_config, get_config
     ├── from src.core.logger import get_logger
     └── from src.tools.agent_facing_tools import
              audit_truthfulness, validate_ats_compliance, analyze_jd_keyword_coverage

  engines.py
     └── from src.data_models.evaluation import QualityReport

  RULE: agent.py imports config + framework + agent-facing tools only.
        engines.py imports the contract only. Neither imports orchestration.


================================================================================
                       DESIGN DECISIONS
================================================================================

  1. THREE TOOLS, ONE PER REPORT DIMENSION
     QualityReport has exactly three scored dimensions (accuracy, relevance, ats).
     Each gets one grounding tool so the agent's scores are backed by a real check,
     not unaided impression. Three tools is justified by three distinct KINDS of
     evidence (lessons Rule 5), and maps 1:1 to the contract.

  2. CODE-OWNED PASS/FAIL GATE
     The agent produces scores; engines.apply_quality_gate sets
     passed_quality_threshold = overall_quality_score >= 80 deterministically. The
     render trigger reads a code-computed boolean, never an LLM self-assessment.
     This is the "agents reason, code measures" split applied to the final gate.

  3. REUSE THE CONTRACT, DO NOT REDEFINE
     QualityReport already exists in data_models. Adding a models.py would create a
     shadow contract (Principle 8 / Reuse-Over-Rewrite). The module imports it.

  4. CONFIG FAIL-FAST
     _load_agent_config validates role, goal, backstory, llm at creation time and
     raises RuntimeError on any missing field.


================================================================================
                  WIRING (NOT DONE — written plan only)
================================================================================

  Proves itself in isolation via trigger_quality_assessment.py. To wire in later:

    1. src/orchestration/nodes/quality.py
         - import create_quality_assessment_agent from src.agents.quality_assessment
         - import apply_quality_gate from src.agents.quality_assessment.engines
         - run_agent_task(..., "assess_quality_task", output_model=QualityReport)
         - report = apply_quality_gate(report) before returning it
         - update format_quality_assurance_context to accept the new optimized type
           (it currently takes the old monolith OptimizedResume).

    2. src/orchestration/graph.py  (the render gate)
         - add a plain-code node render_final_resume that calls
           src.tools.document_rendering.render_resume_document(final_resume, out_path)
         - replace the run_quality_assurance -> END edge with a conditional edge keyed
           on should_render_resume(report): True -> render_final_resume -> END,
           False -> END.

    3. Delete the old monolith src/agents/quality_assurance_agent.py once unused.
```
