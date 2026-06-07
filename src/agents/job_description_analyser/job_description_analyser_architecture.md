# Job Description Analyst — Architecture

```
================================================================================
                         INTERNAL ARCHITECTURE
================================================================================

                      ┌──────────────────────────────────────────┐
                      │     JOB DESCRIPTION ANALYSER MODULE      │
                      │  src/agents/job_description_analyser/     │
                      │                                          │
  agents.yaml ────────┤  agent.py                                │
  (role, goal,        │  ┌────────────────────────────────────┐  │
   backstory, llm)    │  │  create_job_analyzer_agent()       │  │
                      │  │                                    │  │
                      │  │  1. _load_agent_config()           │  │
                      │  │     validates: role, goal,         │  │
                      │  │     backstory, llm MUST exist      │  │
                      │  │     Raises RuntimeError if broken  │  │
                      │  │                                    │  │
                      │  │  2. _build_runtime_limits()        │  │
                      │  │     merges agent config with       │  │
                      │  │     global defaults (max_rpm etc.) │  │
                      │  │                                    │  │
                      │  │  3. builds Agent with:             │  │
                      │  │     - LLM instance                 │  │
                      │  │     - ZERO runtime tools           │  │
                      │  │     - resilience params            │  │
                      │  │                                    │  │
                      │  │  4. returns Agent                  │  │
                      │  └────────────────────────────────────┘  │
                      │                                          │
                      │  job_description_analyser_architecture.md│
                      │  (this file)                             │
                      └──────────────────────────────────────────┘


================================================================================
                          END-TO-END DATA FLOW
================================================================================

  LANGGRAPH                    ORCHESTRATION NODE              MODULE
  ═════════                    ══════════════════              ══════

  pipeline state                src/orchestration/
  (jd_path)                     nodes/ingestion.py
       │                              │
       │  state["jd_path"]            │  1. convert_document_to_markdown()
       ├─────────────────────────────►│     (code — no LLM)
       │                              │
       │                              │  2. create_job_analyzer_agent()
       │                              │     ──────────────────────────►
       │                              │     agent ◄──────────────────
       │                              │
       │                              │  3. run_agent_task(
       │                              │       agent,
       │                              │       "analyze_job_description_task",
       │                              │       context=jd_markdown,
       │                              │       output_model=JobDescription
       │                              │     )
       │                              │          │
       │                              │          ▼
       │                              │   ┌─────────────────────────┐
       │                              │   │  CrewAI Agent (LLM)     │
       │                              │   │  ZERO runtime tools     │
       │                              │   │  Reasons over:          │
       │                              │   │  - jd_markdown in ctx   │
       │                              │   │  - task instructions    │
       │                              │   │  - agent persona        │
       │                              │   │  Produces:              │
       │                              │   │    JobDescription       │
       │                              │   │    (output_pydantic)    │
       │                              │   └─────────────────────────┘
       │         JobDescription       │
       │◄─────────────────────────────│  returns state["job_description"]
       │
       ▼
  Downstream nodes:
    ├── strategy.py     → run_gap_analysis()
    ├── experience.py   → optimize_experience()
    ├── skills.py       → optimize_skills()
    └── summary.py      → write_professional_summary()


================================================================================
                      MODULE STRUCTURE
================================================================================

  src/agents/job_description_analyser/
  ├── agent.py                  ← factory only: create_job_analyzer_agent()
  ├── __init__.py               ← exports: create_job_analyzer_agent
  └── job_description_analyser_architecture.md  ← this file

  Crew/Task execution lives in:
    src/orchestration/crew_task_execution.py → run_agent_task()
    src/orchestration/nodes/ingestion.py     → analyze_job() node


================================================================================
                       IMPORT DEPENDENCY GRAPH
================================================================================

  agent.py
     │
     ├──── from crewai import LLM, Agent
     │
     ├──── from src.core.config import get_agents_config, get_config
     │
     └──── from src.core.logger import get_logger

  RULE: agent.py imports config + framework only.
        It does NOT import data models, tools, or orchestration.
        Crew/Task construction belongs in src/orchestration/.


================================================================================
                       DESIGN DECISIONS
================================================================================

  1. WHY FACTORY ONLY — NO CREW/TASK IN THIS MODULE
     Crew and Task construction belong in the orchestration layer
     (src/orchestration/crew_task_execution.py). The agent module has one
     concern: build and return a configured Agent. This matches the pattern
     used by resume_parser, professional_experience, and skill_optimizer.

  2. WHY ZERO AGENT TOOLS
     The agent is a pure extraction LLM. The orchestration node converts the
     job document to Markdown before calling the agent, and the Markdown
     arrives as task context. No file parsing, no document conversion, no
     external lookups are needed at agent runtime.

  3. WHY FAIL-FAST CONFIG
     _load_agent_config validates all 4 required fields (role, goal,
     backstory, llm) at agent creation time. No silent fallback to defaults
     that causes a confusing failure 3 steps later. Broken config raises
     a clear RuntimeError immediately.

  4. WHY _build_runtime_limits IS A SEPARATE HELPER
     CrewAI runtime limits (max_rpm, max_iter, max_execution_time, etc.) come
     from two sources: per-agent overrides in agents.yaml and global defaults
     in settings.yaml. The helper merges them cleanly and keeps the factory
     function readable.


================================================================================
                  HOW TO CONTROL OUTPUT QUALITY
================================================================================

  Quality is controlled in TWO places. Change the right one for what you need.


  ┌─────────────────────────────────────────────────────────────────────┐
  │  LEVER 1: Extraction instructions (WHAT to extract)                 │
  │  src/config/tasks.yaml  —  analyze_job_description_task             │
  ├─────────────────────────────────────────────────────────────────────┤
  │                                                                     │
  │  Controls:                                                          │
  │    - What fields to extract and their structure                     │
  │    - How to categorise requirements (must-have / should-have /      │
  │      nice-to-have)                                                  │
  │    - How to detect experience level (years, title patterns)         │
  │    - What counts as an ATS keyword                                  │
  │                                                                     │
  │  Change when you want:                                              │
  │    - Different categorisation rules                                 │
  │    - More/fewer fields in the output                                │
  │    - Different extraction heuristics                                │
  └─────────────────────────────────────────────────────────────────────┘


  ┌─────────────────────────────────────────────────────────────────────┐
  │  LEVER 2: Agent persona (HOW the agent thinks)                      │
  │  src/config/agents.yaml  —  job_description_analyst                 │
  ├─────────────────────────────────────────────────────────────────────┤
  │                                                                     │
  │  Controls:                                                          │
  │    - temperature (0.2 default: low for consistent extraction)       │
  │    - llm model (affects extraction quality and cost)                │
  │    - role/goal/backstory (shape the agent's reasoning persona)      │
  │                                                                     │
  │  Change when you want:                                              │
  │    - Higher temperature (more creative extraction, riskier)         │
  │    - Different model (better extraction vs cost tradeoff)           │
  │    - Different persona (e.g., industry-specific analyst)            │
  └─────────────────────────────────────────────────────────────────────┘
```
