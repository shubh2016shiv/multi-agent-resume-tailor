# Gap Analysis Agent — Architecture

```
================================================================================
                         INTERNAL ARCHITECTURE
================================================================================

                      ┌──────────────────────────────────────────┐
                      │        GAP ANALYSIS MODULE               │
                      │  src/agents/gap_analysis/                │
                      │                                          │
  agents.yaml ────────┤  agent.py                                │
  (role, goal,        │  ┌────────────────────────────────────┐  │
   backstory, llm)    │  │  create_gap_analysis_agent()       │  │
                      │  │                                    │  │
                      │  │  1. _load_agent_config()           │  │
                      │  │     validates: role, goal,         │  │
                      │  │     backstory, llm MUST exist      │  │
                      │  │     Raises RuntimeError if broken  │  │
                      │  │                                    │  │
                      │  │  2. builds Agent with:             │  │
                      │  │     - LLM instance                 │  │
                      │  │     - 1 tool: match_job_requirements│  │
                      │  │     - resilience params from config│  │
                      │  │                                    │  │
                      │  │  3. returns Agent                  │  │
                      │  └────────────────────────────────────┘  │
                      │                                          │
                      │  engines.py                               │
                      │  ┌────────────────────────────────────┐  │
                      │  │  check_strategy_quality()          │  │
                      │  │    AlignmentStrategy -> dict       │  │
                      │  │    (score, issues, warnings)       │  │
                      │  │                                    │  │
                      │  │  calculate_coverage_stats()        │  │
                      │  │    AlignmentStrategy -> dict       │  │
                      │  │    (matches, gaps, ratio)          │  │
                      │  └────────────────────────────────────┘  │
                      │                                          │
                      │  gap_analysis_architecture.md            │
                      │  (this file)                             │
                      └──────────────────────────────────────────┘


================================================================================
                          END-TO-END DATA FLOW
================================================================================

   Resume ─────────────┐
                       ├──→ Gap Analysis Agent ──→ AlignmentStrategy
   JobDescription ─────┘        │                      │
                                │ calls                ├──→ Summary Writer
                                ▼                      │     .professional_summary_guidance
                      match_job_requirements()         │
                      ┌─────────────────────────┐      ├──→ Experience Optimizer
                      │ ZONE 2: @tool adapter    │      │     .experience_guidance
                      │                          │      │
                      │  1. Parse:               │      └──→ Skills Optimizer
                      │     Resume JSON +         │            .skills_guidance
                      │     JobDescription JSON   │
                      │                          │            ALSO provides:
                      │  2. Run 2 engines:        │            .overall_fit_score
                      │     ├─ match_requirements │            .identified_matches[]
                      │     │  (LLM judgment)     │            .identified_gaps[]
                      │     │  "does Flask cover  │            .keywords_to_integrate[]
                      │     │   FastAPI req?"     │            .summary_of_strategy
                      │     │                     │
                      │     └─ keyword_coverage   │
                      │        (mechanical)       │
                      │        "is 'K8s' literally │
                      │         in the text?"     │
                      │                          │
                      │  3. Merge → ReviewResult  │
                      │  4. Render → agent string │
                      └──────────────────────────┘
                                │
                                │ evidence report (str)
                                ▼
                      ┌─────────────────────────┐
                      │ ZONE 3: Agent (LLM)     │
                      │                         │
                      │  Reasons over:          │
                      │  - prompt context       │
                      │    (formatted Resume     │
                      │     + JobDescription)   │
                      │  - tool evidence        │
                      │    (requirement matches, │
                      │     gaps, keywords)      │
                      │                         │
                      │  Produces:              │
                      │    AlignmentStrategy    │
                      │    (output_pydantic)    │
                      └─────────────────────────┘


================================================================================
                      THREE-ZONE LAYERED VIEW
================================================================================

  ZONE 3: Agent Factory          (src/agents/gap_analysis/agent.py)
  +---------------------------------------------------------------+
  |  create_gap_analysis_agent()                                  |
  |                                                               |
  |  Loads YAML config, validates, builds Agent with 1 tool.      |
  |  No business logic. Fail-fast on bad config.                  |
  +---------------------------------------------------------------+
         |                              |
         | assigns tools                | imports config
         v                              v
  +---------------------------+   +---------------------------+
  | src/tools/                |   | src/core/config.py        |
  | agent_facing_tools.py     |   | src/config/agents.yaml    |
  |                           |   +---------------------------+
  | match_job_requirements()  |
  | (1 @tool function)        |
  +---------------------------+
         |
         | calls (inside the @tool)
         v
  ZONE 1: Engines                (src/tools/job_matching/)
  +---------------------------------------------------------------+
  |  match_requirements()           Resume+Job -> ReviewResult    |
  |    LLM judgment: classifies each requirement as               |
  |    matched / partial / gap. Score = must-have coverage.       |
  |                                                               |
  |  analyze_keyword_coverage()     text+keywords -> ReviewResult |
  |    Mechanical: exact token match. Coverage %, absent list.    |
  +---------------------------------------------------------------+


================================================================================
                     DOWNSTREAM CONSUMERS MAP
================================================================================

  AlignmentStrategy
         │
         ├── .overall_fit_score ────────────► QA Agent (context)
         │
         ├── .summary_of_strategy ──────────► QA Agent (context)
         │
         ├── .identified_matches[] ─────────► Logging / monitoring
         │    .resume_skill
         │    .job_requirement
         │    .match_score
         │    .justification
         │
         ├── .identified_gaps[] ────────────► All downstream agents (context)
         │    .missing_skill
         │    .importance
         │    .suggestion
         │
         ├── .keywords_to_integrate[] ──────► Summary Writer
         │                                     Experience Optimizer
         │                                     Skills Optimizer
         │
         ├── .professional_summary_guidance ─► Summary Writer
         │
         ├── .experience_guidance ───────────► Experience Optimizer
         │
         └── .skills_guidance ───────────────► Skills Optimizer


================================================================================
                       IMPORT DEPENDENCY GRAPH
================================================================================

  agent.py
     │
     ├──── from crewai import Agent, LLM
     │
     ├──── from src.core.config import get_agents_config, get_config
     │
     ├──── from src.core.logger import get_logger
     │
     └──── from src.tools.agent_facing_tools import match_job_requirements

  engines.py
     │
     └──── from src.data_models.strategy import AlignmentStrategy

  RULE: Zone 3 (agent) imports Zone 2 (tool adapter).
        Zone 2 imports Zone 1 (engines).
        Zone 1 imports NOTHING from Zone 2 or Zone 3.


================================================================================
                       DESIGN DECISIONS
================================================================================

  1. WHY ONE TOOL, NOT ZERO
     match_job_requirements bundles the two things the agent needs:
     evidence-based requirement matching (LLM judgment) and literal
     keyword scanning (mechanical). Without this tool the agent is
     guessing — with it, the agent reasons over real data.

  2. WHY ONE TOOL, NOT MULTIPLE
     match_job_requirements already composites two engines via _merge().
     Adding more tools would force the agent to orchestrate tool calls
     instead of focusing on strategy. The tool provides the evidence;
     the agent provides the reasoning. One call, one report.

  3. WHY ENGINES ARE POST-HOC, NOT RUNTIME
     check_strategy_quality and calculate_coverage_stats validate
     the agent's OUTPUT — they are test utilities and QA checks.
     They are never called during agent execution. The agent itself
     produces the strategy; these functions answer "was it any good?"

  4. WHY NO PIPELINE LAYER
     The engines are importable directly from src/tools/. The tool
     adapter is importable from src/tools/agent_facing_tools.py.
     A third wrapper layer between them adds indirection with no
     added logic. Two layers is the production standard.

  5. WHY FAIL-FAST CONFIG
     _load_agent_config validates all 4 required fields at creation
     time. No silent fallback to defaults with a missing LLM field
     that causes a confusing failure 3 steps later. Broken config
     means a clear RuntimeError immediately.
```
