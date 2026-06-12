# Professional Summary Agent — Architecture

```
================================================================================
                         INTERNAL ARCHITECTURE
================================================================================

                      ┌──────────────────────────────────────────┐
                      │     PROFESSIONAL SUMMARY MODULE          │
                      │  src/agents/professional_summary/        │
                      │                                          │
  agents.yaml ────────┤  agent.py                                │
  (role, goal,        │  ┌────────────────────────────────────┐  │
   backstory, llm)    │  │  create_professional_summary_agent()│  │
                      │  │                                    │  │
                      │  │  1. _load_agent_config()           │  │
                      │  │     validates: role, goal,         │  │
                      │  │     backstory, llm MUST exist      │  │
                      │  │     Raises RuntimeError if broken  │  │
                      │  │                                    │  │
                      │  │  2. builds Agent with:             │  │
                      │  │     - LLM instance                 │  │
                      │  │     - 1 tool: audit_summary        │  │
                      │  │     - temperature 0.7 (creative)   │  │
                      │  │     - resilience params from config│  │
                      │  │                                    │  │
                      │  │  3. returns Agent                  │  │
                      │  └────────────────────────────────────┘  │
                      │                                          │
                      │  models.py                                │
                      │  ┌────────────────────────────────────┐  │
                      │  │  SummaryDraft                      │  │
                      │  │    version_name, strategy_used,    │  │
                      │  │    content, critique, score        │  │
                      │  │                                    │  │
                      │  │  ProfessionalSummary                │  │
                      │  │    drafts[], recommended_version,  │  │
                      │  │    writing_notes                   │  │
                      │  └────────────────────────────────────┘  │
                      │                                          │
                      │  engines.py                               │
                      │  ┌────────────────────────────────────┐  │
                      │  │  check_summary_quality()           │  │
                      │  │    ProfessionalSummary + Strategy  │  │
                      │  │    -> dict (per-draft scores)      │  │
                      │  │                                    │  │
                      │  │  analyze_keyword_integration()     │  │
                      │  │    str + keywords -> dict          │  │
                      │  │    (integrated, missing, rate)     │  │
                      │  └────────────────────────────────────┘  │
                      │                                          │
                      │  professional_summary_architecture.md    │
                      │  (this file)                             │
                      └──────────────────────────────────────────┘


================================================================================
                          END-TO-END DATA FLOW
================================================================================

  Resume ──────────────────┐
  JobDescription ──────────┤
  AlignmentStrategy ───────┤
                           │
  .professional_summary_guidance
  .keywords_to_integrate   │
                           ▼
                 Professional Summary Agent ──→ ProfessionalSummary
                         │                          │
                         │ calls                    ├── .drafts[]
                         ▼                          │     .version_name
               audit_summary()                      │     .strategy_used
               ┌──────────────────────┐             │     .content
               │ ZONE 2: @tool adapter│             │     .critique
               │                      │             │     .score
               │  1. Parse:           │             │
               │     Resume JSON      │             ├── .recommended_version
               │                      │             │
               │  2. Run engine:      │             └── .writing_notes
               │     summary_quality  │
               │     (hybrid)         │                   Downstream:
               │                      │                   ATS Optimization
               │  mechanical:         │                   QA Agent
               │    length check      │
               │    first-person scan │
               │                      │
               │  judgment (LLM):     │
               │    generic phrasing? │
               │    value proposition?│
               │                      │
               │  3. Render -> string │
               └──────────────────────┘


================================================================================
                      THREE-ZONE LAYERED VIEW
================================================================================

  ZONE 3: Agent Factory          (src/agents/professional_summary/agent.py)
  +---------------------------------------------------------------+
  |  create_professional_summary_agent()                          |
  |                                                               |
  |  Loads YAML config, validates, builds Agent with 1 tool.      |
  |  No business logic. Fail-fast on bad config.                  |
  +---------------------------------------------------------------+
         |                              |
         | assigns tool                 | imports config
         v                              v
  +---------------------------+   +---------------------------+
  | src/tools/                |   | src/core/config.py        |
  | agent_facing_tools.py     |   | src/config/agents.yaml    |
  |                           |   +---------------------------+
  | audit_summary()           |
  | (1 @tool function)        |
  +---------------------------+
         |
         | calls (inside the @tool)
         v
  ZONE 1: Engine                 (src/tools/resume_diagnostics/)
  +---------------------------------------------------------------+
  |  audit_summary_quality()        Resume -> ReviewResult        |
  |    Hybrid: length + 1st-person (mechanical, HIGH conf)        |
  |          + generic/value proposition (LLM, MEDIUM conf)       |
  +---------------------------------------------------------------+


================================================================================
                     DOWNSTREAM CONSUMERS MAP
================================================================================

  ProfessionalSummary
         │
         ├── .drafts[].content ──────────────► ATS Optimization (assembly)
         │
         ├── .recommended_version ───────────► ATS Optimization (pick winner)
         │
         ├── .drafts[].critique ─────────────► QA Agent (context)
         │
         └── .writing_notes ────────────────► QA Agent (context)


================================================================================
                       IMPORT DEPENDENCY GRAPH
================================================================================

  agent.py
     │
     ├──── from crewai import Agent, LLM
     │
     ├──── from src.core.settings import get_agents_config, get_config
     │
     ├──── from src.core.logger import get_logger
     │
     └──── from src.tools.agent_facing_tools import audit_summary

  engines.py
     │
     ├──── from src.agents.professional_summary.models import ProfessionalSummary
     │
     └──── from src.data_models.strategy import AlignmentStrategy

  models.py
     │
     └──── from pydantic import BaseModel, ConfigDict, Field


================================================================================
                       DESIGN DECISIONS
================================================================================

  1. WHY audit_summary, NOT DraftEvaluationTool
     The old code had an in-class @tool (DraftEvaluationTool) that did
     mechanical checks only (word count, keyword grep, cliché scan).
     It was never wired (tools=[]). audit_summary replaces it with
     a proper hybrid tool: mechanical checks + LLM judgment on whether
     the summary is generic or has a real value proposition.

  2. WHY temperature 0.7
     Summary writing is creative work — the agent needs to produce
     4 genuinely different drafts using different narrative frameworks.
     Higher temperature enables variety. This comes from config, not
     hardcoded.

  3. WHY models.py IS IN the module, NOT data_models
     ProfessionalSummary + SummaryDraft are the output contracts of
     THIS agent. They should eventually move to src/data_models/ when
     the migration wires this module into the orchestrator. Until then,
     keeping them in-module follows the scope rule: touch nothing
     outside the module directory.

  4. WHY check_summary_quality NEEDS strategy as input
     The quality of a summary IS relative to the job it's targeting.
     A summary that's excellent for a startup role may miss keywords
     critical for an enterprise role. The strategy provides the standard.

  5. WHY ONE TOOL
     audit_summary covers the single quality dimension this agent
     needs feedback on: "is this summary any good?" One composite
     tool = the agent focuses on writing, not tool orchestration.


================================================================================
                  HOW TO CONTROL OUTPUT QUALITY
================================================================================

  Quality is controlled in TWO places. Change the right one for what you need.


  ┌─────────────────────────────────────────────────────────────────────┐
  │  LEVER 1: Writing guidelines (WHAT to write)                       │
  │  src/config/tasks.yaml  —  write_professional_summary_task         │
  ├─────────────────────────────────────────────────────────────────────┤
  │                                                                     │
  │  Controls:                                                          │
  │    - Draft frameworks (Hook-Value-Future, Story Spine, etc.)        │
  │    - Word count targets (75-150 words)                              │
  │    - Internal quality tests (specificity test, energy audit)        │
  │    - Self-critique instructions                                     │
  │                                                                     │
  │  Change when you want:                                              │
  │    - Different drafting strategies                                  │
  │    - Different word count range                                     │
  │    - Different self-review criteria                                 │
  │                                                                     │
  │  Example changes:                                                   │
  │    description: >                                                   │
  │      Generate 3 drafts (was 4) using these frameworks:              │
  │      1. Hook-Value-Future                                           │
  │      2. Problem-Solution-Result                                     │
  │      3. ATS-Optimized                                               │
  │      Each draft must be 50-100 words (was 75-150).                  │
  │      Run a "Truthfulness Check" before scoring.                     │
  │                                                                     │
  └─────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────┐
  │  LEVER 2: Tool-based audit (HOW to evaluate)                       │
  │  src/tools/resume_diagnostics/summary_quality_auditor.py            │
  ├─────────────────────────────────────────────────────────────────────┤
  │                                                                     │
  │  Controls (runtime — agent calls audit_summary tool):               │
  │                                                                     │
  │    Mechanical checks (HIGH confidence, free):                       │
  │      MIN_SUMMARY_WORDS = 50    ← floor: flags if shorter            │
  │      MAX_SUMMARY_WORDS = 150   ← ceiling: flags if longer           │
  │      FIRST_PERSON_PRONOUNS     ← flags "I", "my", "me" etc.        │
  │                                                                     │
  │    Judgment check (LLM, MEDIUM confidence):                         │
  │      SUMMARY_RUBRIC  ← prompt that defines "generic" and            │
  │                         "value proposition" for the LLM auditor     │
  │                                                                     │
  │  Change when you want:                                              │
  │    - Stricter/looser length limits                                  │
  │    - Different banned pronouns or phrases                           │
  │    - New quality dimensions (passive voice, keyword presence)       │
  │                                                                     │
  │  Example changes:                                                   │
  │    MIN_SUMMARY_WORDS = 75     # was 50 — stricter floor             │
  │    MAX_SUMMARY_WORDS = 120    # was 150 — tighter ceiling           │
  │                                                                     │
  │    SUMMARY_RUBRIC = """                                             │
  │      Also check:                                                    │
  │      3. Passive voice: flag "was responsible for", "was involved"   │
  │      4. Missing keywords: flag if <3 keywords from the target       │
  │         job appear in the summary.                                  │
  │    """                                                              │
  │                                                                     │
  └─────────────────────────────────────────────────────────────────────┘


  ┌─────────────────────────────────────────────────────────────────────┐
  │  NOT a runtime lever (post-hoc only):                               │
  │  src/agents/professional_summary/engines.py                         │
  ├─────────────────────────────────────────────────────────────────────┤
  │  check_summary_quality() and analyze_keyword_integration()          │
  │  validate the agent's OUTPUT in tests/QA. They do NOT affect        │
  │  what the agent produces at runtime. Change these when you want     │
  │  different pass/fail criteria in your test suite.                   │
  └─────────────────────────────────────────────────────────────────────┘
```
