# Skill Optimizer Agent — Architecture

```
================================================================================
                         INTERNAL ARCHITECTURE
================================================================================

                      ┌──────────────────────────────────────────┐
                      │        SKILL OPTIMIZER MODULE            │
                      │  src/agents/skill_optimizer/             │
                      │                                          │
  skill_optimizer.yaml┤  agent.py                                │
  (role, goal,        │  ┌────────────────────────────────────┐  │
   backstory, llm)    │  │  create_skill_optimizer_agent()    │  │
                      │  │                                    │  │
                      │  │  1. load_agent_config()             │  │
                      │  │     validates: role, goal,         │  │
                      │  │     backstory, llm MUST exist      │  │
                      │  │                                    │  │
                      │  │  2. builds Agent with:             │  │
                      │  │     - LLM instance                 │  │
                      │  │     - tools=[] (audit code-owned)  │  │
                      │  │     - temperature 0.4              │  │
                      │  │     - resilience params from config│  │
                      │  │                                    │  │
                      │  │  3. returns Agent                  │  │
                      │  └────────────────────────────────────┘  │
                      │                                          │
                      │  skill_optimizer_architecture.md         │
                      │  (this file)                             │
                      └──────────────────────────────────────────┘


================================================================================
                          END-TO-END DATA FLOW
================================================================================

  Resume ──────────────────┐
  JobDescription ──────────┤
  AlignmentStrategy ───────┤
                           │
  .skills_guidance         │
  .keywords_to_integrate   │
                           ▼
              Skill Optimizer Agent ──→ OptimizedSkillsSection
              (tools=[])                      │
                                              ├── .optimized_skills[]
                                              │     (reordered, prioritized)
                                              │
                                              ├── .skill_categories{}
                                              │     (domain-appropriate groups)
                                              │
                                              ├── .added_skills[]
                                              │     (inferred with justification)
                                              │
                                              ├── .removed_skills[]
                                              │
                                              ├── .optimization_notes
                                              │
                                              └── .ats_match_score

                     │
                     │ code-owned audit (after agent finishes)
                     ▼
            check_skills_evidence engine
            ┌──────────────────────────────┐
            │ src/tools/truthfulness/      │
            │                              │
            │ Judgment (LLM):              │
            │   Is every listed skill      │
            │   backed by evidence in     │
            │   the resume?               │
            │                              │
            │ Returns: ReviewResult        │
            │   with per-skill findings    │
            └──────────────────────────────┘
                     │
                     ▼
            Orchestration decides:
            accept or one rewrite


================================================================================
                      THREE-ZONE LAYERED VIEW
================================================================================

  ZONE 3: Agent Factory          (src/agents/skill_optimizer/agent.py)
  +---------------------------------------------------------------+
  |  create_skill_optimizer_agent()                               |
  |                                                               |
  |  Loads YAML config, validates, builds Agent. tools=[].        |
  |  No business logic. Fail-fast on bad config.                  |
  +---------------------------------------------------------------+
         |
         | (no tools assigned — audit runs code-owned)
         |
         v
  ZONE 1: Engine                  (src/tools/engines/truthfulness/skills_evidence.py)
  +---------------------------------------------------------------+
  |  validate_skills_evidence()      Resume -> ReviewResult       |
  |    Judgment (LLM): is each skill evidenced in experience?     |
  +---------------------------------------------------------------+


================================================================================
                     DOWNSTREAM CONSUMERS MAP
================================================================================

  OptimizedSkillsSection
         │
         ├── .optimized_skills[] ────────────► ATS Optimization (assembly)
         │
         ├── .skill_categories{} ────────────► ATS Optimization (grouping)
         │
         └── .optimization_notes ────────────► QA Agent (context)


================================================================================
                       DESIGN DECISIONS
================================================================================

  1. WHY tools=[]
     The agent writes skills. Audit (check_skills_evidence) runs code-owned
     on typed output after the agent finishes. This avoids the TOON→JSON
     serialization gap and keeps the agent focused on writing.

  2. WHY NO engines.py
     The old file had 20+ functions (infer_missing_skills, validate_skill_inference,
     prioritize_and_categorize_skills, _infer_skill_categories with hardcoded
     category patterns, etc.). The LLM handles inference and categorization
     as part of reasoning. The check_skills_evidence engine handles evidence
     validation. Hardcoded category lists can't be domain-agnostic — the LLM
     reads the job context and creates appropriate categories dynamically.

  3. WHY OptimizedSkillsSection IS IN data_models
     Already defined in src/data_models/resume.py. No models.py needed.

  4. WHY temperature 0.4
     Skill categorization needs moderate creativity to form domain-appropriate
     groups, but skill selection must be conservative (no invented skills).


================================================================================
                  HOW TO CONTROL OUTPUT QUALITY
================================================================================

  Quality is controlled in TWO places.


  ┌─────────────────────────────────────────────────────────────────────┐
  │  LEVER 1: Task instructions (WHAT to produce)                      │
  │  src/config/tasks/skill_optimizer.yaml  —  optimize_skills_section_task │
  ├─────────────────────────────────────────────────────────────────────┤
  │                                                                     │
  │  Controls:                                                          │
  │    - Skill selection and ordering rules                             │
  │    - Category grouping strategy                                     │
  │    - Truthfulness constraints (never add skills not evidenced)      │
  │    - Keyword integration from strategy                              │
  └─────────────────────────────────────────────────────────────────────┘


  ┌─────────────────────────────────────────────────────────────────────┐
  │  LEVER 2: Code-owned audit (HOW to validate)                       │
  │  src/tools/engines/truthfulness/skills_evidence.py                  │
  ├─────────────────────────────────────────────────────────────────────┤
  │                                                                     │
  │  Judgment engine (LLM):                                             │
  │    For each listed skill, checks whether the resume contains        │
  │    evidence. Returns ReviewResult with per-skill findings.          │
  │                                                                     │
  │  Runs code-owned after the agent finishes — orchestration decides   │
  │  whether to accept or request one rewrite based on findings.        │
  └─────────────────────────────────────────────────────────────────────┘
```
