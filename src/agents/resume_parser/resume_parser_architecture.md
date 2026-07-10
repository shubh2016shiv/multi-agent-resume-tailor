# Resume Parser Agent — Architecture

```
================================================================================
                        THREE-ZONE ARCHITECTURE
================================================================================

  ZONE 3: Agent Factory          (src/agents/resume_parser/agent.py)
  +---------------------------------------------------------------+
  |  create_resume_extractor_agent()                              |
  |                                                               |
  |  Loads YAML config, validates required fields, builds         |
  |  CrewAI Agent with 4 tools. No business logic.                |
  |  Fails fast on bad config (RuntimeError).                     |
  +---------------------------------------------------------------+
         |                              |
         | assigns tools                | imports config
         v                              v
  +---------------------------+   +---------------------------+
  | src/tools/                |   | src/core/config.py        |
  | agent_facing_tools.py     |   | src/config/agents.yaml    |
  | (4 @tool functions)       |   +---------------------------+
  +---------------------------+
         |
         | calls
         v
  ZONE 1: Engines                (src/tools/document_ingestion/)
  +---------------------------------------------------------------+
  |  convert_document_to_markdown()    PDF/DOCX -> str            |
  |  audit_extraction_quality()        str -> ReviewResult        |
  |  redact_pii()                      str -> (str, mapping)      |
  |  extract_resume()                  str -> Resume              |
  |                                                               |
  |  Pure functions. Pydantic in/out. No framework. No LLM.       |
  +---------------------------------------------------------------+


================================================================================
                           DATA FLOW (left to right)
================================================================================

  file_path
     |
     v
  +------------------+     +-------------------+     +------------------+     +-------------------+
  | convert_document |     | audit_extraction  |     |                  |     |                   |
  | _to_markdown     |---->| _quality          |---->| redact_pii       |---->| extract_resume    |
  |                  |     |                   |     |                  |     |                   |
  | PDF -> str       |     | str -> ReviewRes  |     | str -> (str,map) |     | str -> Resume     |
  +------------------+     +-------------------+     +------------------+     +-------------------+
         |                        |                          |                        |
         | ZONE 1                 | ZONE 1                   | ZONE 1                 | ZONE 1
         v                        v                          v                        v
  +---------------------------------------------------------------------------+
  |                        ZONE 2: @tool Adapter Layer                        |
  |  agent_facing_tools.py                                                    |
  |                                                                           |
  |  @tool("Convert Resume Document to Markdown")                             |
  |    -> calls convert_document_to_markdown(file_path)                       |
  |    -> returns str (Markdown)                                              |
  |                                                                           |
  |  @tool("Check Resume Markdown Quality")                                   |
  |    -> calls audit_extraction_quality(markdown)                            |
  |    -> renders ReviewResult -> agent-readable report string                |
  |                                                                           |
  |  @tool("Redact PII from Resume Markdown")                                 |
  |    -> calls redact_pii(markdown)                                          |
  |    -> returns str (redacted Markdown, discards mapping for agent)         |
  |                                                                           |
  |  @tool("Extract Structured Resume from Markdown")                         |
  |    -> calls extract_resume(redacted_md)                                   |
  |    -> returns Resume.model_dump_json() as string                          |
  +---------------------------------------------------------------------------+
         ^
         | tools assigned to agent (ZONE 3)
         |
  +---------------------------------------------------------------------------+
  |                       ZONE 3: Agent Factory                               |
  |  agent.py                                                                 |
  |                                                                           |
  |  create_resume_extractor_agent()                                          |
  |    -> loads config from agents.yaml["resume_content_extractor"]           |
  |    -> validates: role, goal, backstory, llm must exist                    |
  |    -> creates LLM instance                                                |
  |    -> builds Agent with 4 tools                                           |
  |    -> returns Agent                                                       |
  +---------------------------------------------------------------------------+


================================================================================
                        IMPORT DEPENDENCY GRAPH
================================================================================

  agent.py
     |
     |----> from crewai import Agent, LLM
     |
     |----> from src.core.settings import get_agents_config, get_config
     |
     |----> from src.core.logger import get_logger
     |
     |----> from src.tools.agent_facing_tools import (
     |          convert_resume_document_to_markdown,    # @tool
     |          check_resume_markdown_quality,          # @tool
     |          redact_pii_from_resume_markdown,        # @tool
     |          extract_structured_resume_from_markdown # @tool
     |      )

  agent_facing_tools.py
     |
     |----> from crewai.tools import tool
     |
     |----> from src.tools.document_ingestion import (
     |          convert_document_to_markdown,    # engine
     |          audit_extraction_quality,        # engine
     |          redact_pii,                      # engine
     |          extract_resume                   # engine
     |      )
     |
     |----> from src.tools.review_contract.review_models import ReviewResult

  KEY RULE: Dependencies flow DOWN the zones.
            Zone 3 imports Zone 2. Zone 2 imports Zone 1.
            Zone 1 imports NOTHING from Zone 2 or Zone 3.
            Zone 2 imports NOTHING from Zone 3.


================================================================================
                      FILE RESPONSIBILITIES
================================================================================

  __init__.py
    Single export: create_resume_extractor_agent.
    No pipeline exports. Callers import engines or tools directly.

  agent.py
    Agent factory. Config validation + Agent construction.
    One function: create_resume_extractor_agent() -> Agent.

  agent_facing_tools.py (src/tools/)
    Serialization boundary. @tool decorators. String in -> string out.
    Renders ReviewResult for agent consumption.

  document_ingestion/ (src/tools/)
    Pure engines. No @tool. No CrewAI. Testable with pytest.


================================================================================
                        DESIGN DECISIONS
================================================================================

  1. WHY pipeline.py WAS REMOVED
     - Every function was a pass-through: identical signature, zero added logic.
     - agent.py already imports directly from agent_facing_tools.py.
     - No caller anywhere in the codebase imported pipeline.py.
     - Three-layer wrapping (engine -> pipeline -> tool) adds indirection
       with zero benefit. Two layers (engine -> tool) is the production standard.

  2. WHY THE AGENT FACTORY IS SINGLE-RESPONSIBILITY
     - It validates config (fail fast, clear error message).
     - It creates the LLM instance.
     - It builds the Agent with tools.
     - No business logic, no LLM calls, no data manipulation.
     - Under 30 lines total.

  3. WHY TOOLS ARE SEPARATE FROM ENGINES
     - Engines are pure, typed, framework-free. Testable in isolation.
     - Tools are the serialization boundary: JSON parse -> engine call -> render.
     - This separation is the standard pattern in every production CrewAI repo.

  4. WHY ALL 4 TOOLS ARE ASSIGNED
     - convert: deterministic, the agent calls it to get Markdown.
     - check quality: the agent reasons over the report before proceeding.
     - redact PII: deterministic, masks before LLM sees text.
     - extract: schema-constrained LLM call that produces Resume.
     - The pipeline is: convert -> quality check -> redact -> extract.
     - The agent orchestrates this sequence, calling each tool in order.
```
