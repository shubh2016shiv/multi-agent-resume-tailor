# Rules for src/agents/ — loaded ONLY when touching agent files

## Agent Architecture Rules
- Each agent is a standalone class in its own file
- All agents extend `Agent` from `crewai`
- Agents use CrewAI's built-in LLM delegation — never import `openai` or other SDKs directly
- Agent `role`, `goal`, `backstory` come from `src/config/agents.yaml`
- Task definitions for each agent live in `src/config/tasks.yaml`

## Agent-Specific Patterns
- `ats_optimization_agent.py` — Final assembly and ATS compatibility check. Quality gate.
- `experience_optimizer_agent.py` — Rewrites work experience bullets for impact
- `gap_analysis_agent.py` — Identifies skill/experience gaps vs. job requirements
- `job_analyzer_agent.py` — Parses and extracts structured data from job descriptions
- `quality_assurance_agent.py` — Validates output quality and consistency
- `resume_extractor_agent.py` — Parses resume file (PDF/DOCX) and extracts structured text
- `skills_optimizer_agent.py` — Matches and optimizes skills keywords for ATS
- `summary_writer_agent.py` — Generates professional summary tailored to job

## When Adding a New Agent
1. Create file: `src/agents/<agent_name>.py`
2. Add agent config: `src/config/agents.yaml` under a new key
3. Add task config: `src/config/tasks.yaml` under a new key
4. Register in orchestrator: `src/orchestrator/`
5. Write tests: `tests/test_<agent_name>.py`
