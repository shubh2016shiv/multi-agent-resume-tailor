# Resume Tailor — Agent Instructions
# Keep this file lean. Every line loads every session. Trim ruthlessly.
# Domain rules in .claude/rules/<path>/  |  Full protocol in docs/

## Graphify — Knowledge Graph Navigation
- graphify map exists at `graphify-out/GRAPH_REPORT.md` and `graphify-out/graph.json`
- Prefer `graphify query "<question>"` over `grep` or reading files one-by-one for codebase navigation
- Query examples:
  `graphify query "how is token refresh handled"`
  `graphify query "which agents use UserRepository"`
  `graphify query "what calls the experience optimizer"`
- Read GRAPH_REPORT.md at session start for architecture overview (optional, if HANDOFF.md absent)
- Update graph after structural changes: `graphify . --update` (tree-sitter only, zero API calls)

## Project Identity
- **Language/Runtime:** Python 3.12
- **Framework:** CrewAI multi-agent orchestration
- **GenAI/ML Stack:** CrewAI + LangSmith (observability) + structlog
- **Vector DB:** N/A (currently LLM-only pipeline)
- **Primary DB:** N/A (stateless resume processing)
- **Package manager:** uv (use `uv run`, `uv pip install`, `uv add`)
- **Test command:** `uv run pytest tests/ -x -v`
- **Run command:** `uv run python src/agent_orchestrator.py`
- **Lint/format:** `uv run ruff check src/ && uv run ruff format --check src/`
- **Type check:** `uv run pyright src/`
- **Migration command:** N/A

## Directory Map (for orientation — do NOT read source files to discover this)
```
src/agents/          ← CrewAI agent definitions (key business logic)
src/config/          ← YAML configs: agents.yaml, tasks.yaml, settings.yaml
src/core/            ← Shared: config loader, logger, resiliency, token_counter
src/data_models/     ← Pydantic v2 models (job, resume, strategy, evaluation)
src/formatters/      ← Output formatting per agent (PDF, Markdown)
src/observability/   ← LangSmith + structlog integration
src/tools/           ← CrewAI tools (file reading, parsing, PII handling)
src/orchestrator/    ← Crew orchestration logic
tests/               ← pytest suite (mirrors src/ structure)
```

## Critical Conventions
- Agent classes live in `src/agents/<agent_name>.py`, one per file
- All LLM calls go through CrewAI's agent/task framework — never call the SDK directly
- Config uses Pydantic `BaseSettings` with `.env` loading (YAML configs in `src/config/`)
- Data models are Pydantic v2 in `src/data_models/` — `Field(description=...)` on every field
- Observability via `structlog` + `langsmith` — use `logger = structlog.get_logger()` everywhere
- Pre-commit hooks enforce ruff formatting + pyright on commit

## Session Start Protocol (30 seconds, saves thousands of tokens)
1. **HANDOFF.md exists?** → Read it fully. That IS your context. Skip to step 3.
2. **graphify installed?** → `graphify query "<what you need>"`. Otherwise: read only the top-level structure. Do NOT open source files until you know exactly which one.
3. **State in one sentence:** "I will [action] in [specific file(s)] to achieve [goal]."
   If you cannot complete this sentence, the task needs scoping. Ask.

## Context Discipline — Hard Rules
- **Read by line range, not full file.** `sed -n '45,90p' src/...` — NOT `cat src/...`
- **Max 4 source files in context at once.** More than that → the task needs breaking down.
- **Never read these files** unless the task is explicitly about them:
  `pyproject.toml`, `uv.lock`, `.env`, `requirements.txt`, prompt template files in bulk
- **Compact manually at ~65% context fill.** Do NOT wait for auto-compact at 95%.
- **`/clear` between unrelated tasks.** Never carry auth-bug context into embedding refactor.
- **Subagents for research.** Reading docs, running evals, checking logs, grep exploration → delegate.
- **Self-check — STOP and re-scope if:**
  - More than 4 source files in context
  - Reading a file over 200 lines in its entirety
  - Running `grep -r` on the whole repo as first action
  - Full test suite for one failing unit
  - Re-reading a file already in this session's context

## Coding Decision Protocol (before writing ANY code)
State these four things. If you cannot answer all without reading >2 files, scope down.
1. **Root cause** — one sentence. From HANDOFF or graph, not speculation.
2. **Scope** — exact files to change (bug fix: 1–2, new feature: ≤5).
3. **Out of scope** — files/areas explicitly NOT to touch this session.
4. **Verification** — the exact test command that confirms the fix.

## Change Size Contracts
| Change type | Max files | Approach |
|---|---|---|
| Bug fix | 1–2 | Surgical edit + targeted test |
| New agent | 1–2 | Agent class + registration |
| New data model | 1–2 | Model file + tests |
| New formatter | 1–2 | Formatter + test |
| New tool | 1–2 | Tool definition + registration |
| Refactor | **Stop.** Subagent for impact analysis first. |

## GenAI-Specific Hard Rules (this project IS an LLM pipeline)
- **Prompts are files, not strings.** Any multi-line prompt → `src/config/prompts/` as `.yaml` or `.j2`. Never inline in business logic.
- **Token budget check before LLM calls.** Any code path calling an LLM must have a `count_tokens()` guard or `max_tokens` cap. Never call LLM unbounded.
- **Model name is config, not hardcode.** If you find a model name string in code, extract to `src/config/settings.yaml`.
- **Eval runs are isolated.** Never run evaluation harnesses in the main session. Delegate to subagent.

## Coding Discipline (full protocol: `docs/agentic-coding-control-protocol.md`)
- **Think before coding:** state assumptions, surface tradeoffs, STOP if unclear.
- **Skeleton-first:** output signatures + data flow, WAIT for "implement" before writing bodies.
- **Complexity budget:** 3 pts/request (class=1, model=1, new file=1). Stop and ask if exceeded.
- **YAGNI + KISS:** flat before layered, function before class. No future-proofing.
- **Max 20 lines/function, 4 params, 3 nesting levels, 3 call depth.**
- **Docstrings mandatory:** what it does, what it expects, what it returns.
- **Surgical changes only:** touch only what the request asks. Match existing style.
- **Goal-driven:** transform tasks into verifiable goals with verify checks. Loop until passes.
- **BANNED:** `sys.path` hacks, `eval`, bare `except:`, mutable defaults, hardcoded absolutes, decorative Unicode.
- **Ask before:** new class, new file, new abstraction, new dependency, exceeding budget.
- **TODO format:** `# TODO: [case] — Proposed: [handling] — Deferred because: [reason]`

## Model Tiering
| Task | Model | Why |
|---|---|---|
| Architecture, complex multi-agent debugging | Claude Opus / GPT-4o | Full reasoning needed |
| New agent, service logic, LLM pipeline code | Claude Sonnet / GPT-4.1 | Daily workhorse |
| Test writing, docstrings, type hints, formatting | Claude Haiku / GPT-4o-mini | Sonnet is overkill |
| Eval runs, doc research, log analysis | Haiku in subagent | Isolated context, cheap |
| Linting, renaming, simple edits | Inline tool or Haiku | No deep model needed |

Disable extended thinking for routine tasks: `MAX_THINKING_TOKENS=3000` or `/effort low`.

## Anti-Patterns (agent must refuse or flag)
| Anti-pattern | Why wrong | Correct |
|---|---|---|
| `cat src/.../file.py` to "understand it" | Loads entire file, 80% irrelevant | Read by line range or ask targeted question |
| `grep -r` across repo as first action | Loads every match into context | Graph query → then targeted read |
| `pytest tests/` after small bug fix | Full suite burns tokens + time | `pytest tests/path/test_file.py::test_name -x` |
| Rewriting module because one function is wrong | Destroys unrelated working code | Fix only the broken function |
| Inlining prompt as Python string | Untestable, unversioned, token-heavy | Move to `src/config/` or `src/prompts/` |
| Loading `.env` to check API keys | Security risk + token waste | Read env var names from code, not values |
| Carrying context across unrelated tasks | Cross-contamination, bad decisions | `/clear` between unrelated tasks |
| Re-reading file already in context | Pure waste | Reference what is already in session |
| Waiting for 95% context fill to compact | Context already bloated | Compact at ~65% |

## Session End Protocol
Write HANDOFF.md BEFORE `/compact` or session close if ANY file was modified.
```
# HANDOFF.md
_Updated: [ISO timestamp] | Goal: [one-line session task]_

## Completed
- [change 1: what changed, which file:function]
- [change 2]

## Current state
- ✅ Working: [what is confirmed]
- ❌ Broken: [what exactly is not working]
- ⏸ Blocked: [what is needed to unblock, if anything]

## Files modified
| File | Change | Why |
|------|--------|-----|

## Exact next action
<!-- Next agent should act on this without re-reading code. -->
- [ ] [specific file:line] — [exact change to make]
- [ ] Verify: `[exact test command]`

## Do NOT touch
- [files/areas explicitly off-limits]
```

## Scoped Rules (load ONLY when touching files in that directory)
- `src/agents/` → `.claude/rules/agents/RULES.md`
- `src/config/` → `.claude/rules/config/RULES.md`
- `src/data_models/` → `.claude/rules/data_models/RULES.md`
- `src/tools/` → `.claude/rules/tools/RULES.md`
