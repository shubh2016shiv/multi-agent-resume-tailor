# Graphify — The Practical Guide to Knowledge Graph Navigation for Agentic Coding
## What It Is, How It Works, and How to Integrate It Into Your Development Workflow

> **Scope:** Claude Code · OpenAI Codex CLI · GitHub Copilot · Cursor · Aider · Gemini CLI  
> **Audience:** Backend Engineers · GenAI Engineers · Solo Developers · Engineering Teams  
> **Edition:** 2025–2026  
> **Tool Version:** graphify v0.8.30

---

## Quick Start — Setup Checklist

Three steps. Once per project. After that, everything is automatic.

### Step 1: Build the initial knowledge graph

Two options. For token savings alone, Option A is sufficient.

**Option A: Code-only (no API key needed)**

```bash
uv run graphify update . --no-cluster
```

Runs only tree-sitter AST extraction. Zero API calls. Produces `graph.json` — the
queryable graph agents use for navigation. Skips `GRAPH_REPORT.md` and `graph.html`.

- `update` is the **no-LLM** command — it re-extracts code files using only tree-sitter.
  Works for both initial builds and incremental updates.
- `--no-cluster` skips Leiden community detection (just writes raw extraction edges).
  Drop this flag if you want community labels (still no API key needed).
- **Do NOT use `extract`** for code-only builds — `extract` is the headless full pipeline
  (AST + semantic LLM) and always requires an API key or `--backend` flag.

**This alone is enough for token savings.** The agent queries `graph.json` instead
of reading files blindly. The report and visualization are optional.

**Option B: Full build (needs API key)**

```bash
graphify .
# or equivalently:
graphify extract . --backend claude
```

Full pipeline: AST + semantic LLM (community naming, report, visualization). Produces
all three output files. Needs `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `GEMINI_API_KEY`
in your environment.

Start with Option A. You can generate the report later from inside your IDE (Claude Code,
Codex, etc.) where the model API is provided automatically via `/graphify .`.

Commit `graphify-out/` after building. First run: 30–90s; later runs are incremental
via SHA256 cache per file.

### Step 2: Wire graphify into your coding assistant

**Automatic approach (most projects):**

```bash
graphify claude install   # or: codex, cursor, aider, gemini, etc.
```

This modifies two files that you then commit:
- `AGENTS.md` (or `CLAUDE.md`) — adds instructions telling the agent to prefer
  `graphify query "<question>"` over `grep` and file reading.
- `.claude/settings.json` — adds a `PreToolUse` hook that blocks `grep -r`, `rg`,
  and `find . -name` when `graphify-out/graph.json` exists.

**Manual integration (when you need full control over wording and placement):**

If your project already has heavily customized instruction and hook files, you may prefer
to integrate the graphify additions by hand. This approach was used for the Resume Tailor
project because `AGENTS.md` and `.claude/settings.json` already contained extensive custom
hooks (HANDOFF enforcement, test-suite blocking, coding discipline rules). The manual
integration involved two changes:

1. **`AGENTS.md`** — Added a "Graphify — Knowledge Graph Navigation" section at the top
   telling the agent: prefer `graphify query` over `grep` or file reading; example queries
   provided; read `GRAPH_REPORT.md` for architecture overview at session start.

2. **`.claude/settings.json`** — Extended the existing `PreToolUse` → `Bash` hook to also
   intercept `grep -r`, `grep -rn`, `rg`, and `find . -name`. If `graphify-out/graph.json`
   exists, the hook blocks the command and redirects the agent to `graphify query`. If the
   graph does not exist, the command proceeds normally (graceful degradation).

Both approaches produce the same result. Choose automatic for simplicity, manual for control.

### Step 3: Auto-rebuild the graph on every git commit

```bash
uv run graphify hook install
```

If graphify is installed globally (via `uv tool install graphifyy`), omit `uv run`:
```bash
graphify hook install
```

Installs a post-commit hook in `.git/hooks/`. Every `git commit` triggers an incremental
AST extraction (SHA256-cached, under a second for typical commits). Also installs a merge
driver for `graph.json` to prevent conflict markers when two branches modify the graph.

**Important: the hook only rebuilds the AST graph (`graph.json`) — it does NOT regenerate
the LLM-based report (`GRAPH_REPORT.md`).** This is intentional: AST extraction costs
zero API calls and completes in under a second; regenerating the report with an LLM on
every commit would be slow and expensive. See Step 4 below for report updates.

This step is **per-machine** because `.git/hooks/` is not tracked by version control.
Every developer runs it once after cloning:

```bash
git clone <repo>
uv sync --group graphify
uv run graphify hook install
```

### Step 4: Generate the human-readable report (Phase 2, optional)

The code-only build from Step 1 (Option A) produces `graph.json` — sufficient for
token savings. The human-readable report (`GRAPH_REPORT.md`), HTML visualization, and
community labels require an LLM call. Run this when you want the architecture summary:

**From the CLI (needs API key):**
```bash
# Source your .env first, then run the full build
set -a && source .env && set +a
graphify .

# Or pass the key inline (choose your provider)
ANTHROPIC_API_KEY=sk-ant-... graphify .
OPENAI_API_KEY=sk-... graphify .
GEMINI_API_KEY=... graphify .

# Or use the extract command with explicit backend
graphify extract . --backend claude     # uses ANTHROPIC_API_KEY
graphify extract . --backend openai     # uses OPENAI_API_KEY
graphify extract . --backend gemini     # uses GEMINI_API_KEY
```

Graphify auto-detects which provider to use based on available environment variables
(priority: `GEMINI_API_KEY` → `ANTHROPIC_API_KEY` → `OPENAI_API_KEY` →
`DEEPSEEK_API_KEY`). It uses the provider'''s default model unless overridden:
```bash
graphify extract . --backend openai --model gpt-4o
```

**From inside your IDE (no API key needed):**
Inside Claude Code, Codex, or Cursor, simply type `/graphify .` — the IDE session
provides the model API automatically. This is the simplest path to a full build.

**The git hooks (Step 3) do NOT auto-update the report.** The hooks rebuild only the AST
graph (`graph.json`). To refresh the report after code changes, run one of the above
commands manually. A good cadence: regenerate the report after major architectural changes
(splitting a module, adding/removing agents), not on every commit.

### Summary

| Step | Command | Frequency | Committed? |
|---|---|---|---|
| 1 | `uv run graphify update . --no-cluster` | Once; `uv run graphify update .` after restructures | Output (`graphify-out/`) yes |
| 2 | `graphify claude install` (or manual) | Once per project | Modified config files yes |
| 3 | `uv run graphify hook install` | Once per machine after clone | No — hooks are per-machine |
| 4 | `graphify .` (full build with API key) | Occasionally, after major architecture changes | Output (`graphify-out/`) is updated and recommitted |

After setup: every `git commit` silently rebuilds the AST graph. The agent queries it
instead of reading files blindly. Regenerate the report when you want updated architecture
docs. Zero ongoing effort.

---

## Table of Contents

1. [What Problem Graphify Solves](#1-what-problem-graphify-solves)
2. [How Graphify Works — The Extraction Pipeline](#2-how-graphify-works--the-extraction-pipeline)
3. [What Graphify Produces — The Output Artifacts](#3-what-graphify-produces--the-output-artifacts)
4. [Installation and Dependencies](#4-installation-and-dependencies)
5. [Configuring What Gets Indexed — The Graphifyignore File](#5-configuring-what-gets-indexed--the-graphifyignore-file)
6. [Platform Integration — How Coding Assistants Use the Graph](#6-platform-integration--how-coding-assistants-use-the-graph)
7. [The Token-Saving Mechanism — Concrete Before and After](#7-the-token-saving-mechanism--concrete-before-and-after)
8. [Project Setup — What Is Committed vs. What Is Manual](#8-project-setup--what-is-committed-vs-what-is-manual)
9. [Day-to-Day Usage and Queries](#9-day-to-day-usage-and-queries)
10. [Keeping the Graph Current — Incremental Updates](#10-keeping-the-graph-current--incremental-updates)
11. [Team Setup and Version Control](#11-team-setup-and-version-control)
12. [Advanced Usage — MCP Server, CI, Headless Extraction](#12-advanced-usage--mcp-server-ci-headless-extraction)
13. [Troubleshooting and Edge Cases](#13-troubleshooting-and-edge-cases)
14. [Alternatives Considered — Why Graphify Over Other Tools](#14-alternatives-considered--why-graphify-over-other-tools)
15. [Future Considerations and Extensions](#15-future-considerations-and-extensions)

---

## 1. What Problem Graphify Solves

Every AI coding agent faces the same fundamental problem on its first interaction with a codebase: it has no idea where anything lives. The agent compensates for this ignorance by reading broadly. It opens files speculatively. It runs broad `grep` searches whose results compound into the conversation history. It reads configuration files, test files, and documentation in an attempt to build a mental model of the project structure. Every file it opens, every search result it receives — all of that becomes permanent context that is re-sent to the model on every subsequent turn in the session.

The token efficiency guide documents that this orientation phase consumes 60–80% of all tokens in a typical agent session. An agent reading 25 files to answer a question about three functions is not an edge case; it is the default behaviour of an agent that has no structural understanding of the code it is working with. That interaction costs 12,000 tokens. The answer required 800.

Graphify solves this by giving the agent a pre-built, queryable map of the entire codebase before the agent opens a single source file. Instead of reading files to discover structure, the agent queries a knowledge graph that already knows the structure. The difference is not incremental — documented benchmarks show 6.8× to 49× token savings per navigation task compared to blind file reading, depending on codebase size and query complexity.

The tool achieves this through a two-pass extraction pipeline that processes code locally with zero API calls and optionally enriches the graph with semantic understanding of documentation and other non-code artifacts. The resulting graph captures not just what exists in the codebase but how things relate to each other — which functions call which, which classes depend on which, and which modules form natural clusters of related concern.

---

## 2. How Graphify Works — The Extraction Pipeline

Understanding the extraction pipeline is important because it explains why the tool can index a large codebase in under two minutes while consuming zero API tokens for the code portion. The pipeline has distinct stages, each with a clear input, output, and cost profile.

### Stage 1: File Discovery and Filtering

Graphify walks the project directory tree and collects every file that matches its supported formats. It respects a `.graphifyignore` file that uses the same syntax as `.gitignore`, including `!` negation patterns. Files matching `.gitignore` patterns are also excluded automatically. This stage produces a filtered list of file paths and costs nothing beyond the filesystem walk.

For a Python project like the Resume Tailor system, this typically means every `.py` file under `src/` and `tests/` is collected, while virtual environments, build artifacts, cache directories, lock files, and documentation are excluded by configuration.

### Stage 2: AST Extraction — The Core Engine

This is the most important stage to understand because it is where the majority of the graph's value is generated at zero cost. Graphify uses tree-sitter, a language-agnostic incremental parsing library, to parse every collected file and extract two categories of information: symbol definitions and symbol relationships.

Symbol definitions include functions, classes, methods, module-level variables, and import statements. For each symbol, graphify records its name, its source file, its line number, and its type. Symbol relationships include imports between modules, function calls within and across files, class inheritance hierarchies, and variable usage. These relationships form the directed edges of the graph.

The critical property of this stage is that tree-sitter operates entirely locally. There is no network call. There is no LLM invocation. There is no API key requirement. The parser reads the file bytes, builds an abstract syntax tree in memory using a grammar compiled to C, walks the tree to collect symbols and relationships, and writes the results to a structured extraction dictionary. For a Python project with 10,000 lines of code across 50 files, this stage completes in seconds.

Each file's extraction is cached using a SHA256 hash of its contents. On subsequent runs, only files whose contents have changed are re-extracted. This makes incremental updates fast enough to run on every git commit without noticeable latency.

### Stage 3: Graph Construction and Community Detection

The extraction dictionaries from all files are merged into a single NetworkX graph — a well-established Python library for graph analysis used in production systems ranging from social network analysis to supply chain optimization. Nodes in the graph represent symbols (functions, classes, modules). Edges represent relationships (calls, imports, inheritance).

Once the graph is built, graphify applies the Leiden algorithm for community detection. Leiden is a state-of-the-art algorithm that partitions a graph into clusters of densely connected nodes. In the context of a codebase, a community represents a group of symbols that are tightly coupled — functions that call each other frequently, classes that share inheritance hierarchies, modules that form a cohesive subsystem. These communities become the basis for understanding the architecture: the agent can see at a glance which parts of the codebase are related, even if they live in different files or directories.

The output of this stage is a graph object with every node annotated with its community membership, a confidence score for each edge (EXTRACTED, INFERRED, or AMBIGUOUS), and centrality metrics that identify "god nodes" — the most-connected symbols that everything else flows through.

### Stage 4: Analysis and Report Generation

The final stage consumes the annotated graph and produces human-readable and machine-readable output. The analysis identifies god nodes, surprising connections (relationships between symbols in different communities or directories), and generates suggested questions that the graph is uniquely positioned to answer. The report also extracts inline comments marked with specific prefixes — `# NOTE:`, `# WHY:`, `# HACK:` — and surfaces them as separate nodes linked to the code they explain, giving the agent visibility into design rationale and technical debt without reading the code.

### Stage 5: Optional Semantic Extraction

For projects that include significant documentation, PDFs, or other non-code artifacts, graphify offers an optional second pass that uses an LLM to extract semantic meaning from these files. This pass is only triggered when the agent explicitly runs `/graphify` within an IDE session (using the IDE's own model API, no separate key needed) or when a headless `graphify extract` command is invoked with an explicit backend flag.

For projects that choose to index only source code — as the Resume Tailor project does — this stage is skipped entirely. The graph is built purely from AST extraction, consuming zero API tokens and requiring zero API keys.

---

## 3. What Graphify Produces — The Output Artifacts

Running `graphify .` produces three files in a `graphify-out/` directory at the project root. Each serves a different purpose in the agent workflow.

### graph.json — The Machine-Queryable Graph

This is the authoritative data structure. It is a NetworkX graph serialized to JSON, containing every extracted node and edge with their metadata. When an agent runs `graphify query "how is auth handled"`, it reads this file and traverses the graph to find relevant nodes and their neighbours. The file is the single source of truth for all graph-based navigation.

For a typical Python project with 50 files and 500 symbols, `graph.json` is on the order of 200–500 KB — small enough to load in milliseconds. The file is designed to be committed to version control so that every developer on the team starts with a pre-built map.

### GRAPH_REPORT.md — The Human and Agent Readable Summary

This is a Markdown file that serves as the agent's first point of contact with the codebase. It contains a structured overview of the project architecture: the most-connected symbols (god nodes), surprising cross-module connections, community clusters with human-readable labels, and a set of suggested questions that demonstrate what the graph can answer.

An agent following the session start protocol reads this file first — before opening any source file — to orient itself. The report is concise enough that reading it costs a few hundred tokens, compared to the thousands of tokens the agent would spend discovering the same information through file reading.

For the Resume Tailor project, a GRAPH_REPORT.md might surface that `ats_optimization_agent.py` is the most-connected file because every other agent feeds its output into the ATS optimization step, or that `config.py` in `src/core/` is a god node because every module imports it for settings access.

### graph.html — The Interactive Visualization

This is an HTML file that can be opened in any browser to explore the graph visually. Nodes can be clicked to see their details, filtered by type or community, and searched by name. The visualization is primarily a human tool — useful for understanding the architecture at a glance, presenting the codebase structure to new team members, or debugging unexpected connections.

The HTML file can be large for projects with thousands of nodes. For very large graphs (over 5,000 nodes), the `--no-viz` flag skips HTML generation and reduces output to just the report and JSON.

---

## 4. Installation and Dependencies

Graphify is distributed as a Python package on PyPI under the name `graphifyy` (note the double-y — other `graphify`-prefixed packages on PyPI are not affiliated). The CLI command remains `graphify`.

### Core Installation

The recommended installation method is `uv tool install`, which isolates the package in its own environment and places the CLI binary on the user's PATH automatically:

```bash
uv tool install graphifyy
```

Alternative methods include `pipx install graphifyy` (same isolation semantics) or `pip install graphifyy` (requires manual PATH configuration for the scripts directory).

For projects using uv as their package manager, graphify can also be declared as a dependency group in `pyproject.toml` under `[dependency-groups]`. This keeps it separate from the project's runtime dependencies and makes it installable via `uv sync --group graphify`. This is the approach used in the Resume Tailor project.

### Dependency Tree

The core package has a focused dependency tree designed for code-only extraction:

- **networkx** — The graph data structure and traversal algorithms. A mature, well-maintained library used extensively in scientific computing and data engineering.
- **tree-sitter** (core) plus 27 language-specific grammar packages — The AST parsing engine. Each grammar is a compiled C library that tree-sitter loads dynamically. Grammars exist for Python, JavaScript, TypeScript, Go, Rust, Java, C, C++, Ruby, C#, Kotlin, Scala, PHP, Swift, Lua, Zig, PowerShell, Elixir, Objective-C, Julia, Verilog, Fortran, Bash, JSON, and several others. SQL and BYOND DreamMaker grammars are available as optional extras.
- **datasketch** — MinHash-based deduplication for identifying near-duplicate symbols across the graph.
- **rapidfuzz** — Fast fuzzy string matching for disambiguating symbol names that differ by minor variations.

For code-only extraction, these are the only dependencies. No LLM SDK, no API client library, no database driver. The entire extraction pipeline runs on the developer's machine with no external service dependencies.

### Optional Extras

Graphify offers installable extras for additional capabilities, each gated behind its own dependency group:

| Extra | Additional dependencies | Purpose |
|---|---|---|
| `pdf` | pypdf, markdownify | Extract text and structure from PDF documents |
| `office` | python-docx, openpyxl | Extract text from .docx and .xlsx files |
| `video` | faster-whisper, yt-dlp | Transcribe video and audio files (runs entirely locally) |
| `mcp` | mcp | Expose the graph as an MCP server for tool-based agent access |
| `leiden` | graspologic | Community detection via the Leiden algorithm (Python < 3.13 only) |
| `anthropic` / `openai` / `gemini` | respective SDKs | Headless extraction via specific LLM backends for CI pipelines |
| `neo4j` | neo4j | Push the graph to a Neo4j database for persistent querying |
| `svg` | matplotlib | Export the graph as an SVG image |

For the Resume Tailor project's code-only configuration, none of these extras are needed. The core package suffices.

---

## 5. Configuring What Gets Indexed — The Graphifyignore File

The `.graphifyignore` file in the project root controls which files graphify includes in its extraction. The syntax is similar to `.gitignore` — but **`!` negation patterns are not supported** by the `update` command (the no-LLM path). Use exclude-only patterns instead of blanket-exclude-then-whitelist.

### The Decision for Code-Only Indexing

The Resume Tailor project made a deliberate decision to index only Python source files under `src/` and `tests/`. This decision was driven by several factors:

First, the project's documentation files — AGENTS.md, the token efficiency guide, the control protocol, and various implementation summaries — are valuable human references but do not contain code structure information that would help an agent navigate the codebase. Including them in the graph would consume extraction time and graph space without adding actionable navigation value.

Second, the project contains sample documents (PDFs, DOCX files) that are binary blobs. These would require the PDF or office extras and an LLM call to extract meaningfully. The token cost of extracting them would exceed the token savings from having them in the graph, since the agent rarely needs to navigate them.

Third, the project has numerous root-level test scripts (`test_*.py`, `validate_*.py`, `verify_*.py`) that are exploratory or one-off in nature. They are not imported by the main source tree and do not represent the project's architecture. Including them would add noise to the graph without adding signal.

The resulting `.graphifyignore` uses an exclude-only approach: list every non-source directory and file type to skip. While this is more verbose than a whitelist, it works reliably with the `update` command.

### Pattern Reference

```
# Exclude non-source directories
__pycache__/
.git/
.venv/
graphify-out/
docs/
raw/
scripts/
.claude/

# Exclude non-Python files at project root
*.md
*.txt
*.toml
*.lock
*.yaml
*.yml
*.json
*.cfg
.env*
.graphifyignore
.gitignore
Dockerfile
Makefile
LICENSE
```

> **Why not `!` negation?** The `!` prefix (re-include) works in `.gitignore` but is not
> supported by graphify's `update` command. A blanket `*` exclude followed by `!src/**`
> results in zero files being found. If you need whitelist-style patterns, use the
> `extract` command instead (requires API key). For no-LLM builds, use exclude-only.

This pattern can be adapted for any project. For a monorepo with multiple services, you might add each non-source directory individually. For a project where documentation is tightly coupled to code (e.g., a library with docstring-driven API docs), you might include the `docs/` directory and install the appropriate extras for Markdown extraction.

---

## 6. Platform Integration — How Coding Assistants Use the Graph

Graphify integrates with coding assistants at two levels: instruction files and hook-based enforcement. The integration mechanism varies by platform, but the goal is the same everywhere: make the agent prefer graph queries over file reads for codebase navigation.

### Claude Code Integration

Claude Code offers the deepest integration because of its hooks system. Graphify's `graphify claude install` command (or its manual equivalent, as used in the Resume Tailor project) makes three changes:

**Instruction file injection.** A block is added to the project's instruction file (`CLAUDE.md` or `AGENTS.md`) that tells the agent to prefer `graphify query` over `grep` and file reading for codebase navigation. The block includes example queries that demonstrate the pattern: "how is X handled", "which modules depend on Y", "what calls Z".

**PreToolUse hook for search operations.** A hook is installed that fires before every Bash tool call. The hook inspects the command being executed. If it is a broad search command — `grep -r`, `grep -rn`, `rg`, or `find . -name` — and a `graphify-out/graph.json` file exists, the hook blocks the command and tells the agent to use `graphify query` instead. The hook allows the search command to proceed if the graph does not exist or if the graph query would be inappropriate for the specific search.

This is the critical enforcement layer. Without it, the agent's instruction to "prefer graphify" is a suggestion that the stochastic model may or may not follow. With the hook, the preference becomes a hard rule: the agent cannot run `grep -r` without first exhausting graph-based queries.

**Session start protocol.** The agent's session start protocol — defined in `AGENTS.md` — directs it to read `GRAPH_REPORT.md` (if it exists and no `HANDOFF.md` is present) before opening any source file. This gives the agent an architectural overview of the entire codebase in a few hundred tokens rather than the thousands it would spend discovering the same information through file reading.

### Other Platform Integrations

On platforms without a hooks system (Codex, OpenCode, Cursor, Aider, Copilot CLI), graphify writes persistent instruction files that achieve the same guidance through the platform's native mechanism. Cursor receives a `.cursor/rules/graphify.mdc` file with `alwaysApply: true` — Cursor includes it in every conversation automatically. Codex and Aider receive additions to their instruction files (`AGENTS.md`). VS Code Copilot Chat receives a skill file.

The guidance is the same across all platforms: query the graph before reading files. The enforcement mechanism differs, but the behavioural outcome is designed to be identical.

### The Skill File System

Graphify uses a skill file architecture for IDE-based operation. When the agent types `/graphify .` inside the IDE, the skill file teaches the agent how to invoke graphify's extraction and query commands. The skill file and its reference sidecar are installed into the platform's skill directory (`.claude/skills/` for Claude Code, `.agents/skills/` for Codex, etc.) and can be committed to version control when using the `--project` flag.

The skill file itself is a Markdown document that the agent reads on demand — it does not load automatically into every session. This is an intentional design choice: the skill file is reference material that the agent consults only when it needs to build or query the graph, not overhead that inflates every session's token consumption.

---

## 7. The Token-Saving Mechanism — Concrete Before and After

The most effective way to understand the value of graph-based navigation is to trace a specific task through both the traditional approach and the graph-based approach. The following comparison uses a realistic scenario from the Resume Tailor codebase.

### Scenario: Find All Agents That Depend on the Experience Optimizer's Output

**Task:** An agent needs to understand the data flow between agents to debug an issue where experience optimization results are not propagating to the quality assurance step. The agent needs to identify every file that imports from or references `experience_optimizer_agent.py`.

### Without Graphify — The Traditional Approach

The agent has no structural knowledge of the project. It falls back to discovery through file reading:

```
Turn 1: Agent runs: grep -r "experience_optimizer" src/
        → 47 matches across 12 files, loaded as a tool result into context
        → Approximate cost: 1,200 tokens

Turn 2: Agent reads src/agents/experience_optimizer_agent.py
        → Entire file (approximately 300 lines) loaded into context
        → Approximate cost: 800 tokens

Turn 3: Agent reads src/agents/ats_optimization_agent.py
        → Looking for how it consumes experience optimizer output
        → Another 500 lines loaded into context
        → Approximate cost: 1,200 tokens

Turn 4: Agent reads src/orchestrator/ (discovers orchestration file)
        → Reads the orchestrator to understand the agent pipeline
        → Approximate cost: 900 tokens

Turn 5: Agent reads src/data_models/strategy.py
        → Checking the data model that passes between agents
        → Approximate cost: 600 tokens

Cumulative cost for navigation alone: approximately 4,700 tokens.
Of this, the answer (the list of dependent files) required roughly 200 tokens worth of information.
The remaining 4,500 tokens were wasted on discovery.
```

All of these file contents are now permanent context in the session. Every subsequent message — whether about debugging, refactoring, or an entirely unrelated task — will re-send these files to the model.

### With Graphify — The Graph-Based Approach

The agent follows the session start protocol:

```
Turn 1: Agent reads HANDOFF.md (or GRAPH_REPORT.md if no HANDOFF exists)
        → Gets architectural overview
        → Approximate cost: 300 tokens

Turn 2: Agent runs: graphify query "which agents depend on or import from experience_optimizer_agent"
        → Graph traversal returns:
          - ats_optimization_agent.py: imports ExperienceOptimizerOutput (line 12)
          - orchestrator/__init__.py: instantiates ExperienceOptimizerAgent (line 45)
          - quality_assurance_agent.py: imports ExperienceOptimizerOutput (line 8)
          - gap_analysis_agent.py: references optimization results (line 67)
          - data_models/strategy.py: defines OptimizationResult model
        → Approximate cost: 250 tokens

Turn 3: Agent reads only the specific lines it needs to verify
        → sed -n '40,50p' src/orchestrator/__init__.py
        → sed -n '8,15p' src/agents/quality_assurance_agent.py
        → Approximate cost: 150 tokens

Cumulative cost for navigation: approximately 700 tokens.
Answer acquired with 85% fewer tokens than the traditional approach.
```

The difference compounds across a session. The traditional approach leaves 4,500 tokens of file contents in the context that are paid for on every subsequent turn. The graph-based approach leaves 400 tokens of targeted information. Over a 20-turn debugging session, this single optimization can save 80,000 tokens.

### The Multiplication Effect

The savings are not merely additive — they multiply because of how context compounding works. Every file read in the traditional approach becomes permanent context. If the session continues for 15 more turns, each of those turns pays for all accumulated file reads. The graph-based approach not only uses fewer tokens for the initial discovery but also leaves less accumulated context for future turns to carry.

Documented benchmarks from production codebases show that replacing file-based discovery with graph-based discovery reduces per-session token consumption by 40–70% for navigation-heavy tasks. The exact savings depend on codebase size and session length, but the pattern is consistent: the graph eliminates the largest category of token waste — orientation — and prevents it from compounding.

---

## 8. Project Setup — What Is Committed vs. What Is Manual

A key design decision in integrating graphify into a project is determining which artifacts belong in version control and which are per-machine setup steps. The Resume Tailor project follows a specific division that balances developer convenience with the technical constraints of the tools involved.

### Committed Artifacts — Everyone Gets These Automatically

**`.graphifyignore`** — The indexing configuration. This file tells graphify which files to include and exclude. By committing it, every developer who runs `graphify .` gets the same graph scope without additional configuration. The Resume Tailor project's `.graphifyignore` is configured for code-only extraction of `src/` and `tests/`.

**`pyproject.toml` dependency group** — The `[dependency-groups] graphify` section declares graphify as a development tool dependency. Running `uv sync --group graphify` installs it. This is distinct from the project's runtime dependencies and does not affect production deployments.

**`AGENTS.md` graphify section** — The instruction file tells every agent session to prefer `graphify query` over file reading for codebase navigation. It includes example queries and references the `GRAPH_REPORT.md` location. This section was manually integrated rather than generated by `graphify claude install`, giving the project full control over the wording and placement.

**`.claude/settings.json` hooks** — The PreToolUse hook that blocks broad `grep`/`rg`/`find` commands when a graph exists. This hook was manually integrated into the existing hooks configuration, merging graphify's enforcement with the project's other hook-based rules (block full test suite runs, block `cat src/`, enforce HANDOFF.md before compaction).

**`graphify-out/` directory** — The graph itself. The `graph.json`, `GRAPH_REPORT.md`, and `graph.html` files are committed so that every clone starts with a pre-built map. Two files within `graphify-out/` should be gitignored: `manifest.json` (contains machine-specific timestamps that break after clone) and `cost.json` (tracks per-machine extraction costs). The `cache/` subdirectory can optionally be committed for faster incremental updates or excluded to keep the repository size small.

### Manual Per-Machine Steps — Run Once Per Clone

**Building the initial graph.** Running `graphify .` builds the graph from the current state of the codebase. This is a one-time operation for the first developer who sets up the project; subsequent developers get the pre-built `graphify-out/` from version control. After major structural changes (adding or removing agents, reorganizing modules), any developer can run `graphify . --update` to refresh the graph.

**Installing the git hooks.** Running `graphify hook install` sets up post-commit and post-checkout hooks that automatically rebuild the graph. These hooks live in `.git/hooks/`, which is inside the `.git/` directory and intentionally not tracked by version control. This is a security feature of git — you cannot distribute executable hooks via `git clone`. Each developer must run this command once after cloning.

The git hooks installed by graphify serve two purposes. The post-commit hook runs an incremental AST extraction after every commit, keeping the graph synchronized with the committed code. The post-checkout hook runs a rebuild after switching branches, ensuring the graph reflects the current branch's codebase. The merge driver prevents `graph.json` from accumulating conflict markers when two developers commit graph changes in parallel — it union-merges the graphs automatically.

**Why this distinction exists.** The split between committed configuration and manual setup is not accidental. Configuration files (`.graphifyignore`, hooks in `.claude/settings.json`, instructions in `AGENTS.md`) describe what the project wants — they are declarative and portable. Git hooks describe how to enforce behaviour on a specific machine — they are procedural and machine-specific. The industry standard is to commit the former and document the latter.

---

## 9. Day-to-Day Usage and Queries

Once the graph is built and the hooks are installed, graphify operates in the background. The developer does not need to think about it. The agent's hooks redirect it to graph queries automatically, and the git hooks keep the graph current without manual intervention.

### Query Patterns

The most common query pattern is a natural-language question about codebase structure. The graph traverses nodes and edges to find relevant symbols and returns them with their locations:

```bash
graphify query "how is token refresh handled"
graphify query "which modules import UserRepository"
graphify query "what calls the experience optimizer agent"
graphify query "dependencies of ats_optimization_agent"
```

Queries can be scoped with a token budget to limit the response size:

```bash
graphify query "auth flow" --budget 500
```

For finding the path between two symbols — useful for understanding data flow:

```bash
graphify path "ResumeExtractorAgent" "AtsOptimizationAgent"
```

For getting a focused explanation of a specific symbol:

```bash
graphify explain "ExperienceOptimizerAgent"
```

### Inside the Agent Session

When working inside Claude Code (or any graphify-aware coding assistant), the agent uses these queries transparently. The developer asks a question about the codebase, and the agent resolves it through graph queries rather than file reading. The developer does not need to type `graphify query` themselves — the agent does it as part of its tool loop.

The developer only needs to interact with graphify directly when:
- Building or rebuilding the graph: `graphify .` or `graphify . --update`
- Exporting architecture diagrams: `graphify export callflow-html`
- Running explicit terminal queries to verify the graph's understanding

---

## 10. Keeping the Graph Current — Incremental Updates

The graph is a snapshot of the codebase at a point in time. As code changes, the graph must be updated to remain useful. Graphify provides several update mechanisms with different tradeoffs between automation and control.

### Git Hook — Automatic on Every Commit

The recommended approach is `graphify hook install`, which sets up a post-commit hook. Every time the developer runs `git commit`, the hook triggers an incremental AST extraction. Because the extraction is cached per file by SHA256 hash, only files that actually changed in the commit are re-parsed. For a typical commit that touches 2–3 files, the update completes in under a second.

This approach is fully automated: the developer does not need to remember to update the graph. It happens as a side effect of committing code, which is already part of the development workflow.

The hook also installs a git merge driver for `graph.json`. When two branches modify the graph in parallel and a merge occurs, the merge driver union-merges the two graphs automatically. This prevents conflict markers from appearing in `graph.json`, which would be meaningless for a JSON graph file.

### Manual Incremental Update

For cases where the developer wants to update the graph without committing (e.g., during active development before the code is ready to commit):

```bash
graphify . --update
```

This re-extracts only files whose contents have changed since the last extraction, using the same SHA256 cache mechanism as the git hook. It is fast enough to run at any point during a development session without interrupting flow.

### Full Rebuild

For cases where the project structure has changed significantly — directories renamed, modules reorganized, many files added or removed — a full rebuild ensures the graph is clean:

```bash
graphify . --force
```

The `--force` flag overwrites the existing graph even if the new graph has fewer nodes than the old one. This is important after refactors that delete files: without `--force`, graphify conservatively preserves old nodes to avoid losing information, which can lead to ghost nodes that no longer correspond to any source file.

### Watch Mode

For continuous development where the developer wants the graph to update on every file save:

```bash
graphify ./src --watch
```

This runs as a background process that watches the filesystem and triggers incremental updates on file changes. It is useful during extended refactoring sessions where multiple changes accumulate before a commit. The process can be killed with Ctrl+C when the session ends.

---

## 11. Team Setup and Version Control

Graphify is designed for team workflows. The `graphify-out/` directory is intended to be committed to version control, giving every team member an immediate graph without running their own initial build.

### Recommended Git Configuration

Add these lines to `.gitignore`:

```
graphify-out/manifest.json    # machine-specific timestamps
graphify-out/cost.json        # per-machine cost tracking
# graphify-out/cache/         # optional: commit for speed, exclude to keep repo small
```

The `manifest.json` file contains modification-time-based cache keys that are meaningless on a different machine. Including it would cause unnecessary full rebuilds after every clone. The `cost.json` file tracks per-machine extraction costs and is not relevant to other developers.

The `cache/` directory contains SHA256-hashed extraction results. Committing it makes incremental updates faster because the cache is pre-populated after clone. Excluding it keeps the repository smaller. The tradeoff is speed vs. repository size; for most projects under 100 files, the speed difference is negligible and excluding the cache is the simpler choice.

### Workflow

1. **First developer** runs `graphify .`, reviews the output, and commits `graphify-out/` along with the configuration files (`.graphifyignore`, updated `AGENTS.md`, updated `.claude/settings.json`).

2. **Subsequent developers** clone the repository and get the pre-built graph immediately. Their agent reads `GRAPH_REPORT.md` on the first session. No additional setup is required for graph-based navigation to work.

3. **All developers** run `graphify hook install` once after cloning to enable automatic updates on commit. This is the only per-machine setup step.

4. **After major structural changes** (adding a new agent class, splitting a module, reorganizing directories), any developer can run `graphify . --update` to refresh the graph. The update is committed alongside the code changes so that the graph stays synchronized with the codebase.

### Merge Conflicts

Because `graph.json` is a single JSON file, git might produce conflict markers when two branches modify the graph. Graphify's merge driver prevents this by union-merging graph changes automatically. The merge driver is installed by `graphify hook install` and requires no configuration beyond running that command.

If a merge conflict does occur (e.g., because the merge driver was not installed), the fix is to run `graphify . --force` after resolving the merge, which rebuilds the graph from the current state of the codebase.

---

## 12. Advanced Usage — MCP Server, CI, Headless Extraction

Beyond session-time navigation, graphify offers several advanced usage patterns for teams with more sophisticated workflows.

### MCP Server — Tool-Based Agent Access

Graphify can expose the graph as an MCP (Model Context Protocol) server, giving agents structured tool-based access rather than requiring shell commands:

```bash
python -m graphify.serve graphify-out/graph.json
```

This starts a stdio-based MCP server that provides tools like `query_graph`, `get_node`, `get_neighbors`, and `shortest_path`. The agent can call these tools directly within its tool loop without spawning shell processes for each query. This is more efficient than command-line queries because the graph is loaded once into memory and reused across queries.

Registering with an MCP-compatible client (e.g., Kimi Code, Claude Desktop):

```bash
kimi mcp add --transport stdio graphify -- python -m graphify.serve graphify-out/graph.json
```

The MCP extra (`graphifyy[mcp]`) must be installed for this functionality.

### CI Integration — Headless Graph Updates

For teams that want to enforce graph freshness in CI, graphify provides a headless extraction command:

```bash
graphify extract ./src --backend ollama
```

The `extract` command runs the extraction pipeline without an IDE session. It requires a backend specification because there is no IDE to provide the model API. Available backends include `ollama` (fully local, no API key), `claude` (requires `ANTHROPIC_API_KEY`), `openai` (requires `OPENAI_API_KEY`), `gemini` (requires `GEMINI_API_KEY`), and `bedrock` (uses AWS IAM, no API key).

For code-only projects, the AST extraction runs regardless of backend because tree-sitter is always local. The backend is only needed if the project includes documentation or other non-code artifacts that require semantic extraction.

A CI workflow might:
1. Check out the repository.
2. Run `graphify extract ./src --backend ollama` (or skip if code-only and use `graphify . --update` which requires no backend).
3. Compare the generated `graph.json` with the committed version.
4. Fail the build if the graph is out of date, alerting the developer to run `graphify . --update` and commit the result.

### Cross-Project Global Graph

Graphify supports a global graph that spans multiple repositories:

```bash
graphify global add graphify-out/graph.json my-project
graphify global list                              # show all registered repos
graphify query "auth flow" --global               # query across all registered repos
```

This is useful for developers working across multiple services or microservices, where understanding cross-service dependencies requires a unified view. The global graph is stored at `~/.graphify/global.json` and is not committed to any repository.

### Call-Flow Architecture Diagrams

Graphify can generate Mermaid-based architecture diagrams from the graph:

```bash
graphify export callflow-html
```

This produces an HTML file with Mermaid call-flow diagrams showing the architecture of the codebase, organized by community cluster. The diagrams are interactive and can be opened in any browser. This is useful for architectural reviews, onboarding documentation, and understanding the flow of data through the system.

For the Resume Tailor project, a call-flow diagram might show the pipeline from `resume_extractor_agent.py` through each optimization agent to `ats_optimization_agent.py`, with data models flowing between them.

---

## 13. Troubleshooting and Edge Cases

### Graph Has Fewer Nodes After Update

If a refactor deleted files, the old nodes may linger because graphify conservatively preserves them to avoid data loss. Pass `--force` to overwrite with the new, potentially smaller graph:

```bash
graphify . --update --force
```

### Ghost Duplicate Nodes

If semantic extraction and AST extraction produced different node IDs for the same symbol, the graph may contain duplicates. A full re-extract clears this:

```bash
graphify . --force
```

### Graph Not Updating on Commit

The git hook may not have been installed, or the hook file may lack execute permissions. Verify with:

```bash
graphify hook status
```

Reinstall if needed:

```bash
graphify hook install
```

### Command Not Found After pip install

`pip install graphifyy` installs the script to a user bin directory that may not be on PATH. On Linux, add `~/.local/bin` to PATH. On macOS, add `~/Library/Python/3.x/bin`. The recommended fix is to use `uv tool install graphifyy` instead, which manages PATH automatically.

### Skill Version Mismatch Warning in IDE

The installed graphify package version differs from the skill file version. Update both:

```bash
uv tool upgrade graphifyy
graphify install          # overwrites the skill file
```

### Large Graph HTML Won't Open

For graphs with over 5,000 nodes, the HTML visualization can become too large for browsers. Skip HTML generation and use the graph JSON directly:

```bash
graphify . --no-viz
graphify query "..."      # queries still work against graph.json
```

---

## 14. Alternatives Considered — Why Graphify Over Other Tools

The token efficiency guide documents five approaches to static code intelligence. Understanding why graphify was chosen over the alternatives for this project provides context for the decision and guidance for other projects evaluating similar tools.

### Aider's Repo Map — Mathematical Rigor, Narrow Integration

Aider's repo map uses PageRank-based ranking to select the most relevant symbols for a given task context. It is mathematically the most sophisticated approach and is deeply integrated into Aider's own agent loop. However, it is not a standalone tool — it only operates within Aider sessions. For projects using Claude Code or other agents as their primary interface, Aider's repo map is not accessible.

Graphify was chosen because it is platform-agnostic. It works with Claude Code (the primary agent for this project), Codex, Cursor, and every other major coding assistant through the same installation and query interface.

### CodeGraph — Deeper Integration, Smaller Ecosystem

CodeGraph provides an MCP-native server with deep agent integration. The agent calls `codegraph.search()` as a tool rather than running shell commands. This is architecturally cleaner for MCP-aware agents. However, CodeGraph has a smaller community, less documentation, and narrower platform support compared to graphify. At the time of evaluation, graphify's broader ecosystem and more active development made it the safer choice for a project that may evolve its tooling stack.

### Repomix — Batch Context, Not Session Navigation

Repomix packages an entire codebase into a single compressed file for one-time context feeding — generating AGENTS.md, providing architectural overview, or feeding a complete codebase snapshot to a model. It excels at batch operations but does not support session-time navigation. An agent cannot query a Repomix output to find where a specific function lives without loading the entire compressed archive into context.

Graphify was chosen because the primary use case is session-time navigation: the agent needs to answer "where is X" and "what depends on Y" repeatedly throughout a session without accumulating context.

### Code2Prompt — Template-Driven, Not Queryable

Code2Prompt generates structured prompts from codebases using templates. It is fast and flexible but produces static output. The agent cannot ask follow-up questions or traverse the codebase dynamically. It is best suited for programmatic context generation in LLM pipelines, not for interactive agent sessions.

### The Decision Criteria

For the Resume Tailor project, the evaluation criteria were:
- **Platform support:** Must work with Claude Code as the primary agent interface.
- **Session-time queryability:** The agent must be able to ask targeted questions during a session without loading large artifacts.
- **Zero API cost for code:** The extraction of Python source must be fully local, consuming no tokens and requiring no API keys.
- **Commitability:** The graph must be storable in version control so the team inherits it automatically.
- **Active maintenance:** The tool must be under active development with a responsive maintainer.

Graphify satisfied all five criteria. No alternative satisfied more than three.

---

## 15. Future Considerations and Extensions

The graphify setup documented here is the foundation. Several extensions are worth considering as the project and the tool ecosystem evolve.

### Installing the Leiden Extra for Community Detection

The Leiden algorithm provides more sophisticated community detection than the built-in clustering. It identifies modules that form natural subsystems, which helps the agent understand architectural boundaries. For Python 3.12 (which this project uses), the Leiden extra is compatible:

```bash
uv tool install "graphifyy[leiden]"
```

After installing, rebuild the graph with community detection:

```bash
graphify . --force
graphify . --cluster-only --resolution 1.5  # finer-grained communities
```

The resulting communities appear in `GRAPH_REPORT.md` as named clusters, making the architecture report more informative.

### MCP Server for Tool-Based Queries

If the project adopts MCP-aware agents (Claude Desktop, Kimi Code, or any tool that supports the Model Context Protocol), exposing the graph as an MCP server eliminates the shell-command overhead of graph queries:

```bash
uv tool install "graphifyy[mcp]"
python -m graphify.serve graphify-out/graph.json
```

The agent then calls `query_graph("how is auth handled")` as a direct tool call rather than spawning a shell process. This is more efficient and more reliable because the graph is loaded once into memory.

### CI Enforcement of Graph Freshness

A GitHub Actions workflow that verifies the committed graph matches the current codebase ensures the graph never goes stale. If a developer restructures the codebase without updating the graph, CI catches it before the PR is merged:

```yaml
# .github/workflows/graphify-check.yml
name: Graph Freshness Check
on: [pull_request]
jobs:
  check-graph:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install graphify
        run: uv sync --group graphify
      - name: Check graph freshness
        run: |
          uv run graphify update . --no-cluster
          if ! git diff --quiet graphify-out/graph.json; then
            echo "::error::graphify-out/ is out of date. Run: graphify update . --no-cluster && git add graphify-out/ && git commit"
            exit 1
          fi
```

### Extending to Documentation Indexing

If the project's documentation grows to the point where agents benefit from navigating it structurally, the PDF and Markdown extras can be added to extract documentation into the same graph:

```bash
uv tool install "graphifyy[pdf]"
```

Then update `.graphifyignore` to include the `docs/` directory. The graph would then contain nodes for documentation sections, linked to the code they reference. An agent could ask "where is the token efficiency strategy documented" and get a graph result linking to both the code implementing the strategy and the documentation explaining it.

### Neo4j for Persistent, Multi-Project Graphs

For teams working across multiple repositories, pushing the graph to a Neo4j database enables cross-project queries:

```bash
uv tool install "graphifyy[neo4j]"
graphify . --neo4j-push bolt://localhost:7687
```

The Neo4j graph can be queried with Cypher for complex multi-hop traversals that are impractical in memory. This is most valuable for organizations with many interdependent services where understanding cross-service dependencies is a recurring challenge.

---

*This document reflects the state of graphify as of version 0.8.30 (June 2026) and the integration decisions made for the Resume Tailor project. The tool is under active development; consult the repository at `github.com/safishamsi/graphify` for the latest features and configuration options.*
