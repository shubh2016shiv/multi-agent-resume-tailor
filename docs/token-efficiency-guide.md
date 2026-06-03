# The Practical Guide to Token Efficiency in Agentic Coding
## How to Reduce Cost, Manage Context, and Build Sustainable AI-Assisted Development Workflows

> **Scope:** Claude Code · OpenAI Codex CLI · GitHub Copilot · Aider · Cursor · Windsurf  
> **Audience:** Backend Engineers · GenAI Engineers · Engineering Teams  
> **Edition:** 2025–2026  

---

## Table of Contents

1. [Why This Problem Is Harder Than It Looks](#1-why-this-problem-is-harder-than-it-looks)
2. [The Mechanics of Context Explosion](#2-the-mechanics-of-context-explosion)
3. [The Token Drain Taxonomy](#3-the-token-drain-taxonomy)
4. [Layer 1 — The Context Firewall (Project Setup)](#4-layer-1--the-context-firewall)
5. [Layer 2 — Static Code Intelligence (Knowledge Graphs & Repo Maps)](#5-layer-2--static-code-intelligence)
   - 5.1 [Aider's Repo Map — PageRank-Based Context Selection](#51-aiders-repo-map)
   - 5.2 [Graphify — Semantic Knowledge Graph](#52-graphify)
   - 5.3 [CodeGraph — Pre-Indexed MCP Server](#53-codegraph)
   - 5.4 [Repomix — On-Demand Codebase Packaging](#54-repomix)
   - 5.5 [Code2Prompt — Template-Driven Context Generation](#55-code2prompt)
   - 5.6 [Comparison Table](#56-comparison-table)
6. [Layer 3 — Session Discipline](#6-layer-3--session-discipline)
   - 6.1 [The Query Order Protocol](#61-the-query-order-protocol)
   - 6.2 [Compaction Strategy](#62-compaction-strategy)
   - 6.3 [Subagent Isolation](#63-subagent-isolation)
   - 6.4 [Cross-Session Memory via HANDOFF.md](#64-cross-session-memory-via-handoffmd)
7. [Layer 4 — Prompt Caching](#7-layer-4--prompt-caching)
   - 7.1 [How Caching Works Mechanically](#71-how-caching-works-mechanically)
   - 7.2 [Anthropic Prompt Caching](#72-anthropic-prompt-caching)
   - 7.3 [OpenAI Prompt Caching](#73-openai-prompt-caching)
   - 7.4 [What Silently Breaks the Cache](#74-what-silently-breaks-the-cache)
8. [Layer 5 — Model Tiering](#8-layer-5--model-tiering)
9. [Layer 6 — Hooks and Enforcement](#9-layer-6--hooks-and-enforcement)
   - 9.1 [Claude Code Hooks System](#91-claude-code-hooks-system)
   - 9.2 [Practical Hook Patterns](#92-practical-hook-patterns)
10. [Platform-Specific Playbooks](#10-platform-specific-playbooks)
    - 10.1 [Claude Code](#101-claude-code)
    - 10.2 [OpenAI Codex CLI](#102-openai-codex-cli)
    - 10.3 [GitHub Copilot](#103-github-copilot)
    - 10.4 [Aider](#104-aider)
    - 10.5 [Cursor](#105-cursor)
11. [Measuring What You Are Spending](#11-measuring-what-you-are-spending)
12. [Anti-Patterns Reference](#12-anti-patterns-reference)
13. [Organizational Governance Framework](#13-organizational-governance-framework)

---

## 1. Why This Problem Is Harder Than It Looks

Most engineers discover the token cost problem after the damage is done. The first week of using an AI coding agent feels transformative. The second week, the bills arrive. The pattern is consistent enough across teams that it has become something of an industry rite of passage: initial enthusiasm, growing usage, unexpected invoice, retrospective investigation, scrambled cost controls.

The instinctive response is to blame prompt quality. If prompts are expensive, surely better prompts will be cheaper? This assumption is wrong in almost every material case. Careful investigation — documented publicly by engineers at AI companies and in production codebases — consistently shows that the prompt you actually typed is a small fraction of total token consumption. The bulk of the cost is architectural: it comes from context the agent accumulated, context it loaded without being asked, and context it re-processed dozens of times because no one told it not to.

A 2025 Stanford study referenced across multiple engineering blogs found developers waste thousands of tokens daily from unchecked context accumulation. A widely cited practitioner benchmark measured that 60–80% of tokens consumed in typical agent sessions go toward orientation — the agent figuring out where things are — rather than toward answering your actual question. In one documented case, an agent read 25 files to answer a question about three functions. That interaction cost 12,000 tokens. The answer required 800.

Understanding this means understanding the mechanism, not just applying tips. Every optimization described in this document is grounded in exactly how agents build and consume context at the API level. Without that foundation, the techniques feel arbitrary. With it, they become obvious.

---

## 2. The Mechanics of Context Explosion

To understand why costs compound so aggressively, you need a clear mental model of what the LLM receives on every turn of a session.

```
WHAT THE MODEL ACTUALLY RECEIVES ON EVERY SINGLE MESSAGE
──────────────────────────────────────────────────────────

Turn 1, Message 1:
┌─────────────────────────────────────────────────────┐
│  System Prompt (CLAUDE.md / AGENTS.md)              │ ← always loaded
│  + Tool definitions (MCP tools, bash, read, etc.)   │ ← always loaded
│  + Your message: "Fix the auth bug"                 │ ← the thing you typed
└─────────────────────────────────────────────────────┘

Turn 1, Message 2 (after agent reads 3 files):
┌─────────────────────────────────────────────────────┐
│  System Prompt                                      │ ← re-sent
│  + Tool definitions                                 │ ← re-sent
│  + Message 1: "Fix the auth bug"                    │ ← re-sent
│  + Agent response 1                                 │ ← re-sent
│  + File 1 contents (tool result)                    │ ← re-sent
│  + File 2 contents (tool result)                    │ ← re-sent
│  + File 3 contents (tool result)                    │ ← re-sent
│  + Your follow-up: "It still fails on line 47"      │ ← the new thing
└─────────────────────────────────────────────────────┘

Turn 1, Message 20 (a normal debugging session):
┌─────────────────────────────────────────────────────┐
│  System Prompt                                      │
│  + Tool definitions                                 │
│  + Messages 1–19 (everything said and done)         │
│  + ALL file reads from the entire session           │
│  + ALL tool outputs (bash results, test outputs)    │
│  + Your message 20                                  │
└─────────────────────────────────────────────────────┘
   ↑ Message 20 pays for everything above it.
     This is the compounding cost problem.
```

This architecture is not a bug. It is how transformer-based models work: they have no persistent memory between turns, so the entire conversation must be replayed on each inference call. Understanding this changes how you think about session length. A 40-message debugging session does not cost 40× the cost of a 1-message interaction. It costs roughly 40² ÷ 2 times more than a single message, because the context grows with every turn and every turn pays for all prior context.

The second compounding factor is file reads. When an agent opens a file to answer a question, the full file content becomes a tool result in the conversation history. It stays there for the remainder of the session. Every subsequent message pays for it again. If the agent reads five 300-line files early in a session — a completely normal behaviour — those 1,500 lines of code are paid for on every remaining turn.

```
COST CURVE: Why sessions get expensive fast

Token cost
    │
    │                                         ●
    │                                    ●
    │                               ●
    │                          ●
    │                     ●
    │                ●
    │           ●
    │      ●
    │  ●
    └──────────────────────────────────────── Conversation turns
       1   3   5   7   10  15  20  25  30

● Each dot = cost of one message
  Cost grows because every message pays for all prior context.
  This is NOT linear growth. It is closer to quadratic growth
  as file reads and tool outputs accumulate in history.
```

---

## 3. The Token Drain Taxonomy

Before applying solutions, it helps to classify where tokens actually go. Engineering teams that have instrumented their agent usage report broadly consistent proportions.

```
WHERE YOUR TOKENS ACTUALLY GO (typical large-codebase session)

┌────────────────────────────────────────────────────────────────┐
│                    TOTAL SESSION TOKENS                        │
├─────────────────────┬──────────────────────────────────────────┤
│ Orientation         │ ████████████████████████████  60–80%     │
│ (finding things)    │                                          │
├─────────────────────┼──────────────────────────────────────────┤
│ Repeated context    │ ████████████  10–20%                     │
│ (re-reading files,  │                                          │
│  re-sending history)│                                          │
├─────────────────────┼──────────────────────────────────────────┤
│ System overhead     │ ████████  8–12%                          │
│ (CLAUDE.md, tools,  │                                          │
│  always-on rules)   │                                          │
├─────────────────────┼──────────────────────────────────────────┤
│ Actual work         │ ████  5–15%                              │
│ (your prompts +     │                                          │
│  meaningful answers)│                                          │
└─────────────────────┴──────────────────────────────────────────┘
```

**Orientation tokens** are the largest category and the most addressable. They are consumed when the agent has no structural understanding of the codebase and compensates by reading broadly. It opens files it doesn't need, runs broad grep searches, reads configuration files, and builds understanding incrementally through file reads — each of which compounds the context size.

**Repeated context** is the second largest category. File contents read in turn 2 are still in the context window at turn 35. If the agent re-reads a file it already has — something agents do surprisingly often — that cost is doubled. Conversation history, once accumulated, persists until a `/clear` or compaction.

**System overhead** is fixed per session but not zero. A bloated AGENTS.md of 2,000 tokens loaded every session across 50 sessions per week costs 100,000 tokens per week in overhead alone, before a single line of code is touched.

**Actual work** — the part that produces value — is the smallest proportion of total consumption. This is the counterintuitive truth at the heart of token optimization: the value is cheap; the inefficiency is expensive.

---

## 4. Layer 1 — The Context Firewall

The context firewall is the first and cheapest line of defence. It is a set of files that tell agents what they should not proactively load. This operates at the signal layer: agents use these files to decide what to include in context before making a request. The cost is zero beyond initial setup.

### The .claudeignore File

Claude Code respects a `.claudeignore` file in the project root, with identical syntax to `.gitignore`. Files matching patterns in this file are excluded from proactive context inclusion. Benchmarks show that a well-configured `.claudeignore` can produce an 85% context reduction on the signal layer alone. The recommended baseline for a backend project is:

```
# Build outputs — never relevant
node_modules/
dist/
build/
.next/
out/
__pycache__/
.venv/
venv/

# Lock files — large, noisy, never read
*.lock
package-lock.json
poetry.lock

# Coverage and test outputs
coverage/
htmlcov/
.nyc_output/

# Minified and generated
*.min.js
*.min.css
*.generated.*
*.pb
*.onnx

# Large binary model files
*.gguf
*.bin
*.safetensors
*.pkl
*.pt

# Data directories (common in GenAI projects)
data/raw/
data/processed/
chroma_db/
faiss_index/
.chroma/
qdrant_storage/

# Secrets and environment
.env
.env.*
secrets.*

# Evaluation fixtures
evals/fixtures/
evals/datasets/
```

Copy this file to `.aiderignore` for Aider and reference the same patterns in your `.github/copilot-instructions.md` exclusion configuration. Once committed, every developer on the team automatically inherits the same context discipline.

### The AGENTS.md / CLAUDE.md File — Keep It Lean

The instruction file (`AGENTS.md`, `CLAUDE.md`, `.cursorrules`) is loaded before every session and on every API call within a session. This makes it simultaneously the most powerful context tool and the most expensive if misused. Every line you add to this file costs tokens on every single interaction with the project, forever.

The practical rule, documented by practitioners who have shipped production AI systems, is: this file should contain only things the agent cannot discover by reading code. Stack name and version, test command, build command, critical conventions that violate natural patterns — these belong here. Architecture documentation, detailed API references, lengthy onboarding guides — these do not. A bloated instruction file that loads 2,100 tokens of context where only 300 are relevant per session (a real measurement from a production case study published on Medium) is spending 1,800 tokens per session on waste.

### Path-Scoped Rules — The Right Way to Store Domain Rules

Rather than putting all rules in one global file, modern tooling supports path-scoped rules that load only when the agent accesses files in a specific directory. Claude Code reads `.claude/rules/<path>/RULES.md` and applies it only when touching files under that path. Cursor uses `.cursor/rules/*.mdc` with glob-pattern frontmatter. GitHub Copilot uses `.github/instructions/*.instructions.md` with `applyTo` frontmatter.

```
PROJECT ROOT
├── AGENTS.md                    ← global rules, always loaded, keep under 60 lines
└── .claude/
    └── rules/
        ├── api/RULES.md         ← loads ONLY when touching src/api/ files
        ├── pipelines/RULES.md   ← loads ONLY when touching src/pipelines/ files
        ├── agents/RULES.md      ← loads ONLY when touching src/agents/ files
        └── evals/RULES.md       ← loads ONLY when touching scripts/evals/ files
```

One documented team reduced always-loaded rule overhead from 1,358 lines to 807 lines — a 41% reduction — simply by moving domain-specific rules into scoped subdirectories while keeping only genuinely global rules in the root file.

---

## 5. Layer 2 — Static Code Intelligence

This is the layer that most directly addresses the orientation problem: the 60–80% of tokens spent on figuring out where things are. The tools in this layer build persistent, structured representations of the codebase that agents can query efficiently rather than navigating through file reads.

### 5.1 Aider's Repo Map

Aider's repo map is the most mathematically rigorous approach to this problem and the design that has most influenced the broader ecosystem. Understanding how it works explains why it is so much more efficient than naive file reading.

**The mechanism.** Aider uses tree-sitter, a language-agnostic incremental parsing library, to parse every file in the repository and extract two things: symbol definitions (functions, classes, methods) and symbol references (calls, imports, type usage). From these, it builds a directed dependency graph where each source file is a node and edges connect files that have dependencies on each other.

Once the graph is built, Aider applies PageRank — the same algorithm Google originally used to rank web pages — to score each node. The key insight is that PageRank measures influence: a function called by twenty other files is more important context for understanding the codebase than a private helper called once. The personalization vector for PageRank is biased toward the files currently in the chat, with edge weight multipliers for mentioned identifiers (10×), well-named identifiers (10×), and files explicitly added to chat (50×).

```
AIDER REPO MAP — HOW IT BUILDS CONTEXT

All source files
      │
      ▼
┌─────────────────┐
│   tree-sitter   │  ← zero LLM calls, pure AST parsing
│  Parse all files│  ← SQLite cache: only re-parses changed files
└────────┬────────┘
         │  extracts
         │  definitions + references
         ▼
┌─────────────────┐
│ Dependency Graph│  file A → file B (A references something defined in B)
│ (NetworkX)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    PageRank     │  ← personalized to currently-open files
│  (relevance     │  ← higher score = more important given current task
│   scoring)      │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│  Binary search to fit within token budget       │
│  (default: 1,024 tokens, set via --map-tokens)  │
│  Renders as: file path → function signatures   │
│  (not full file contents — signatures only)     │
└─────────────────────────────────────────────────┘
         │
         ▼
   ~1,024 tokens of precisely relevant structural context
   vs. 12,000–50,000 tokens from reading files directly
```

The output is a compact tree showing file paths and function signatures — not full file contents. The agent understands the structure of the entire codebase in roughly 1,024 tokens rather than reading 25 files and consuming 12,000. The SQLite cache means subsequent sessions only re-parse changed files, so startup cost drops to near zero after the first run.

**Architect mode (--architect).** Aider offers a two-model planner-executor pattern where a more powerful model reasons about what change to make, and a cheaper model applies the actual edits. This is particularly effective for backend engineers working on complex service logic: use Opus or GPT-4o for the planning turn, then Haiku or a local model like DeepSeek Coder for the edit application.

```bash
# Basic usage — add only the files you need to touch
aider src/services/auth_service.py tests/test_auth.py

# Tight token budget on repo map
aider --map-tokens 1024 src/services/auth_service.py

# Architect mode — strong planner, cheap executor
aider --architect --model anthropic/claude-opus-4-6 \
      --editor-model deepseek/deepseek-coder \
      src/pipelines/rag_pipeline.py

# DeepSeek Coder for routine edits (10x cheaper than Sonnet)
aider --model deepseek/deepseek-coder src/api/routes.py
```

**What NOT to do with Aider:**
- Do not use `/add src/` to add entire directories. This bypasses the repo map's efficiency and loads all file contents.
- Do not set `--map-tokens` above 2,048 unless the codebase is genuinely too large and the agent is missing critical context. Higher values add overhead to every turn.

### 5.2 Graphify

Graphify takes a broader approach than Aider's repo map. Where Aider focuses on code structure, Graphify builds a semantic knowledge graph that can incorporate code, documentation, database schemas, PDFs, images, and other project artifacts into a single queryable graph. It is particularly relevant for GenAI engineers whose projects blend code with documentation, research papers, and data specifications.

**How it works.** Graphify operates in two passes. The first pass uses tree-sitter AST parsing — consuming zero LLM tokens — to extract classes, functions, imports, and call graphs across 25 supported languages. The second pass, which is optional, uses LLM-based semantic extraction for non-code files (PDFs, images, markdown documentation) where structural parsing is insufficient. The result is a NetworkX graph with Leiden community detection applied to identify clusters of related concern.

```
GRAPHIFY PIPELINE

Your codebase + docs
        │
        ▼
┌───────────────────────────────────────────┐
│  Pass 1: Tree-sitter AST (ZERO LLM tokens)│
│  ─ functions, classes, imports            │
│  ─ call graphs, dependency edges          │
│  ─ 25 languages supported                │
│  ─ SHA256 per-file cache (incremental)   │
└────────────────────┬──────────────────────┘
                     │
                     ▼
┌───────────────────────────────────────────┐
│  Pass 2: LLM semantic extraction          │
│  (OPTIONAL — only for docs/images/PDFs)   │
│  ─ intent, relationships beyond syntax    │
└────────────────────┬──────────────────────┘
                     │
                     ▼
┌───────────────────────────────────────────┐
│  NetworkX Graph + Leiden Clustering       │
│  ─ communities of related modules         │
│  ─ "god nodes" (high-centrality symbols) │
│  ─ cross-cutting dependencies visible     │
└────────────────────┬──────────────────────┘
                     │
            ┌────────┴────────┐
            ▼                 ▼
  graphify-out/          graphify-out/
  GRAPH_REPORT.md        graph.json
  (human + agent         (queryable,
   readable summary)      persistent)
```

The `GRAPH_REPORT.md` is a structured architecture summary that agents read at session start instead of scanning source files. The `graph.json` is queryable via `graphify query "<natural language question>"` and returns precise, low-token answers about the codebase structure.

**Setup:**
```bash
pip install graphifyy

# Build the graph (first run: 30–90s, subsequent runs: incremental)
graphify .

# Wire into your coding agent
graphify claude  install    # Claude Code
graphify codex   install    # OpenAI Codex CLI
graphify cursor  install    # Cursor
graphify aider   install    # Aider
graphify copilot install    # GitHub Copilot CLI

# Query without opening the agent
graphify query "which services depend on UserRepository"
graphify query "how is the embedding pipeline triggered"
graphify query "what calls the token refresh endpoint"

# Keep graph current after major changes
graphify . --update

# Recommended: wire into CI
# In .github/workflows/graph-update.yml:
# on: push → graphify . --update → commit graphify-out/
```

Token savings benchmarks documented in production: 6.8× to 49× per navigation task compared to blind file reading, depending on codebase size and query type.

### 5.3 CodeGraph

CodeGraph provides an MCP server that pre-indexes your repository and exposes it as a tool that agents can query directly within their tool loop — without switching to a command line. It integrates with Claude Code, Cursor, Codex CLI, Gemini CLI, and others through automatic configuration detection.

```bash
npx @colbymchenry/codegraph

# Initialize and build the index
codegraph init -i

# The MCP server then becomes available inside your agent sessions.
# Agent can call: codegraph.search("token refresh logic")
# Instead of: reading 15 files to find token refresh logic

# Keep the index current
codegraph sync          # manual sync after changes
# Or set up file-watcher: auto-syncs on save (2-3 second delay)
```

The key difference from Graphify is integration depth: CodeGraph operates as an MCP tool directly callable within agent sessions, making the pattern completely transparent to the developer. The agent queries the graph as part of its natural tool loop rather than requiring a separate command.

### 5.4 Repomix

Repomix addresses a different problem: not per-session navigation, but batch context preparation. Its primary use cases are generating or updating your project's AGENTS.md from an existing codebase, providing an overview for architectural review sessions, and feeding context to a model that needs a complete picture of the project once rather than repeatedly.

```bash
npm install -g repomix

# Basic — compress and pack the repo
repomix --compress \
  --ignore "**/*.lock,**/dist/**,**/__pycache__/**,**/*.min.*,**/*.gguf,**/*.bin"

# Output: repomix-output.xml
# This file contains your entire codebase, compressed, in one XML document.
# Feed it to Claude or GPT-4o and say:
# "Read this codebase snapshot and generate AGENTS.md for this project."
# Then delete repomix-output.xml — do not commit it.

# Token count before committing to a session:
repomix --output-show-line-numbers \
  --ignore "**/*.lock,**/dist/**" src/
# Review the token count per file. If any file exceeds 2,000 tokens alone,
# consider adding it to .claudeignore for sessions where it is not relevant.
```

Repomix is also useful for understanding the cost profile of your codebase before starting a large session. It reports token count per file, which tells you exactly which files are most expensive and whether specific directories should be excluded.

### 5.5 Code2Prompt

Code2Prompt is a Rust-based CLI tool that generates structured prompts from your codebase using Handlebars templates. It is fastest tool in this category — benchmarks show it processing the Next.js repo in 5 seconds compared to Repomix's 22 minutes. The template system gives precise control over what context is generated and in what format.

```bash
# Install
cargo install code2prompt
# or: pip install code2prompt-core

# Generate a structured prompt with source tree
code2prompt src/ --template default --output context.md

# Include only specific file types
code2prompt src/ --include "**/*.py" --exclude "**/*test*"

# Generate git-diff focused context for code review
code2prompt . --diff HEAD~3..HEAD

# JSON output for programmatic processing
code2prompt src/ --format json
```

Code2Prompt is most useful in GenAI engineering workflows where you are programmatically building context for LLM pipelines or need to feed structured codebase context to evaluation scripts.

### 5.6 Comparison Table

```
┌─────────────────┬──────────────┬─────────────┬────────────┬─────────────────┐
│ Tool            │ Mechanism    │ LLM Tokens  │ Best For   │ Integration     │
│                 │              │ at Index    │            │                 │
├─────────────────┼──────────────┼─────────────┼────────────┼─────────────────┤
│ Aider Repo Map  │ tree-sitter  │ 0           │ Agent-     │ Built into      │
│                 │ + PageRank   │             │ native nav │ Aider           │
├─────────────────┼──────────────┼─────────────┼────────────┼─────────────────┤
│ Graphify        │ tree-sitter  │ 0 (code)    │ Large      │ Claude/Codex/   │
│                 │ + LLM (docs) │ optional    │ mixed      │ Cursor/Aider    │
│                 │ + Leiden     │ (docs only) │ repos      │ via skill+hook  │
├─────────────────┼──────────────┼─────────────┼────────────┼─────────────────┤
│ CodeGraph       │ AST + vector │ 0           │ MCP-native │ MCP server,     │
│                 │ index        │             │ agent work │ auto-detected   │
├─────────────────┼──────────────┼─────────────┼────────────┼─────────────────┤
│ Repomix         │ File packing │ N/A (batch) │ AGENTS.md  │ CLI, npx, MCP   │
│                 │ + compress   │             │ generation │ server          │
├─────────────────┼──────────────┼─────────────┼────────────┼─────────────────┤
│ Code2Prompt     │ Template-    │ N/A (batch) │ Pipeline   │ CLI, Rust       │
│                 │ driven pack  │             │ context    │ binary          │
└─────────────────┴──────────────┴─────────────┴────────────┴─────────────────┘
```

**Recommendation for backend + GenAI projects:** Use Graphify as the primary session-time navigator, Repomix once to generate your AGENTS.md from an existing codebase, and Aider for any heavy multi-file refactoring work where its repo-map + git-native approach is worth the CLI overhead.

---

## 6. Layer 3 — Session Discipline

Even with a perfect context firewall and an indexed codebase, individual session behaviour has a large effect on cost. These disciplines are behavioural: they require consistent habits from both the engineer and the agent.

### 6.1 The Query Order Protocol

The most important single habit is establishing a query order and following it every time. When an agent follows this order, it cannot waste tokens on orientation, because orientation is handled by the graph before any file is opened.

```
CORRECT QUERY ORDER (follow every time)

TASK ARRIVES
     │
     ▼
 ┌─────────────────────────────────────────┐
 │  Step 1: READ HANDOFF.md               │
 │  If it exists → this is your context.  │
 │  Do not read source files yet.         │
 └──────────────────┬──────────────────────┘
                    │ (if no HANDOFF.md)
                    ▼
 ┌─────────────────────────────────────────┐
 │  Step 2: QUERY THE GRAPH               │
 │  graphify query "<what you need>"      │
 │  codegraph.search("<what you need>")   │
 │  Read GRAPH_REPORT.md for orientation  │
 └──────────────────┬──────────────────────┘
                    │ (only if graph answer is insufficient)
                    ▼
 ┌─────────────────────────────────────────┐
 │  Step 3: TARGETED FILE READ            │
 │  Read ONLY the specific function/range │
 │  sed -n '45,90p' src/services/auth.py  │
 │  NOT: cat src/services/auth.py         │
 └──────────────────┬──────────────────────┘
                    │
                    ▼
 ┌─────────────────────────────────────────┐
 │  Step 4: MAKE THE CHANGE               │
 └──────────────────┬──────────────────────┘
                    │
                    ▼
 ┌─────────────────────────────────────────┐
 │  Step 5: TARGETED TEST                 │
 │  pytest tests/services/test_auth.py    │
 │        ::test_token_refresh -x         │
 │  NOT: pytest tests/ (full suite)       │
 └─────────────────────────────────────────┘
```

The enforcement of this order — moving from graph query to targeted file read rather than jumping to file reads directly — is where the majority of per-session savings are realised.

### 6.2 Compaction Strategy

Context compaction (sometimes called summarisation) replaces accumulated conversation history with a compact summary, freeing up context space while preserving essential information. All major platforms support this, either automatically or manually.

**The critical rule is: compact manually at 60–65% context fill.** Do not wait for automatic compaction at 90–95%. By the time automatic compaction triggers, the model has been paying for bloated context for a long time. Proactive compaction preserves your control over what gets summarised and what survives.

In Claude Code, manual compaction is triggered with `/compact`. You can optionally pass instructions: `/compact summarise only modified files, current blockers, and next step`. In Aider, `/drop` removes specific files from context without ending the session. In OpenAI Codex, context pruning is available via the conversation management API.

**Before compacting, always customise what survives.** Add to your AGENTS.md or CLAUDE.md:

```
When compacting, always preserve:
- The full list of files modified in this session
- Any test commands that were run and their outcomes
- The current blocker or next specific action
- Any architectural decision made during this session
```

**The difference between /compact and /clear.** `/compact` creates a summary and continues the session with that summary as context. Use it when you want to continue the same task across a context boundary. `/clear` wipes everything and starts fresh. Use it between completely unrelated tasks — never carry the context from debugging an auth bug into refactoring the embedding pipeline.

### 6.3 Subagent Isolation

Subagents are one of the most powerful but least-used token efficiency tools. A subagent is a separate agent process with its own context window. The main session delegates a bounded task to a subagent, receives a summary of the result, and continues without the subagent's detailed work ever entering the main context window.

This is the correct pattern for any task that involves: reading a lot of files to produce a small answer, running tests or builds with verbose output, doing exploratory research, reading external documentation, or running evaluation harnesses.

```
WITHOUT SUBAGENTS (expensive):
──────────────────────────────
Main session context:
  [... 20 turns of auth work ...]
  + Read 8 files investigating which services use UserRepository
  + Read 3 test files
  + Run tests: 3,000 tokens of test output
  + Results: "UserService and ProfileService both use it"
  Total cost: adds ~8,000 tokens to main context permanently

WITH SUBAGENTS (cheap):
───────────────────────
Main session context:
  [... 20 turns of auth work ...]
  + "Use a subagent to investigate which services use UserRepository
     and return a summary in under 200 tokens"
  + Subagent summary: "UserService (src/services/user.py:45) and
    ProfileService (src/services/profile.py:112) use UserRepository.
    No others. Both use async get_by_id only."
  Total cost: adds ~200 tokens to main context
  Subagent context: discarded after summary
```

In Claude Code: `Use subagents to investigate X and report a summary`. In practice, experienced Claude Code users delegate all research tasks, log analysis, eval runs, and documentation lookups to subagents as a default habit.

### 6.4 Cross-Session Memory via HANDOFF.md

The HANDOFF.md pattern is the practical solution to Situation 2 from the problem framing: ongoing work across multiple sessions where the agent would otherwise re-read the entire codebase at the start of each session.

The pattern is simple. At the end of every session where files were modified, the agent writes a structured summary to `HANDOFF.md` at the project root. At the start of the next session, the agent reads `HANDOFF.md` before touching any source file. A well-written HANDOFF.md gives the agent everything it needs to continue work: what was done, what is broken, which files were touched, and exactly what the next step is — without opening a single source file.

```
HANDOFF.md — TEMPLATE

# HANDOFF.md
_Updated: 2026-06-03T14:23:00Z | Goal: Add retry logic to LLM client_

## Completed
- Added `call_with_retry()` to src/llm/client.py (lines 45–87)
- 3 retries, exponential backoff, logs on each retry

## Current state
- ✅ Unit test passes: tests/llm/test_client.py::test_retry_on_timeout
- ❌ Integration test failing: tests/integration/test_rag_pipeline.py::test_embedding
  Error: "RetryError: max retries exceeded" — embedding model timeout > retry window
- Not started: updating RAGPipeline to pass timeout config to client

## Files modified this session
| File | Change | Why |
|------|--------|-----|
| src/llm/client.py | Added call_with_retry() at line 45 | Upstream flakiness |
| tests/llm/test_client.py | Added test_retry_on_timeout | Coverage |

## Next exact action
- [ ] Increase timeout config in src/config/llm_config.py line 23 from 10s to 30s
- [ ] Re-run: pytest tests/integration/test_rag_pipeline.py::test_embedding -x

## DO NOT TOUCH
- src/pipelines/rag_pipeline.py — unrelated refactor in progress on branch feat/rag-v2
```

A HANDOFF.md like this eliminates the need for the next session to read any source file before starting work. It replaces ~15,000 tokens of codebase re-reading with ~500 tokens of structured summary.

A community-standardised version of this pattern is available via `session-handoff` on LobeHub, which adds timestamped entries, git commit hashes, and validation commands to the format.

---

## 7. Layer 4 — Prompt Caching

Prompt caching is a server-side mechanism offered by both Anthropic and OpenAI that stores a processed version of your context and serves it at a fraction of the normal input token price on subsequent requests. It is one of the highest-ROI cost levers available, particularly for API-based workflows and for teams building agent pipelines on top of these models.

### 7.1 How Caching Works Mechanically

When you send a request to Claude or GPT-4o, the model processes every token in the input — system prompt, conversation history, tool definitions, everything — from scratch. This processing has a cost in both tokens and latency. Prompt caching changes this: the model stores a processed version of a stable prefix of your request on the server. When the next request begins with that same prefix, the server loads the stored version instead of reprocessing it.

```
WITHOUT CACHING (full cost every turn):
────────────────────────────────────────
Turn 1:  [System: 2,000 tokens] + [Tools: 1,500 tokens] + [User: 50 tokens]
         → pay full price for 3,550 tokens

Turn 2:  [System: 2,000 tokens] + [Tools: 1,500 tokens] + [Turn 1: 500 tokens] + [User: 50 tokens]
         → pay full price for 4,050 tokens

Turn 10: [System: 2,000 tokens] + [Tools: 1,500 tokens] + [Turns 1–9: 4,500 tokens] + [User: 50 tokens]
         → pay full price for 8,050 tokens

WITH CACHING (system + tools cached after Turn 1):
───────────────────────────────────────────────────
Turn 1:  [System: 2,000 tokens] + [Tools: 1,500 tokens] + [User: 50 tokens]
         → pay full price + 25% cache write premium = 3,550 × 1.25

Turn 2:  [System: 200 tokens ≈ 10%] + [Tools: 150 tokens ≈ 10%] + [Turn 1: 500] + [User: 50]
         → cached prefix costs 10% of normal input price

Turn 10: [System: 200] + [Tools: 150] + [Turns 1–9: 4,500] + [User: 50]
         → only the new turns pay full price; system + tools cost 10%

Cumulative savings over 10-turn session with 3,500-token stable prefix:
  ≈ 85% reduction on that prefix across turns 2–10
```

### 7.2 Anthropic Prompt Caching

Anthropic's prompt caching uses explicit `cache_control` breakpoints in API requests. You mark specific content blocks as cacheable. The cache TTL is 5 minutes by default, extended on each access. A 1-hour TTL is available at a higher write rate.

```python
# Python SDK — explicit cache_control breakpoints
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": "You are a backend engineer's assistant...",
            "cache_control": {"type": "ephemeral"}  # ← cache this block
        }
    ],
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "<large_codebase_context>...</large_codebase_context>",
                    "cache_control": {"type": "ephemeral"}  # ← and this
                },
                {
                    "type": "text",
                    "text": "Now fix the auth bug in the session handler."
                    # ← this is the variable part — NOT cached
                }
            ]
        }
    ]
)

# Check if the cache hit
print(response.usage.cache_read_input_tokens)   # > 0 means cache hit
print(response.usage.cache_creation_input_tokens)  # > 0 means cache miss (write)
```

For multi-turn conversations, use automatic caching:

```python
# Automatic caching for multi-turn — simplest approach
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system="Your system prompt here...",
    messages=conversation_history,
    # Add at the top level — system handles breakpoint placement automatically
    betas=["prompt-caching-2024-07-31"]
)
```

**Key pricing facts for Anthropic (mid-2026):**
- Cache write: 1.25× standard input token price (one-time, first call)
- Cache read: 0.10× standard input token price (every subsequent call)
- Break-even point: if you use the same prefix more than twice, caching saves money
- Minimum cacheable block: 1,024 tokens (Sonnet/Opus), 2,048 tokens (Haiku)
- Default TTL: 5 minutes (extended on each access)
- 1-hour TTL: available via API, higher write rate

**Claude Code specific:** Enable the 1-hour cache TTL via environment variable:
```bash
ENABLE_PROMPT_CACHING_1H=1 claude  # extends cache window to 1 hour
```

### 7.3 OpenAI Prompt Caching

OpenAI's caching is automatic — no explicit markup required. The platform automatically caches prompt prefixes that exceed a token threshold. The pricing model is different from Anthropic's: cached tokens cost 50% of the standard input rate (rather than Anthropic's 10%), but there is no cache write premium.

```python
# OpenAI — caching is automatic, no code changes needed
# Just ensure stable prefixes appear at the beginning of requests

from openai import OpenAI

client = OpenAI()

# Structure your messages so the stable part (system prompt, codebase context)
# comes FIRST and the variable part (current question) comes LAST.
# OpenAI caches the stable prefix automatically.

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": stable_system_prompt  # ← stable, will be cached
        },
        {
            "role": "user",
            "content": large_codebase_context  # ← stable, will be cached
        },
        {
            "role": "user",
            "content": "Fix the auth bug"      # ← variable, not cached
        }
    ]
)

# Check cache usage
print(response.usage.prompt_tokens_details.cached_tokens)
```

**OpenAI pricing (mid-2026):**
- Cached tokens: 50% of standard input rate
- No cache write premium
- Automatic, no explicit breakpoints needed
- Minimum threshold for caching: 1,024 tokens of stable prefix

### 7.4 What Silently Breaks the Cache

Cache misses are silent — you pay full price and receive no warning that caching failed. These are the most common causes:

```
CACHE BREAKERS — what silently kills your cache hits

1. Changing the system prompt between requests
   Even adding a timestamp, a session ID, or changing one word
   invalidates the entire cache for that prefix.

2. Reordering tool definitions
   Tool definitions are part of the cached prefix.
   If the order changes between calls (e.g. dynamically added tools),
   the cache misses.

3. Compacting the conversation history
   Compaction rewrites history into a summary.
   The summary is not byte-identical to the original.
   This forces a cache miss and new write.
   → Compact infrequently and at predictable boundaries.

4. Injecting dynamic content into stable context
   "Current time: 14:23:07" in the system prompt = cache miss every call.
   Move dynamic content to the user turn, not the system prompt.

5. Changing file content between requests
   If you include a codebase snapshot in your context, any file change
   forces a new cache write for everything after the changed file.
   → Place the most stable content (system prompt, tool definitions)
     FIRST, and the most volatile content (codebase snapshots) LAST.
```

---

## 8. Layer 5 — Model Tiering

Model selection is the single highest-leverage cost lever available to a team, but it requires discipline to use correctly. The instinct is to always use the most powerful available model. In practice, using Opus for writing unit tests costs 5× more than using Haiku for the same task, with no quality difference in the output.

The professional discipline is to match model capability to task complexity. The following table reflects real-world usage patterns from AI engineering teams:

```
MODEL TIERING FOR BACKEND & GENAI ENGINEERING

Tier   │ Claude         │ OpenAI         │ Use For
───────┼────────────────┼────────────────┼──────────────────────────────────────
High   │ Claude Opus    │ GPT-4o         │ Architecture decisions
       │ (Opus 4.6/4.8) │ o3             │ Complex cross-service debugging
       │                │                │ Novel GenAI pipeline design
       │                │                │ Multi-file refactoring with deep
       │                │                │ semantic dependencies
       │                │                │ Security vulnerability analysis
───────┼────────────────┼────────────────┼──────────────────────────────────────
Mid    │ Claude Sonnet  │ GPT-4.1        │ New API endpoints
       │ (Sonnet 4.5/   │                │ Service logic implementation
       │  4.6)          │                │ LLM pipeline code
       │                │                │ RAG component development
       │                │                │ Integration debugging
       │                │                │ Code review responses
───────┼────────────────┼────────────────┼──────────────────────────────────────
Low    │ Claude Haiku   │ GPT-4o-mini    │ Unit test writing
       │ (Haiku 4.5)    │ o4-mini        │ Docstrings, comments, type hints
       │                │                │ Renaming, formatting, simple edits
       │                │                │ Subagent research tasks
       │                │                │ Eval harness runs
       │                │                │ Log analysis
───────┼────────────────┼────────────────┼──────────────────────────────────────
Local  │ Ollama/        │ Local via      │ Exploration, throwaway experiments
       │ LM Studio      │ LiteLLM        │ Sensitive codebases (no cloud)
       │ (via Aider     │                │ High-volume low-stakes tasks
       │  --model)      │                │ Offline work
```

**Extended thinking / reasoning models** deserve special mention. Claude's extended thinking mode and OpenAI's o3 are billed at output token rates for their reasoning tokens, which are significantly more expensive than standard input tokens. A single complex planning request with extended thinking can consume 20,000–50,000 reasoning tokens. The default behaviour in Claude Code is to enable extended thinking, which is appropriate for complex planning but wasteful for routine tasks.

Disable extended thinking for all tasks that are not architectural or deeply analytical:
```bash
# Claude Code — set in /config or via environment
MAX_THINKING_TOKENS=3000   # routine tasks (vs default of up to 50,000)
/effort low                # for simple edits within a session
```

For Aider's architect mode, the thinking-heavy model is only used for the planning turn. The edit-application turn uses a cheaper executor model. This is an efficient use of reasoning capacity.

---

## 9. Layer 6 — Hooks and Enforcement

The layers described above — context firewall, code intelligence, session discipline, caching, model tiering — all depend on consistent behaviour from the agent. Agents are stochastic systems: they read instructions but do not always follow them. This is where hooks become critical. Hooks are executable enforcement mechanisms that fire at specific points in the agent's lifecycle and can block, modify, or redirect agent behaviour before it takes an expensive action.

### 9.1 Claude Code Hooks System

Claude Code supports a declarative hooks system configured in `.claude/settings.json`. Hooks fire at named lifecycle events and can approve, deny, or modify the action in question. The most important hook event for token efficiency is `PreToolUse`, which fires before any tool call and can block it.

```
CLAUDE CODE HOOK LIFECYCLE

User sends message
        │
        ▼
   UserPromptSubmit hook (can modify prompt before Claude sees it)
        │
        ▼
   Claude plans next action
        │
        ▼
   PreToolUse hook ──→ APPROVE: tool runs normally
                   ──→ DENY: tool blocked, Claude gets explanation
                   ──→ ASK: Claude is prompted to reconsider
        │ (approved)
        ▼
   Tool executes (Read file / Bash / Grep / Write / etc.)
        │
        ▼
   PostToolUse hook (can validate output, trigger checks)
        │
        ▼
   Claude processes result, plans next action
        │
        ▼
   Stop / SubagentStop hooks (cleanup, summary, HANDOFF.md write)
```

Hook types available:
- **Command hooks:** run a shell script or command
- **Prompt hooks:** send a single-turn prompt to an LLM for evaluation
- **Agent hooks:** spawn a subagent with tool access for deeper verification

Hooks are configured per-project in `.claude/settings.json` and per-developer in `~/.claude/settings.json`. Project-level hooks apply to everyone working on the codebase; user-level hooks apply globally to the individual developer.

### 9.2 Practical Hook Patterns

**Pattern 1: Block broad grep before graph query**

This hook intercepts `grep -r`, `rg`, and `find` commands and blocks them if a graphify index exists, redirecting the agent toward graph queries.

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [{
          "type": "command",
          "command": "bash -c 'cmd=$(echo \"$CLAUDE_TOOL_INPUT\" | python3 -c \"import sys,json; d=json.load(sys.stdin); print(d.get(\\\"command\\\",\\\"\\\"))\" 2>/dev/null); case \"$cmd\" in *\"grep -r\"*|*\"grep -rn\"*|*\"rg \"*|*\"find . -name\"*) if [ -f graphify-out/graph.json ]; then echo \"STOP: graphify-out/graph.json exists. Run: graphify query \\\"<what you need>\\\" instead of grep. Use grep only if graphify returns nothing.\"; exit 2; fi ;; *) exit 0 ;; esac'"
        }]
      }
    ]
  }
}
```

**Pattern 2: Enforce HANDOFF.md before compaction**

This hook fires before any compaction event and blocks it if HANDOFF.md has not been written or is out of date.

```json
{
  "hooks": {
    "PreCompact": [
      {
        "matcher": ".*",
        "hooks": [{
          "type": "command",
          "command": "bash -c 'if [ ! -f HANDOFF.md ]; then echo \"BLOCKED: Write HANDOFF.md before compacting. See SESSION END PROTOCOL.\"; exit 2; fi'"
        }]
      }
    ]
  }
}
```

**Pattern 3: Token budget warning**

This hook fires when the auto-loaded context (AGENTS.md + scoped rules) exceeds a token threshold, alerting the developer that the instruction file is getting too large.

```bash
# Install claude-token-optimizer (community tool: nadimtuhin/claude-token-optimizer)
npm install -g claude-token-optimizer

cto hooks install pre-tool-token-guard
cto hooks settings   # prints JSON block to merge into settings.json

# Default thresholds:
# CTO_WARN_TOKENS=2000  — warn when auto-loaded context > 2,000 tokens
# CTO_BLOCK_TOKENS=8000 — block when auto-loaded context > 8,000 tokens
```

**Pattern 4: Session-end HANDOFF.md auto-write**

This hook fires when the session stops and automatically triggers a HANDOFF.md write if files were modified.

```json
{
  "hooks": {
    "Stop": [
      {
        "matcher": ".*",
        "hooks": [{
          "type": "prompt",
          "prompt": "If any files were modified during this session, write HANDOFF.md now following the template in AGENTS.md. Include: what was done, current state (working/broken), files modified, and the next exact action."
        }]
      }
    ]
  }
}
```

---

## 10. Platform-Specific Playbooks

### 10.1 Claude Code

Claude Code is Anthropic's terminal-native coding agent. It has the richest hook system and the deepest integration with the techniques described in this document.

**Setup checklist for a new project:**
```bash
# 1. Initialise project files
cat > .claudeignore << 'EOF'
node_modules/ dist/ build/ __pycache__/ .venv/
*.lock coverage/ *.min.* *.gguf *.bin *.safetensors
data/raw/ data/processed/ chroma_db/ .env .env.*
EOF

# 2. Install and initialise graphify
pip install graphifyy && graphify . && graphify claude install

# 3. Create scoped rules directory
mkdir -p .claude/rules

# 4. Set up hooks
# Create .claude/settings.json with the hook patterns from Section 9.2

# 5. Enable 1-hour prompt caching
echo 'export ENABLE_PROMPT_CACHING_1H=1' >> ~/.zshrc   # or .bashrc

# 6. Set conservative thinking budget for routine work
echo 'export MAX_THINKING_TOKENS=4000' >> ~/.zshrc
```

**Within-session commands:**
```
/model sonnet          — switch to Sonnet for routine tasks
/model opus            — switch to Opus for complex decisions
/model haiku           — switch to Haiku for tests/docs
/effort low            — reduce reasoning effort (saves tokens)
/compact               — manual compaction (do at 60–65% context)
/clear                 — wipe context between unrelated tasks
/btw <question>        — ask a question without adding to context history
```

**Recommended session rhythm:**
```
START: Read HANDOFF.md → state scope in one sentence → begin
DURING: /compact after each resolved sub-task
        /clear when switching to unrelated task
        delegate exploration to subagents
END: Write HANDOFF.md before /compact or closing session
```

### 10.2 OpenAI Codex CLI

Codex CLI operates in a similar pattern to Claude Code but with OpenAI's model stack. The AGENTS.md file is natively read. Context discipline follows the same principles.

```bash
# Install
npm install -g @openai/codex

# Set reasoning effort (critical for cost control)
codex --model o4-mini --reasoning-effort low   # routine tasks
codex --model gpt-4o                            # standard tasks
codex --model o3 --reasoning-effort high        # architecture only

# Context management
codex --no-context-file    # skip auto-loading project files (when irrelevant)
codex --files src/services/auth.py  # explicit file inclusion (like Aider's approach)
```

Codex reads AGENTS.md natively and supports graphify via `graphify codex install`. Cross-session memory follows the same HANDOFF.md pattern. The `--reasoning-effort` flag is particularly important: `o3` with `--reasoning-effort high` is extremely expensive and should be reserved for architectural decisions only.

### 10.3 GitHub Copilot

GitHub Copilot has two distinct modes with very different cost profiles: inline completions (cheap, local context) and Chat (expensive, repo-wide indexing). Understanding this distinction is the most important Copilot-specific efficiency insight.

**Inline completions** use a small, local context window focused on the current file and cursor position. They are fast, cheap, and appropriate for day-to-day coding. **Copilot Chat** performs broad repository indexing when answering questions, which can consume significantly more context. Reserve Chat for complex debugging and architectural questions; use inline suggestions for routine coding.

**Key configuration files:**
```
.github/
├── copilot-instructions.md     ← global instructions (keep under 20 lines)
└── instructions/
    ├── api.instructions.md     ← loads for **/*.py in src/api/ (applyTo frontmatter)
    ├── tests.instructions.md   ← loads for tests/**/*.py
    └── pipelines.instructions.md  ← loads for src/pipelines/**
```

Example scoped instruction file:
```markdown
---
applyTo: "src/pipelines/**"
---
All LLM calls go through src/llm/client.py — never call the SDK directly.
Prompt templates live in src/prompts/ as .j2 files — never inline as Python strings.
Always add a token count guard before calls: check count_tokens(prompt) < MAX_TOKENS.
```

**Enterprise configuration:** In GitHub's organisation settings, configure Content Exclusions to prevent Copilot from indexing generated files, lock files, binary model files, and data directories. This reduces the context Copilot builds during indexing. Organisation admins can also set usage limits per repository and team, and restrict premium model (GPT-4o, o3) access to specific teams that genuinely need it.

```
GitHub Org Settings → Copilot → Policies:
  - Content exclusions: target/, *.generated.*, *.lock, data/
  - Premium model access: restrict to Architecture and Platform teams
  - Usage limits: set per-team monthly caps
  - Audit logging: enable for compliance and cost visibility
```

### 10.4 Aider

Aider is the most token-efficient of the major coding agents for routine engineering work, primarily because of the repo map's PageRank-based context selection. It is particularly well-suited for large-scale refactoring, multi-file changes, and any workflow where predictable, explicit context control matters.

```bash
# Install
pip install aider-chat

# ALWAYS name specific files — never add directories
aider src/services/auth_service.py tests/test_auth_service.py

# Tight repo map budget
aider --map-tokens 1024 src/services/auth_service.py

# Architect mode — strong planner, cheap executor (recommended pattern)
aider --architect \
      --model anthropic/claude-sonnet-4-6 \
      --editor-model deepseek/deepseek-coder \
      src/pipelines/rag_pipeline.py

# Multi-model strategy for maximum cost efficiency:
# DeepSeek Coder for exploration (~10× cheaper than Sonnet)
aider --model deepseek/deepseek-coder src/api/routes.py

# Switch to Sonnet only for the change that will be committed
aider --model anthropic/claude-sonnet-4-6 src/api/routes.py

# Useful session commands:
/add src/services/new_service.py   # add a file to context
/drop src/services/old_service.py  # remove a file (reduces context)
/clear                             # wipe context, start fresh
/tokens                            # show current context token usage
/map                               # show repo map for current context
```

Aider's git integration means every AI-generated change is auto-committed with a generated commit message. This makes it easy to review, revert, or cherry-pick AI-generated changes. For backend engineers practicing careful code review, this is a significant workflow advantage.

### 10.5 Cursor

Cursor is a VS Code fork with native AI integration. It uses a local vector index for context retrieval rather than file reading, which makes it behave differently from terminal agents. Its RAG-like retrieval automatically surfaces relevant files without explicit file selection, but this automation comes with less control over what gets indexed and included.

**Key efficiency practices for Cursor:**
- Open only the service or module you are working on. Cursor indexes open files and their neighbours.
- Use Composer mode (Ctrl+I) for multi-file changes. It is more token-efficient than Chat for coding tasks because it keeps context focused on the specific files to change.
- Use `@file` syntax to explicitly scope Chat questions: `@src/services/auth.py What does the token refresh logic do?` rather than broad questions that trigger full-repo indexing.
- Configure content exclusions in `.vscode/settings.json`:

```json
{
  "github.copilot.enable": {
    "*.xml": false,
    "*.lock": false,
    "*.generated.*": false
  },
  "cursor.indexing.exclude": [
    "node_modules",
    "dist",
    "data/raw",
    "*.gguf",
    "*.bin",
    "evals/datasets"
  ]
}
```

Cursor rules (`.cursor/rules/*.mdc`) support glob-pattern activation:
```markdown
---
description: "Rules for LLM pipeline development"
globs: "src/pipelines/**"
alwaysApply: false
---
All LLM calls through src/llm/client.py.
Prompt templates in src/prompts/ as .j2 files.
```

---

## 11. Measuring What You Are Spending

You cannot optimise what you cannot see. Before applying any of the techniques in this document, instrument your current usage.

**For Claude Code**, the community tool `claude-token-lens` provides real-time token attribution:
```bash
npm install -g claude-token-lens
claude-token-lens            # live dashboard showing which tool/file is spending tokens
```

The `egorfedorov/claude-context-optimizer` plugin (available via `npx skills add`) tracks per-file usefulness scores, session waste, and provides heatmaps of which files are loaded vs. actually referenced.

**For API-based workflows**, instrument your code to log cache hit rates and per-call token usage:
```python
# Log cache efficiency on every call
response = client.messages.create(...)
usage = response.usage

print(f"Cache read: {usage.cache_read_input_tokens} tokens")
print(f"Cache write: {usage.cache_creation_input_tokens} tokens")
print(f"Input (uncached): {usage.input_tokens} tokens")
print(f"Output: {usage.output_tokens} tokens")

cache_hit_rate = usage.cache_read_input_tokens / (
    usage.cache_read_input_tokens + usage.input_tokens + 1
)
print(f"Cache hit rate: {cache_hit_rate:.1%}")
# Target: > 80% cache hit rate on long sessions
```

**Baseline metrics to track before and after implementing this guide:**

| Metric | How to measure | Target |
|--------|---------------|--------|
| Tokens per session | Platform dashboard or `claude-token-lens` | Reduce 40–70% within 2 weeks |
| Cache hit rate | `response.usage.cache_read_input_tokens` | > 80% on API workflows |
| Sessions requiring re-orientation | Count sessions with no HANDOFF.md | < 10% after first month |
| Orientation tokens (graph vs file read) | Graphify query count vs file read count | > 70% queries via graph |
| Context fill at compaction | % full when /compact triggered | < 70% (not 95%) |

---

## 12. Anti-Patterns Reference

This section documents the most common wasteful behaviours observed across teams, with the correct alternatives.

| Anti-Pattern | Why It Is Wasteful | Correct Approach |
|---|---|---|
| `cat src/services/llm_service.py` to "understand it" | Loads entire file (~500 lines) when 1 function is needed | `graphify query "llm_service"` or `sed -n '45,90p'` |
| `grep -r "embedding"` as first step | Loads every match into context as a tool result | `graphify query "embedding usage"` first |
| Opening the entire `src/` directory | Volumes of irrelevant code in context immediately | Read only the specific file being modified |
| `pytest tests/` after a 2-line bug fix | Full test output (thousands of tokens) added to context | `pytest tests/path/test_file.py::test_name -x` |
| Rewriting a module because one function is wrong | Destroys unrelated working code, creates review burden | Fix the specific broken function only |
| Inlining prompts as Python strings in services | Untestable, unversioned, bloats the service file in context | Move to `src/prompts/filename.j2` |
| Using Opus to write unit tests | 5× the cost for zero quality difference | Use Haiku for test generation |
| Loading `.env` to see what API keys exist | Security risk and token waste | Read env var names from code, not values |
| Carrying auth bug context into embedding refactor | Cross-contamination leads to bad decisions | `/clear` between unrelated tasks always |
| Running eval harnesses in the main session | Fixture data and verbose output explodes context | Delegate to subagent; receive only summary |
| Re-reading a file already in session context | Pure waste, identical information already present | Reference by filename; already in context |
| Waiting for 95% context fill before compacting | Context is already bloated; compaction loses control | Compact proactively at 60–65% |
| Using `--map-tokens 8192` in Aider | Oversized repo map competes with conversation context | Default 1,024 or 2,048 for large repos |
| Dynamic content in system prompt | Breaks prompt cache every call | Move timestamps, IDs to user turn |
| Global rules for domain-specific patterns | All sessions pay for rules they never need | Use path-scoped rules |

---

## 13. Organizational Governance Framework

The techniques in this document are most effective when applied consistently across a team rather than sporadically by individuals. This section provides a practical governance framework for backend engineering organisations adopting AI-assisted development at scale.

### Why Governance Is Necessary

Anthropic's enterprise usage data shows an average cost of $13 per developer per active day and $150–250 per developer per month, with 10% of developers consuming costs above $30 per active day. Without governance, teams find that a small number of high-usage developers — often those doing exploratory or architectural work — account for a disproportionate fraction of total cost, and that cost is invisible until the invoice arrives.

Governance does three things: it makes cost visible before it is a problem, it establishes consistent baselines that enable meaningful comparison, and it distributes institutional knowledge about efficient practices so that new team members start efficiently rather than learning through expensive trial and error.

### The Four-Layer Governance Model

```
ORGANIZATIONAL GOVERNANCE ARCHITECTURE

┌─────────────────────────────────────────────────────────────┐
│  Layer 1: STANDARDS (what goes in every repository)         │
│  ─ AGENTS.md template, .claudeignore baseline               │
│  ─ Hook configuration standard                              │
│  ─ Scoped rules structure requirement                       │
│  ─ HANDOFF.md template                                      │
│  Owned by: Platform / DX team                              │
│  Enforced by: Project initialisation script                 │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: MEASUREMENT (what gets tracked)                   │
│  ─ Per-developer token consumption dashboard                │
│  ─ Per-repository session cost tracking                     │
│  ─ Cache hit rate monitoring                                │
│  ─ Weekly cost anomaly alerts                               │
│  Owned by: Engineering managers + Finance                  │
│  Enforced by: CI token reporting + billing alerts           │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: TRAINING (what developers know)                   │
│  ─ Onboarding module: AI cost fundamentals                  │
│  ─ Tool-specific playbook per platform used                 │
│  ─ Anti-pattern library with worked examples                │
│  ─ Quarterly review of new efficiency techniques            │
│  Owned by: Senior engineers / Tech leads                   │
│  Enforced by: Onboarding checklist                          │
├─────────────────────────────────────────────────────────────┤
│  Layer 4: CONTROLS (what is technically prevented)          │
│  ─ Budget caps per developer (Copilot: Org Settings)        │
│  ─ Premium model restrictions (Opus/o3 require approval)    │
│  ─ API key rate limits for team builds                      │
│  ─ Content exclusions via organisational policy             │
│  Owned by: Platform / Security teams                       │
│  Enforced by: Platform configuration + PR checks            │
└─────────────────────────────────────────────────────────────┘
```

### Project Initialisation Standard

Create a CLI script that every developer runs when starting a new project. This script creates the required files, installs graphify, and configures hooks — eliminating the "I'll set it up later" deferral that causes early sessions to be expensive.

```bash
#!/usr/bin/env bash
# ai-project-init.sh — run once per new repository
# Store at: tools/scripts/ai-project-init.sh (commit to org template repo)

set -e

echo "Initialising AI-assisted development setup..."

# 1. Create token firewall
cat > .claudeignore << 'EOF'
node_modules/ dist/ build/ __pycache__/ .venv/ venv/
*.lock package-lock.json poetry.lock
coverage/ htmlcov/ .nyc_output/
*.min.js *.min.css *.generated.*
*.gguf *.bin *.safetensors *.pkl *.pt *.onnx
data/raw/ data/processed/
chroma_db/ faiss_index/ .chroma/ qdrant_storage/
.env .env.* secrets.*
evals/fixtures/ evals/datasets/
EOF
cp .claudeignore .aiderignore
echo "✓ .claudeignore created"

# 2. Create scoped rules structure
mkdir -p .claude/rules/{api,services,pipelines,agents,evals}
echo "✓ .claude/rules/ structure created"

# 3. Install and build knowledge graph
if command -v pip &>/dev/null; then
    pip install graphifyy -q
    graphify .
    graphify claude install
    graphify codex install
    echo "✓ Graphify installed and indexed"
fi

# 4. Create hook configuration
mkdir -p .claude
cat > .claude/settings.json << 'EOF'
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [{
          "type": "command",
          "command": "bash -c 'cmd=$(echo \"$CLAUDE_TOOL_INPUT\" | python3 -c \"import sys,json; d=json.load(sys.stdin); print(d.get(\\\"command\\\",\\\"\\\"))\" 2>/dev/null); case \"$cmd\" in *\"grep -r\"*|*\"rg \"*|*\"find . -name\"*) [ -f graphify-out/graph.json ] && echo \"STOP: Use graphify query instead of grep.\" && exit 2 || exit 0 ;; *) exit 0 ;; esac'"
        }]
      }
    ],
    "Stop": [
      {
        "matcher": ".*",
        "hooks": [{
          "type": "prompt",
          "prompt": "If any files were modified, write HANDOFF.md with: what was done, current state, files modified, and next action."
        }]
      }
    ]
  }
}
EOF
echo "✓ Claude Code hooks configured"

# 5. Copy org AGENTS.md template
TEMPLATE_URL="https://raw.githubusercontent.com/YOUR-ORG/ai-standards/main/AGENTS.md.template"
if command -v curl &>/dev/null; then
    curl -s "$TEMPLATE_URL" -o AGENTS.md 2>/dev/null || \
        echo "NOTE: Copy AGENTS.md template manually from ai-standards repo"
fi

echo ""
echo "Setup complete. Next steps:"
echo "1. Fill in PROJECT IDENTITY section of AGENTS.md (5 minutes)"
echo "2. Set ENABLE_PROMPT_CACHING_1H=1 in your shell profile"
echo "3. Set MAX_THINKING_TOKENS=4000 in your shell profile for routine work"
echo "4. Review .claudeignore and add any project-specific paths"
```

### Tiered Model Access Policy

```
MODEL ACCESS TIERS — RECOMMENDED POLICY

Tier A (All developers, no approval needed):
  Claude Haiku, GPT-4o-mini, DeepSeek Coder, local models
  → Test writing, docstrings, renaming, formatting

Tier B (All developers, standard quota):
  Claude Sonnet, GPT-4.1, GPT-4o in Auto mode
  → Daily coding, implementation, debugging

Tier C (Senior developers + Tech leads, monitored):
  Claude Opus, o3, GPT-4o manually selected
  → Architecture decisions, complex refactoring
  → Must document justification in commit message

Tier D (Per-request approval):
  Extended thinking / high reasoning effort at large context
  → Reserved for critical path architecture only
  → Request via tech lead; budget reviewed monthly
```

In GitHub Copilot Enterprise: Settings → Copilot → Policies → Model Access restricts Tier C/D models to specific teams. In Claude Code on Teams/Enterprise plans, spend limits per developer are configurable in the admin dashboard.

### HANDOFF.md as an Organisational Asset

Beyond individual productivity, HANDOFF.md files become an organisational asset when they accumulate over time. They create a searchable, git-committed record of every significant AI-assisted change: what decision was made, which files were involved, what was working and broken at each checkpoint.

Enforce HANDOFF.md discipline via CI: a simple GitHub Actions job that checks for HANDOFF.md recency on PRs can flag stale files and prompt developers to update them before merge.

```yaml
# .github/workflows/handoff-check.yml
name: HANDOFF.md Freshness Check
on: [pull_request]
jobs:
  check-handoff:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Check HANDOFF.md
        run: |
          if [ ! -f HANDOFF.md ]; then
            echo "::warning::No HANDOFF.md found. Consider adding one for AI session continuity."
            exit 0
          fi
          DAYS_OLD=$(( ($(date +%s) - $(git log -1 --format=%ct -- HANDOFF.md)) / 86400 ))
          if [ $DAYS_OLD -gt 7 ]; then
            echo "::warning::HANDOFF.md is $DAYS_OLD days old. Update if AI work is ongoing."
          else
            echo "HANDOFF.md is current ($DAYS_OLD days old)."
          fi
```

### Quarterly Review Cadence

Token efficiency is not a one-time setup. The tool ecosystem is evolving rapidly — graphify, codegraph, and session management patterns are all under active development. A quarterly review cadence ensures teams stay current.

Recommended quarterly review agenda:
1. Review per-developer and per-repository cost trends. Identify outliers.
2. Audit AGENTS.md files across repositories for bloat. Trim files that have grown beyond 60 lines of global rules.
3. Check graphify index freshness. Re-run `graphify . --update` on repos that have seen major structural changes.
4. Review new platform capabilities. Claude Code, Codex, and Copilot ship updates frequently; features that were absent when the team last reviewed may now be available.
5. Update the anti-pattern library with any new wasteful patterns observed.
6. Confirm that all new team members have completed AI efficiency onboarding.

### Summary: The Governance Checklist

```
EVERY REPOSITORY:
  [ ] .claudeignore/.aiderignore with standard baseline
  [ ] AGENTS.md under 60 lines of global rules
  [ ] .claude/rules/ with path-scoped domain rules
  [ ] .claude/settings.json with PreToolUse + Stop hooks
  [ ] graphify-out/ committed and current
  [ ] HANDOFF.md present and updated within 7 days (if AI work is active)

EVERY DEVELOPER:
  [ ] ENABLE_PROMPT_CACHING_1H=1 in shell profile
  [ ] MAX_THINKING_TOKENS=4000 in shell profile (routine work)
  [ ] Completed AI efficiency onboarding module
  [ ] Personal cost dashboard enabled

EVERY TEAM:
  [ ] Model access tier policy documented and enforced
  [ ] Monthly cost review scheduled
  [ ] Anti-pattern library maintained and shared
  [ ] Quarterly tool ecosystem review on calendar

EVERY QUARTER:
  [ ] AGENTS.md audit across all repositories
  [ ] Knowledge graph freshness check
  [ ] Platform capability review
  [ ] Onboarding material updated
```

---

*This document reflects the state of tools and practices as of 2025–2026. The ecosystem is evolving rapidly. Key repositories to monitor for updates: `safishamsi/graphify`, `Aider-AI/aider`, `rohitg00/awesome-claude-code-toolkit`, `affaan-m/everything-claude-code`, `olivomarco/github-copilot-token-optimization`. Anthropic's official cost documentation lives at `code.claude.com/docs/en/costs`.*

