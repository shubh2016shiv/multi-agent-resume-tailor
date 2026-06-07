# Lessons Learned — Writing Agents From Scratch

> **What this is:** Mistakes I made while creating this module, what each mistake taught me,
> and rules you can reuse when writing ANY agent from scratch.
> **What this is NOT:** A migration guide. The old `gap_analysis_agent.py` is a reference, not a spec.

---

## Rule 1: The old code is a GUIDE, not a BIBLE

**Mistake:** I copied the `_SYNONYMS` dict verbatim from the old file into `engines.py`.

```python
# What I wrote (wrong):
_SYNONYMS: dict[str, str] = {
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "k8s": "kubernetes",
    ...
}

def normalize_skill(skill: str) -> str:
    return _SYNONYMS.get(skill.lower().strip(), skill.lower().strip())
```

**Why it's wrong:** That dict isn't just incomplete — it's conceptually the wrong solution.
Acronyms are domain-specific ("LLM" = Large Language Model in tech, Master of Laws in legal).
No static map can be exhaustive. The LLM is already doing this reasoning during semantic
comparison — a hardcoded dict is a worse version of what the agent already does.

**Lesson:** Before copying anything from old code, ask:

```
Is this problem best solved by:
  (a) Static code (mechanical)  →  implement it
  (b) The LLM (judgment)        →  let the agent reason, don't pre-solve
  (c) A tool the agent calls    →  create an @tool
```

Skill normalization is (b). The LLM normalizes concepts as part of semantic matching.
A static synonym map is (a) pretending to solve (b) — always wrong.

---

## Rule 2: If a data source exists, consult it — but think, don't parrot

**Mistake:** I initially designed the gap analysis agent with ZERO tools — pure LLM reasoner.

**Correction:** `src/tools/tool_documentations/README.md` clearly maps `match_job_requirements`
to the Gap Analysis agent. The tool bundles two engines (requirement evidence matching +
keyword coverage) into one agent-readable report. One tool call gives the agent everything
it needs to reason over real data instead of guessing.

**Lesson:** The tool catalog tells you WHAT exists and WHO uses it. Use it. But also verify:
the catalog said Gap Analysis uses `match_job_requirements` — and that made sense once I
understood the tool's composite design (LLM judgment + mechanical scan merged into one report).

```
BEFORE building an agent:
  1. Read tool_documentations/README.md
  2. Find the tools mapped to your agent
  3. Verify: does each tool actually solve a sub-problem the agent faces?
  4. If a tool exists but doesn't fit → document why, don't wire it blindly
```

---

## Rule 3: Stay in scope — the module IS the deliverable

**Mistake:** After creating `src/agents/gap_analysis/`, I updated imports in
`src/agent_orchestrator.py`, `src/agents/__init__.py`, and `src/orchestration/nodes.py`.

**Why it's wrong:** Nobody asked me to wire the module into the system. The old file still
works. The orchestrator is being rewritten separately. By touching those files I created
merge conflicts for a migration nobody requested.

**Lesson:**

```
When creating a NEW module:
  ✅  Write the module files (__init__, agent, engines, architecture doc)
  ✅  Write a trigger script that runs the module in isolation
  ❌  Do NOT update imports in orchestrator or __init__.py
  ❌  Do NOT delete the old file
  ❌  Do NOT touch any file outside the module directory + trigger script
```

The module proves itself by running isolated. Wiring it in is a separate task.

---

## Rule 4: Engines are post-hoc validators — not agent runtime helpers

**Mistake:** I included `normalize_skill` in `engines.py` as if the agent would call it.

**Correction:** `engines.py` in an agent module contains functions that validate the agent's
OUTPUT after the fact. They are never called during agent execution. They answer "was the
output any good?"

```
engines.py — what belongs:
  ✅  check_strategy_quality(strategy)   →  score the agent's output
  ✅  calculate_coverage_stats(strategy) →  derive metrics from output

engines.py — what does NOT belong:
  ❌  normalize_skill(skill)             →  the LLM does this during reasoning
  ❌  Any function the agent calls at runtime
  ❌  Any function that pre-processes input (that's the formatter's job)
```

---

## Rule 5: One agent = one tool (unless you can prove otherwise)

**Mistake:** None here — but this was a decision I had to justify.

The gap analysis agent gets ONE tool: `match_job_requirements`. That tool is already a
composite — it bundles two engines via `_merge()`. Adding more tools would force the agent
to orchestrate tool calls instead of focusing on strategy.

```
When deciding tool count:
  1 tool  ←  the tool composites everything the agent needs (default)
  2-3     ←  the agent needs fundamentally different KINDS of information
  4+      ←  reconsider: are you turning the agent into an orchestrator?
```

---

## Rule 6: Config validation = fail fast, one place only

**Mistake (avoided):** The old code validated config in `_load_agent_config` (soft —
returned defaults with missing `llm`) and again in `create_gap_analysis_agent` (hard —
raised RuntimeError). Two validation points, one of which silently degraded.

**Lesson:**

```
config validation rules:
  1. Validate ONCE, in _load_agent_config
  2. Validate ALL required fields (role, goal, backstory, llm)
  3. Fail on missing fields — NEVER silently fall back to defaults
  4. The factory trusts the config loader's output
```

---

## Rule 7: The module is 4 files. No more, no less.

```
src/agents/<agent_name>/
├── __init__.py              ← re-exports create_*_agent() only
├── agent.py                 ← factory + config helper, ~40 lines
├── engines.py               ← post-hoc validators, pure functions
└── <agent>_architecture.md  ← ASCII-art: zones, data flow, decisions
```

```
What goes WHERE:
  agent.py          →  Agent creation. Config, LLM, tools, resilience params.
  engines.py        →  Functions that VALIDATE the agent's output. Pydantic in, dict out.
  __init__.py        →  One export: the factory function.
  architecture.md   →  Zones, data flow, downstream consumers, design decisions.
```

---

## Quick Self-Check — Before Declaring an Agent Module "Done"

```
□ Does agent.py import ALL framework classes at the top? (no inline imports)
□ Is _load_agent_config a separate function that fails on missing fields?
□ Are tools declared as a module-level list? (_<NAME>_TOOLS = [...])
□ Does the logger report all tool names, not just one?
□ Does engines.py contain ONLY post-hoc validators? (nothing the agent calls)
□ Does __init__.py export ONLY the factory function?
□ Does architecture.md show ASCII-art zones, data flow, and decisions?
□ Is there a trigger_<agent>.py script at project root that runs it in isolation?
□ Have I NOT touched orchestrator, __init__, or the old file?
```
