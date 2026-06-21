# Observability — How We See What the Agents Are Doing

This folder is the one place that answers a simple question:

> **"My 8 AI agents just ran. What actually happened inside them — what did each
> one say to the LLM, what did the LLM say back, how many tokens did it burn, and
> what did it cost me?"**

This README teaches you how that works, end to end, assuming you've *never* set up
observability before. No prior knowledge. No vendor jargon.

---

## 1. The problem, in plain words

When you run the resume tailor, this happens:

```
orchestrator → 8 agents → each agent talks to an LLM (GPT-4o) one or more times
```

Normally, all of that is a **black box**. The agents run, you get a resume out the
other end, and you have no idea:

- What prompt did the "Experience Optimizer" actually send to GPT-4o?
- What did GPT-4o reply?
- Did it call the LLM once, or loop 5 times trying to improve a bullet point?
- How many tokens did that cost? In dollars?

**Observability = opening the black box.** We attach a "recorder" to the agents so
that every LLM call and every agent step gets sent to a website where you can click
through it visually, like a flight recorder for your AI.

The website we use is called **LangSmith** (https://smith.langchain.com). It's made
by the LangChain people, specifically for watching LLM apps.

---

## 2. The mental model: a "trace" is a tree

LangSmith records everything as a **trace**. A trace is just a tree of nested boxes.
We call each box a **box** in this doc. (LangSmith's own UI calls the same thing
a "run" or a "span" — same idea, just their words.) Here is what one resume trace
looks like in the dashboard:

```
▼ run_experience_optimizer                  ← a "chain" run (the whole agent)
  │   input:  {resume, job_desc, strategy}
  │   output: {optimized_bullets}
  │   tokens: 4,210   cost: $0.021   time: 8.3s
  │
  ├─▼ LLM call #1  (gpt-4o)                  ← an "llm" run (one model call)
  │     prompt:     "Rewrite these bullets for impact..."
  │     completion: "• Led migration of 12 services..."
  │     tokens: 1,800   cost: $0.009
  │
  ├─▼ audit_summary  (tool)                  ← a "tool" run (a helper function)
  │     input:  "• Led migration..."
  │     output: {score: 78, issues: ["add metrics"]}
  │
  └─▼ LLM call #2  (gpt-4o)                  ← it looped and tried again
        prompt:     "Improve based on critique..."
        completion: "• Led migration of 12 services, cutting latency 40%..."
        tokens: 2,410   cost: $0.012
```

**Read that tree top-to-bottom and you understand exactly what the agent did.**
That tree is the entire point of this folder. Everything below is about how we get
your agents to produce it.

Three kinds of run you'll see:
- **chain** — a high-level step (a whole agent). The parent box.
- **tool** — a helper function the agent called (e.g. `audit_summary`).
- **llm** — one single call to GPT-4o, with prompt + reply + tokens + cost.

---

## 3. How we capture it: TWO layers working together

Here's the key idea. We capture the tree using **two separate mechanisms** that
stack on top of each other. You need both, and they do different jobs.

```
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 2 — the readable labels  (we add this)                        │
│  "@trace_agent" decorator → makes the big named boxes:               │
│  "run_experience_optimizer", "run_job_analyzer", etc.                │
│  WITHOUT this, the dashboard is a flat list of raw LLM calls with    │
│  no idea which agent they belong to.                                 │
└─────────────────────────────────────────────────────────────────────┘
                                ▲  wraps around
                                │
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 1 — the automatic LLM recorder  (one line of setup)           │
│  Every call to GPT-4o is captured automatically: the prompt, the     │
│  reply, the token count, the dollar cost. This is where tokens &     │
│  cost actually come from. You write ZERO code per agent for this.    │
└─────────────────────────────────────────────────────────────────────┘
```

### Layer 1 — the automatic LLM recorder (where tokens & cost come from)

Your agents are built with **CrewAI**. CrewAI doesn't talk to OpenAI directly — it
talks through a library called **LiteLLM**. So *every* LLM call in the entire app,
from all 8 agents, funnels through this one chokepoint:

```
agent → CrewAI → LiteLLM → OpenAI (GPT-4o)
                    ▲
                    └── we tap the wire HERE
```

LiteLLM has a built-in feature: you can hand it a list of "callbacks" — recorders
it notifies after every call. One of the recorders it ships with is literally named
`"langsmith"`. So our entire Layer 1 is **one line**:

```python
litellm.callbacks = ["langsmith"]
```

After that line runs, *every* GPT-4o call from *any* agent is automatically shipped
to LangSmith with its prompt, reply, **token counts, and cost** — and we never had
to touch a single agent's code. That's why you'll see tokens and cost "for free."

> This line lives in `init_observability()` inside `langsmith_backend.py`. You don't
> call it yourself; it runs once at startup.

### Layer 2 — the readable labels (so you know which agent is which)

Layer 1 alone gives you a flat pile of LLM calls. You couldn't tell which call
belonged to the Summary Writer vs the ATS Optimizer. Layer 2 fixes that by drawing
the **named parent box** around each agent.

We do it with a Python **decorator** — a tag you put above a function:

```python
from src.observability import trace_agent

@trace_agent                       # ← this is Layer 2
def run_experience_optimizer(resume, job_desc, strategy):
    ...                            # inside here, the agent calls the LLM (Layer 1)
    return optimized_bullets
```

That `@trace_agent` tag tells LangSmith: *"start a named box called
`run_experience_optimizer`, and anything that happens inside — including the Layer 1
LLM calls — goes inside this box."* That's how the nesting in the tree from Section 2
gets created.

There are two decorators:
- `@trace_agent` → draws a **chain** box (use on whole-agent functions)
- `@trace_tool`  → draws a **tool** box (use on helper functions like `audit_summary`)

**You don't need to add these everywhere.** They're already on the 8 agent methods
in the orchestrator. You only add one if you write a *new* function you want to see.

---

## 4. The full flow, start to finish

Here is the complete path, from you hitting "run" to a tree showing up online:

```
                         YOU run the orchestrator
                                   │
                                   ▼
   ┌───────────────────────────────────────────────────────────────┐
   │ STARTUP (happens once)                                         │
   │   orchestrator calls init_observability("resume-tailor-agents")│
   │     1. reads config: is observability.enabled = true?          │
   │     2. reads the secret LANGSMITH_API_KEY from .env             │
   │     3. if both OK →  litellm.callbacks = ["langsmith"]   ◄ Layer 1 armed
   │                      sets LANGSMITH_TRACING=true                │
   │     4. if NOT OK →  logs why, returns False, app runs normally  │
   │                     (tracing just stays off — nothing breaks)   │
   └───────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
   ┌───────────────────────────────────────────────────────────────┐
   │ DURING THE RUN (happens for every agent)                       │
   │                                                                │
   │   @trace_agent on run_experience_optimizer()    ◄ Layer 2      │
   │        │  opens a named "chain" box in LangSmith               │
   │        ▼                                                       │
   │   agent asks CrewAI to do work                                 │
   │        │                                                       │
   │        ▼                                                       │
   │   CrewAI → LiteLLM → GPT-4o                                    │
   │        │      │                                                │
   │        │      └── LiteLLM's "langsmith" callback fires  ◄ Layer 1
   │        │          → ships prompt + reply + tokens + cost       │
   │        │            as an "llm" box INSIDE the chain box       │
   │        ▼                                                       │
   │   (optional) log_iteration_metrics("exp_optimizer", 2, {...})  │
   │        └── attaches your custom numbers (scores, deltas)       │
   │            onto the current box's metadata                     │
   └───────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
        LangSmith website shows the tree from Section 2.
        Open https://smith.langchain.com → project "resume-tailor-agents"
```

---

## 5. Where the capture is wired — the exact 3 places (real files & lines)

If you're new to this codebase and want to *see* where the recording actually
happens, it is in exactly **three** places. There is no hidden magic anywhere else.

### Place 1 — the global LLM recorder is armed once, at startup

File: `src/observability/langsmith_backend.py`, inside `init_observability()`.
This is the single line that makes every GPT-4o call get recorded (Layer 1):

```python
# src/observability/langsmith_backend.py  (inside init_observability)
litellm.callbacks = [*litellm.callbacks, "langsmith"]
```

And `init_observability()` is *called* from one spot — the orchestrator's startup:

```python
# src/agent_orchestrator.py : ~line 101
init_observability(project_name=project_name)
```

So: app starts → orchestrator calls `init_observability` → that line arms LiteLLM.
After that, you write nothing else for tokens/cost/prompts to be captured.

### Place 2 — each agent gets a named box via one decorator line

The named per-agent boxes (Layer 2) come from a `@trace_agent` line sitting on the
orchestrator's `_run_*` methods. There are 8 of them. Here is a real one:

```python
# src/agent_orchestrator.py : ~line 331
@trace_agent                                    # ← THIS line creates the box
def _run_resume_extraction(self, text: str) -> Resume:
    agent = create_resume_extractor_agent()
    return self._create_and_run_crew(agent=agent, ...)
```

The `@trace_agent` lines are at roughly lines **146, 331, 341, 351, 366, 388, 412,
436, 466** of `src/agent_orchestrator.py`. Each wraps one agent's run method. That's
the *entire* per-agent wiring — one line per agent.

### Place 3 — the on/off switch and the secret key (config)

```
src/config/settings.yaml      → observability.enabled (true/false), project name
.env                          → LANGSMITH_API_KEY=...   (the secret, gitignored)
```

### Where it is NOT wired (so you don't go hunting)

> **The 8 agent definition files in `src/agents/` were NOT modified for this.**
> Open `src/agents/professional_summary/agent.py` (or any other) and you will find
> **no tracing code** — just the normal CrewAI agent. That is on purpose. The agents
> stay clean; capture is bolted on at the LiteLLM layer (Place 1) and the
> orchestrator (Place 2). If you go looking for "the tracing change" inside an agent
> file, you'll find nothing, and that's correct.
>
> (A few helper functions in `src/agents/` use `@trace_tool` to show up as their own
> small boxes — that's optional extra detail, not where agent capture happens.)

**That's the whole thing.** Three places: one line arms LiteLLM, one decorator per
agent in the orchestrator, one switch + one key in config.

---

## 6. How to turn it ON

By default it is **OFF**, so nothing changes until you opt in. Two things to set:

**Step 1 — give it your API key (the secret).**
Get a key from https://smith.langchain.com (Settings → API Keys), then put it in
your `.env` file (this file is gitignored, so the secret never gets committed):

```
LANGSMITH_API_KEY=lsv2_your_real_key_here
```

**Step 2 — flip the switch.**
In `src/config/settings.yaml`, find the `observability:` block and set:

```yaml
observability:
  enabled: true                    # ← change false to true
  project: "resume-tailor-agents"  # the bucket your traces show up under
  endpoint: "https://api.smith.langchain.com"
```

That's it. Run the orchestrator normally and traces will appear in the dashboard.

> **Why two steps?** The key is a *secret* (goes in `.env`, never in git). The
> switch and project name are *settings* (go in the YAML, safe to commit). We keep
> secrets and settings in separate places on purpose.

### How to turn it OFF
Set `enabled: false` (or just remove the API key). The app runs exactly the same,
untraced. The tracing code is written to do **nothing and never crash** when it's
off — see Section 8.

---

## 7. What the public functions do (the API)

You import everything from `src.observability`. There are five things:

| You call this | What it does | When you'd use it |
|---|---|---|
| `init_observability(project)` | Turns the whole system on (arms Layer 1). | Once, at startup. Already wired in the orchestrator — you won't call it. |
| `@trace_agent` | Wraps a function as a named **chain** span. | Above a whole-agent function. |
| `@trace_tool` | Wraps a function as a named **tool** span. | Above a helper function you want to see. |
| `log_iteration_metrics(name, i, {...})` | Attaches your numbers (scores, token tallies, deltas) to the current span. | Inside critique/retry loops to chart quality over iterations. |
| `is_observability_enabled()` | Returns True/False — is tracing live right now? | To check status in a script. |

Example of the metrics one (this is what powers the per-iteration charts):

```python
from src.observability import log_iteration_metrics

log_iteration_metrics(
    agent_name="experience_optimizer",
    iteration=2,
    metrics={"quality_score": 78, "tokens_used": 2410, "improvement_delta": 13},
)
```

Even when LangSmith is off, this still prints the numbers to your local logs, so you
always have *some* record.

---

## 8. The most important design rule: it never breaks your app

Read this so you trust it. **Every function here is built to fail silently and keep
the pipeline running.** If the API key is missing, if the LangSmith library isn't
installed, if the network is down — the agents still run and you still get a resume.

```
init_observability():
    no key / disabled?  → log a warning, return False, app continues untraced
@trace_agent / @trace_tool:
    tracing off?        → return the function UNCHANGED (does literally nothing)
log_iteration_metrics():
    always logs locally; only ALSO sends to LangSmith if it's on, wrapped in
    try/except so a tracing error can never crash an agent
```

Observability is a *passive observer*. It watches; it never gets in the way.

---

## 9. The files in this folder

```
src/observability/
├── __init__.py          ← the front door. Every other file imports the 5
│                          functions from here, never from the backend modules.
│                          This file does nothing but re-export them in one place.
│
├── langsmith_backend.py ← startup: init_observability(), LiteLLM callback
│                          registration, and is_observability_enabled().
├── tracing.py           ← @trace_agent / @trace_tool decorators that wrap
│                          functions with langsmith.traceable.
├── iteration_metrics.py ← log_iteration_metrics(): structlog + LangSmith
│                          metadata attachment for custom domain metrics.
└── README.md            ← this file.
```

The config that controls it all lives outside this folder:
- `src/config/settings.yaml` → the `observability:` block (on/off, project name)
- `src/core/settings/schema.py` → `ObservabilityConfig` (the typed shape of that block)
- `src/core/settings/runtime.py` → reads `LANGSMITH_API_KEY` from `.env`
- `.env` → your actual secret key (gitignored)

---

## 10. Reading the dashboard

Open https://smith.langchain.com, pick the project **resume-tailor-agents**. You'll
see a list of traces (one per run). Click one to get the tree from Section 2.

What to look at:
- **The tree** — click any box to expand it. Click an `llm` box to see the exact
  prompt and reply. This is how you debug "why did the agent say that?"
- **Tokens & Cost** — shown on every box and totaled at the top. This is your spend.
- **Latency** — how long each step took. Find your slow agents here.
- **Metadata** — the numbers you sent via `log_iteration_metrics` show up here.

---

## 11. Troubleshooting

**"I ran it but nothing shows up in the dashboard."**
1. Is the switch on? Check `observability.enabled: true` in `settings.yaml`.
2. Is the key set? In a Python shell:
   `from src.core.settings import get_config; print(bool(get_config().langsmith_api_key))`
   → must print `True`.
3. Is it actually live? `from src.observability import is_observability_enabled; print(is_observability_enabled())`
   → must print `True` *after* the orchestrator started.
4. Check your logs for `langsmith_disabled` or `langsmith_api_key_missing` — those
   lines tell you exactly why it didn't start.

**"403 Forbidden in the logs."**
Your API key is wrong or expired. Get a fresh one from the LangSmith site.

**"I want richer names per agent."**
Add `@trace_agent` (with a clear function name) to the specific agent method you
care about. The function's name becomes the box's name in the tree.

---

**One-sentence summary:** LiteLLM auto-records every GPT-4o call (prompt, reply,
tokens, cost) to LangSmith, and our `@trace_agent` decorators group those calls into
named per-agent boxes so the dashboard shows you a readable tree of exactly what each
agent did and what it cost.
