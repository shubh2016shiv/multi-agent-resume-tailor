# `llm_token_tracker`, explained from zero

You should be able to read this top to bottom, never feel lost, and end up
knowing exactly what this folder does, what it doesn't do, where code
actually starts running, and what the "algorithm" is. No prior knowledge of
this codebase assumed.

---

## Part 0 — The problem, before any code

When you call an LLM (GPT-4, Claude, whatever), the provider doesn't charge
you or limit you by *characters* or *words*. It charges and limits you by
**tokens** — small chunks of text, roughly ¾ of a word each.

```
"Hello world"                →  2 tokens (roughly)
a 10-page document as a      →  several thousand tokens
prompt
```

Why does that matter enough to build a whole folder around it? Two reasons:

1. **Cost.** More tokens in = more money spent. Do this at scale (many agents,
   many calls) and an oversized prompt is an expensive mistake.
2. **Limits.** Every model has a maximum number of tokens it can accept. Go
   over it and the provider just rejects your request.

So somewhere in the codebase, *before* a prompt gets sent to an LLM, it would
be useful to ask:

> "Roughly how many tokens is this, and is that too many?"

**That single question is this entire package's job.** Everything else in
this README is detail on top of that one sentence.

---

## Part 1 — One running example, used the whole way through

To keep this concrete, imagine one small piece of the app:

> A tool builds a prompt out of a system message plus a resume's text, and
> wants to ask an LLM to extract structured data from it.

We'll trace *this one prompt* through the package, file by file, so every
piece of code below has a real thing happening to it instead of being
abstract.

---

## Part 2 — What this package is, in one sentence each

- **What it is:** a small toolkit that (a) estimates how many tokens a piece
  of text is, and (b) can refuse to let a prompt through if it's too big.
- **What it is not:** it does not automatically watch your multi-agent
  pipeline, does not know what "stages" your app has, and does not measure
  what the LLM sends *back* to you.

That last point trips people up, so let's be very explicit:

| You might assume...                                   | Actually...                                                                 |
|---------------------------------------------------------|------------------------------------------------------------------------------|
| "It tracks tokens across my whole multi-agent run."     | No. It only ever looks at *one piece of text you hand it*, one call at a time. |
| "It knows about pipeline stages, or agent order."       | No such concept exists anywhere in this folder. That idea lives elsewhere in the app (orchestration code), not here. |
| "It counts the LLM's response too (output tokens)."     | No. It only measures **input** — the text going *to* the model. Output tokens have to be handed to it by you, from the provider's own response. |
| "It's the official record of what I spent."             | No. The real, provider-confirmed numbers live in a different part of the app (observability). This package is a local *estimate*, used as a gate, not a ledger. |

Keep that distinction in mind — **estimator/gate here, real meter elsewhere**
— and the rest of the package will make sense.

---

## Part 3 — The six files, read in the order that actually builds understanding

Don't read these alphabetically. Read them in this order — each one is easy
to understand once you know the one before it.

```
1. usage.py            → what a "result" looks like
2. exceptions.py        → what "failure" looks like
3. counter.py            → the engine that does the actual measuring
4. budget_guard.py       → the one piece that's actually wired into the app
5. tracking_context.py   → an extra tool that exists, but nobody uses yet
6. __init__.py           → the index of everything above
```

### 1. `usage.py` — "what does a result look like?"

Before anything else, it helps to know what a *finished record* of one LLM
call looks like in this system. It's just a small, unchangeable box of facts:

```
TokenUsage
├── agent_name     — who made the call
├── input_tokens   — tokens sent to the model
├── output_tokens  — tokens the model sent back (given to it, not measured by it)
├── model          — which model was used
├── cost_usd       — optional dollar estimate
└── total_tokens   — input + output, computed automatically
```

Nothing in this file *does* anything — it has no logic. It's just a shape.
Think of it as a receipt template.

### 2. `exceptions.py` — "what does failure look like?"

One tiny custom error:

```python
class TokenBudgetExceeded(ValueError):
    ...
```

This means: "the prompt was too big for the budget you set." It's a subtype
of Python's built-in `ValueError`, so:

- `except TokenBudgetExceeded:` catches *only* the "too big" case.
- `except ValueError:` catches that case *and* other, unrelated bad-input
  errors (like passing a negative budget).

That's it. No logic here either — just a name for a specific failure, so code
elsewhere can react to it specifically instead of guessing what went wrong.

### 3. `counter.py` — the engine that actually measures things

This is the one file with real logic, and everything else in the package is
a thin wrapper around it. It defines a class, `TokenCounter`, that knows how
to talk to a third-party library called **LiteLLM**, which is the thing that
actually knows how each model splits text into tokens.

Continuing our example (system prompt + resume text going in):

```
   your text: "You are a resume parser... <resume text>"
          │
          ▼
   count_tokens(text, model)
          │
          ├─ text is empty? ───────────────► return 0
          │
          ▼
   wrap it like a chat message:
       [{ "role": "user", "content": text }]
          │
          ▼
   count_message_tokens(messages, model)
          │
          ├─ LiteLLM not installed, or it errors? ──► return 0  (fail soft)
          │
          ▼
   ask LiteLLM: litellm.utils.token_counter(model=..., messages=...)
          │
          ▼
   return that number as a plain int
```

Two things worth noticing:

- **It fails soft, not loud.** If LiteLLM isn't installed, or the call to it
  breaks for any reason, you get `0` back instead of a crash. That's a
  deliberate choice — a broken counter shouldn't take down the whole app.
  The tradeoff: if it silently returns `0`, a budget check downstream will
  never trigger, because `0` is never "too big."
- **`TokenCounter` is shared, not recreated.** Code doesn't do
  `TokenCounter()` directly. It calls `get_token_counter()`, which hands back
  one single shared instance for the whole running program (using Python's
  `@lru_cache`, which just means "compute this once, then reuse it forever").
  That way, every part of the app agrees on the same "is LiteLLM even
  available?" answer instead of each recomputing it.

`counter.py` also has two smaller abilities, used *after* an LLM call has
already happened and you already know both token counts:

```
estimate_cost(input_tokens, output_tokens, model)
    → asks LiteLLM for USD price of each side, adds them together

build_usage(...) / log_token_usage(...)
    → packages numbers into a TokenUsage record (from usage.py) and
      optionally writes a structured log line
```

Notice: cost estimation needs *both* input and output counts. This package
never invents output tokens for you — you must already have that number
(normally from the real LLM response).

### 4. `budget_guard.py` — the one piece actually wired into the app today

This is the file that answers **"where's the entry point?"** Its single
function is the gatekeeper:

```python
ensure_token_budget(text, model, max_tokens) -> int
```

Algorithm, in full:

```
ensure_token_budget(text, model, max_tokens)
       │
       ├─ max_tokens < 0 ?  ───────────► raise ValueError
       │                                  (budget itself is invalid)
       ▼
  tokens = get_token_counter().count_tokens(text, model)
       │
       ├─ tokens > max_tokens ?  ──────► raise TokenBudgetExceeded
       │                                  (prompt is too big)
       ▼
  return tokens
       (prompt is within budget — caller may proceed)
```

Back to our running example — a tool is about to send `system prompt +
resume text` to an LLM:

```
  tool builds prompt
        │
        ▼
  ensure_token_budget(prompt, model, max_tokens)   ← this package
        │
    too big?              fits?
        │                   │
        ▼                   ▼
  raise error          the tool goes ahead
  (LLM never called)    and calls the LLM
                              │
                              ▼
                     provider sends back a response,
                     including its OWN token counts —
                     those real counts are captured
                     elsewhere (observability), not here
```

Today, in this codebase, this exact function is called from one specific
place: `src/tools/llm_gateway/structured_output.py`, right before it sends a
prompt to the model, using a budget number pulled from config
(`structured_input_token_budget` in settings). That's the *only* production
call site right now. It is not called from every agent, and it is not called
automatically anywhere else.

### 5. `tracking_context.py` — exists, but not turned on anywhere yet

This file offers an optional helper:

```python
with track_agent_tokens(agent_name, model, task_description) as counter:
    ...
```

What it does, step by step:

```
enter the "with" block
      │
      ▼
estimate tokens in task_description (input only, same engine as above)
      │
      ▼
log: "agent_execution_started", with that estimated number
      │
      ▼
hand the shared counter to your code, let your code run
      │
      ▼
(always, even on error) log: "agent_execution_completed"
```

The important, slightly unusual fact about this file: **nothing in this
codebase calls it today.** It's written and ready, but not plugged into any
agent or pipeline yet. If it *were* used, the intent is to wrap it around
one shared choke point (like a task runner), not paste it into every
individual agent file. Right now, treat it as "available, but dormant."

### 6. `__init__.py` — the index

This file doesn't add new behavior. It just decides what outsiders are
allowed to import directly, and repeats the scope boundary (input-only,
not a dashboard) in its docstring so it's the first thing anyone skimming
the package sees.

```python
from src.core.llm_token_tracker import (
    TokenBudgetExceeded,   # exceptions.py
    TokenCounter,          # counter.py (the class)
    TokenUsage,            # usage.py
    ensure_token_budget,   # budget_guard.py — the real entry point
    get_token_counter,     # counter.py (the shared instance getter)
    track_agent_tokens,    # tracking_context.py — the dormant helper
)
```

---

## Part 4 — Who depends on whom

```
                       __init__.py
                    (just re-exports)
                            │
        ┌───────────────────┼────────────────────┐
        ▼                   ▼                    ▼
 budget_guard.py     tracking_context.py     (any outside caller)
 ensure_token_budget  track_agent_tokens      importing from the package
        │                   │                    │
        └─────────┬─────────┘                    │
                  ▼                              │
            counter.py                            │
            TokenCounter / get_token_counter()  ◄──┘
                  │
       ┌──────────┴──────────┐
       ▼                     ▼
  usage.py               LiteLLM
  (TokenUsage record)    (optional external library)

  exceptions.py: raised BY budget_guard.py when the budget check fails
```

**Rule of thumb:** understand `counter.py`, and every other file is a thin
policy layered on top of it.

---

## Part 5 — Direct answer: "how does this track tokens across a multi-agent system?"

Short, honest answer: **it mostly doesn't — and that's the source of the
confusion.** This package does not walk a graph of agents, does not sum
anything across a run, and has no concept of "stage 1, stage 2, stage 3."
Those ideas belong to a *different* part of the codebase (orchestration),
which this package knows nothing about.

What actually happens today, drawn as the full picture:

```
 ┌───────────────────────────────────────────────────────────┐
 │           orchestration / pipeline (elsewhere)             │
 │   e.g. stage 1 → stage 2 → stage 3 → ... (agents run here) │
 └───────────────────────────┬───────────────────────────────┘
                              │
                              ▼
 ┌───────────────────────────────────────────────────────────┐
 │        somewhere inside one of those stages, a TOOL         │
 │        wants to call an LLM (e.g. structured_output.py)     │
 └───────────┬───────────────────────────────────┬─────────────┘
             │                                   │
             ▼                                   ▼
  ┌─────────────────────────┐        ┌─────────────────────────────┐
  │  ensure_token_budget()   │        │  after the call: real usage  │
  │  (llm_token_tracker)     │        │  is captured by observability │
  │  "is this prompt safe    │        │  (LiteLLM → LangSmith), NOT   │
  │   to send?" — LOCAL      │        │  by this package              │
  │   ESTIMATE, before send  │        │  — the actual source of truth │
  └─────────────────────────┘        └─────────────────────────────┘
```

So: if you're picturing a dashboard that adds up every agent's token spend
across a whole multi-agent run — that dashboard exists, but it's
**observability** (`src/observability`), not this folder. This folder is a
small, local seatbelt used right before *one specific kind of LLM call*,
so that call doesn't go out the door too large.

If you wanted this package to genuinely track a whole multi-agent run
yourself, the pattern would be: call `ensure_token_budget` (or at least
`count_tokens`) at one shared point every agent's LLM call passes through,
and separately use `build_usage` / `log_token_usage` after each real
response to accumulate `TokenUsage` records yourself — this package gives
you the pieces, but no one has wired them together across agents yet.

---

## Part 6 — Cheat sheet: what to actually type

**Guard a prompt before sending it:**

```python
from src.core.llm_token_tracker import ensure_token_budget, TokenBudgetExceeded

try:
    n = ensure_token_budget(prompt, model="gpt-4o", max_tokens=100_000)
except TokenBudgetExceeded:
    # prompt is too big — shrink it, split it, or refuse
    raise
# otherwise: n is the token count, and it's safe to call the LLM
```

**Just count, no budget check:**

```python
from src.core.llm_token_tracker import get_token_counter

n = get_token_counter().count_tokens(prompt, "gpt-4o")
```

**Log a finished call once you know both token counts:**

```python
from src.core.llm_token_tracker import get_token_counter

get_token_counter().log_token_usage(
    agent_name="resume_parser",
    input_tokens=1500,     # you measured this before the call
    output_tokens=320,     # the provider's response told you this
    model="gpt-4o",
)
```

---

## Part 7 — FAQ, answered plainly

**Q: Where is the entry point?**
`ensure_token_budget()` in `budget_guard.py`. Today it's called from one
place in the app: `src/tools/llm_gateway/structured_output.py`, right
before that tool sends a prompt to an LLM.

**Q: What are "stages"?**
Not a concept this package has. Stages belong to the orchestration layer
of the app (how agents are sequenced), which is a different, separate part
of the codebase. This package doesn't know it exists.

**Q: What's "the algorithm"?**
Two small ones:
1. *Counting:* wrap text as a chat message → ask LiteLLM to count it →
   return the number, or `0` if that's not possible.
2. *Budgeting:* count the text → compare to your limit → raise an error if
   it's over, otherwise return the count.

**Q: Does it track output/response tokens?**
No. Only input. Output tokens must be supplied to it by you (normally
copied from the real LLM response).

**Q: Is `tracking_context.py` in use?**
No — it's written, exported, and ready, but no production code calls it
yet.

**Q: Is this the "real" record of what my multi-agent system spent?**
No. That's `src/observability` (LiteLLM → LangSmith), which reflects what
the provider actually reported. This package is a cheap, local, sometimes-
imperfect pre-flight estimate used to stop obviously-too-large prompts
before they're sent.

---

## One sentence to keep

**This package is a local, input-only token ruler with one gate function
(`ensure_token_budget`) wired into one call site — it is not a multi-agent
dashboard, and "stages" are someone else's concept.**
