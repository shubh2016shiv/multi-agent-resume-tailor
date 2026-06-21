# Concept 1 — The Review Contract (the one result shape)

> Read this right after the main `README.md`. Everything else depends on it.

## The problem this solves

Imagine every inspection tool returned results in its own format. The formatting checker
returns a string report. The keyword checker returns a percentage. The honesty checker
returns a list of objects. Now the agent that wants to *combine* all those findings has
to know the quirks of every single tool. Add a new tool and you have to teach every
consumer about it. This does not scale.

So the codebase makes one firm rule:

> **Every review tool returns the exact same shape: a `ReviewResult`.**

A `ReviewResult` is, in plain words, *"a list of findings, plus an optional one-line
verdict, plus an optional score."* Each finding points at a specific place in the resume
and says what is wrong and how to fix it.

Because the shape is always the same, anything downstream — an agent, the quality layer,
the UI — can collect findings from twenty different tools, sort them, and weigh them
*without knowing which tool produced which finding.* That uniformity is the whole point.

The shape is defined in one file: **`src/tools/review_contract/review_models.py`**. They
are plain Pydantic models. Let's walk through them.

---

## The three models

```
ReviewResult                      <- what a tool returns
 ├── comments: list[ReviewComment]   <- the individual findings (can be empty = "all good")
 ├── summary: str                    <- optional one-line verdict, e.g. "2 issues found"
 └── score:   float | None           <- optional number, only when a numeric gate makes sense

ReviewComment                     <- ONE finding
 ├── engine_id:        str           <- which engine produced this finding (for tracing)
 ├── message:          str           <- what was noticed, in plain language
 ├── quoted_text:      str           <- the exact snippet the finding is about
 ├── location:         Location      <- a structured pointer to WHERE in the resume
 ├── severity:         Severity      <- how serious: blocker | major | minor | suggestion
 ├── confidence:       Confidence    <- how sure the tool is: high | medium | low
 ├── advice:           str           <- what to change, and why
 └── proposed_rewrite: str | None    <- optional concrete replacement text

Location                          <- WHERE a finding points
 ├── section:        Section         <- summary | experience | skills | education | ...
 ├── bullet_index:   int | None      <- which bullet within the section, when relevant
 └── character_span: [int,int]|None  <- exact character offsets, for precise UI highlighting
```

---

## Each `ReviewComment` field, in plain words

- **`engine_id`** — the name of the engine that produced this finding (for example
  `"quantification_auditor"`). You do *not* set this yourself inside an engine; the LLM
  gateway stamps it on every finding automatically (see Concept 2). It exists so that any
  finding can be traced back to exactly which engine — and which prompt and token cost —
  produced it.

- **`message`** — a short, plain-language description of *what was noticed*. Example:
  `"The bullet lacks a quantified result."`

- **`quoted_text`** — the exact text the finding is reacting to, so a human can see it
  immediately. Example: the bullet `"Responsible for building models."`. For a
  whole-document issue (like a formatting problem with no single line), this can be empty.

- **`location`** — a *structured* pointer to where the finding applies (which section,
  which bullet). Why structured instead of just relying on `quoted_text`? Because a raw
  quote is fragile: if the model misquotes by one comma, the UI can no longer highlight
  it. The `quoted_text` is for humans to read; the `location` is the reliable anchor the
  UI and any rewrite step actually use.

- **`severity`** — how serious the finding is. One of four levels (most to least severe):
  - `blocker` — must be fixed; the resume is not acceptable as-is.
  - `major` — a real problem worth fixing.
  - `minor` — a small issue or polish item.
  - `suggestion` — optional improvement.

- **`confidence`** — how *sure the tool is* about this finding. One of `high`, `medium`,
  `low`. This is the safety valve for judgment tools (see below).

- **`advice`** — what the candidate should change, and why. This is the actionable part.

- **`proposed_rewrite`** — optionally, concrete replacement text. Many engines leave this
  `None` and only give advice.

---

## The two enums that drive decisions: `Severity` and `Confidence`

These two small enums are how the rest of the system *decides what to act on*.

**`Severity`** answers "how bad is it?" — used to prioritise and to gate. For example, an
orchestrator might refuse to finish while any `blocker` remains.

**`Confidence`** answers "how much should we trust this finding?" There is a firm rule:

> **Mechanical engines always set `confidence = high`.** A measurement is certain: if a
> bullet has 37 words, it has 37 words. There is nothing to be unsure about.
>
> **Judgment engines must set confidence honestly.** When the model is sure, it says
> `high`; when the call depends on domain knowledge it is unsure about, it says `low`.

This single rule is what makes "let the AI model be the expert" *safe in production*: a
low-confidence opinion is surfaced to the user as advice but does **not** force changes,
while a high-confidence finding can drive automated fixes. This is called **confidence
gating**, and it is what stops two agents (say, a rewriter and a reviewer) from arguing
forever over a borderline judgment.

---

## An annotated real example

Here is one finding the quantification engine might produce, fully filled in:

```python
ReviewComment(
    engine_id   = "quantification_auditor",          # stamped by the gateway
    message     = "The bullet lacks a quantified result.",
    quoted_text = "Responsible for building models.",  # the exact bullet
    location    = Location(section=Section.EXPERIENCE, bullet_index=0),
    severity    = Severity.SUGGESTION,                # nice-to-have, not a blocker
    confidence  = Confidence.MEDIUM,                  # a judgment call, so not "high"
    advice      = "Add a metric such as number of models built or accuracy gained.",
    proposed_rewrite = None,                          # advice only, no concrete rewrite
)
```

And the `ReviewResult` that wraps it:

```python
ReviewResult(
    comments = [ <the comment above>, ... ],
    summary  = "2 experience bullets lack a quantified result",
    score    = None,    # this engine has no meaningful single number to report
)
```

An **empty** `comments` list is a perfectly normal, healthy result — it means the tool
looked and found nothing wrong. For example
`ReviewResult(comments=[], summary="All experience bullets include a metric")`.

---

## When is `score` used?

`score` is `None` for most engines, because most have no meaningful single number. It is
set only where a numeric gate is genuinely useful — for example
`requirements_matcher` sets `score` to "the fraction of must-have job requirements the
resume actually evidences" (0.0 to 1.0), which the gap-analysis step can threshold on.

---

## How a finding flows through the shape

```
   an engine looks at the resume
            │
            ▼
   finds something worth noting
            │
            ▼
   builds a ReviewComment   ──►  message + quoted_text + location
            │                    + severity + confidence + advice
            │
            ▼
   collects all comments into a ReviewResult(comments=[...], summary=..., score=...)
            │
            ▼
   returns it  ──►  the caller can now merge it with any OTHER engine's
                    ReviewResult, because they are the same shape
```

---

## What to take away

1. Every review tool returns a `ReviewResult` — a list of `ReviewComment`s (+ summary,
   + optional score). No exceptions.
2. A `ReviewComment` always carries *where* (`location`), *what* (`message` /
   `quoted_text`), *how serious* (`severity`), *how sure* (`confidence`), and *what to do*
   (`advice`).
3. Mechanical findings are always `high` confidence; judgment findings rate themselves
   honestly, and that rating is what the system uses to decide whether to act.
4. `engine_id` is stamped automatically so every finding is traceable.

Next: **[02-the-llm-gateway-and-the-engine-types.md](02-the-llm-gateway-and-the-engine-types.md)**
— the single place the LLM is called, and the three engine types in depth.
