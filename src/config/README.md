# Configuration Files

This folder contains declarative configuration data. These files are edited to
change application values without changing Python code.

## Directory Layout

```
src/config/
├── settings.yaml          # Application settings (environment, LLM, logging, workflow)
├── agents/                # Per-agent identity: role, goal, backstory, llm, temperature
│   ├── README.md
│   ├── <submodule>.yaml   # One file per agent submodule in src/agents/
│   └── ...
└── tasks/                 # Per-stage task instructions and expected outputs
    ├── README.md
    ├── <submodule>.yaml   # Mirrors the agents/ filename for the same stage
    └── ...
```

`agents.yaml` and `tasks.yaml` at the root are legacy files. They are no longer
loaded by the application. The active configs are the per-file directories above.

## What Belongs Here

- `settings.yaml`: application settings loaded into typed Python models via `get_config()`.
- `agents/<submodule>.yaml`: one agent's LLM identity (role, goal, backstory, model).
- `tasks/<submodule>.yaml`: one stage's task description(s) and expected output(s).
- Prompt templates when they are pure configuration data with no Python logic.

## What Does Not Belong Here

- Pydantic models, Python loading logic, or environment parsing: those live in `src/core/settings/`.
- Python accessors like `get_config()`, `get_agents_config()`, `get_tasks_config()`: same.
- Business logic or LLM calls: those belong in `src/agents/<submodule>/agent.py`.
- Raw API keys or secrets: use environment variables or `.env`.

## Rule Of Thumb

If you are changing a value, edit this folder. If you are changing how values are
loaded, validated, cached, or exposed to Python, edit `src/core/settings/`.
