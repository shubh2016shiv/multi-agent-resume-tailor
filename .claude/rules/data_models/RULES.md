# Rules for src/data_models/ — loaded ONLY when touching data model files

## Data Model Conventions
- All models are Pydantic v2 (`from pydantic import BaseModel`)
- Use `Field()` with descriptions for all fields — documentation is required
- Validation goes in `@field_validator` or `@model_validator`, never in `__init__`
- Keep models focused: one concern per model file

## Key Models
- `job.py` — Job description structure (title, company, requirements, skills, etc.)
- `resume.py` — Resume structure (sections, experience entries, education, skills)
- `strategy.py` — Tailoring strategy (optimization decisions, gap analysis results)
- `evaluation.py` — Quality scores and evaluation metrics

## What NOT to do
- Do NOT add business logic or LLM calls inside data models
- Do NOT use `Optional[T]` — use `T | None` (Python 3.12+ style)
- Do NOT forget `Field(description=...)` on every field
