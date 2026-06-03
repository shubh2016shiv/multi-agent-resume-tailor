# Rules for src/tools/ — loaded ONLY when touching tool files

## Tool Conventions
- All tools extend `BaseTool` from `crewai`
- Tools handle file I/O (PDF/DOCX parsing, text extraction) — NOT LLM logic
- Use `markitdown` for document conversion (PDF/DOCX → Markdown)
- Use `presidio-analyzer` and `presidio-anonymizer` for PII handling

## What NOT to do
- Do NOT call LLM APIs from tool files — leave that to agents
- Do NOT read entire large files into memory at once
- Do NOT store sensitive data (PII) in tool output caches
