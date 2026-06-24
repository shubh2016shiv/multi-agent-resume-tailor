# Resume Tailor

A multi-agent pipeline that reads a resume and a job description, then produces a tailored resume optimized for both applicant tracking systems and human reviewers. Built on CrewAI and LangGraph.

## What it does

You provide two files — your existing resume and a target job description (PDF, DOCX, Markdown, or plain text). The pipeline runs a sequence of specialized LLM agents that analyze the job requirements, identify gaps between your resume and what the job asks for, then rewrite each resume section to close those gaps. The output is a tailored resume in Markdown, DOCX, and optionally PDF.

Every run is scored by a deterministic quality gate before the resume is released. If the score falls below the threshold, the output is withheld so you can inspect the quality report and decide whether to adjust inputs or accept the draft.

## Architecture

```
         tailor_resume()
             |
    ┌────────┼─────────┐
    ▼        ▼         |
 extract  analyze      |          Stage 1 — parallel ingestion
 resume    job         |
    |        |         |
    └────┬───┘         |
         ▼             |
    run_gap_analysis   |          Stage 2 — identify gaps, build strategy
         |             |
    ┌────┼────────┐    |
    ▼    ▼        ▼    |
 summary exp   skills  |          Stage 3 — parallel content generation
    |    |        |    |
    └────┼────────┘    |
         ▼             |
   assemble_ats        |          Stage 4 — merge into ATS-compliant resume
         |             |
         ▼             |
   evaluate_quality ───|          Stage 5 — deterministic quality gate
    |    |             |
    |    └─► patch_ats |          Stage 5b — recover missing sections (no LLM)
    |              |   |
    └──────────────┼───┘
         ▼         |
   rehydrate_pii ◄─┘             Stage 6 — restore redacted personal data
         |
         ▼
   render_final                  Stage 7 — write .md, .docx, .pdf to disk
```

The pipeline is a LangGraph StateGraph. Parallel stages fan out across threads; sequential stages wait for all upstream nodes before running. A conditional edge after quality evaluation decides whether to render the resume or stop.

## Pipeline stages

**Stage 1 — Ingestion.** The resume and job description are converted to Markdown, redacted for personally identifiable information, and parsed into typed data structures. These two nodes run in parallel.

**Stage 2 — Gap analysis.** A code-owned matching engine computes requirement coverage and keyword overlap between the resume and the job description. The gap analysis agent reads those pre-computed facts and produces an alignment strategy — which sections need rewriting and how.

**Stage 3 — Content generation.** Three agents run in parallel, each responsible for one resume section: professional summary, work experience, and skills. The experience node runs one agent call per past role, using a thread pool. A code-owned evidence audit checks the skills output and triggers one rewrite if it finds unsupported claims.

**Stage 4 — ATS assembly.** A single agent merges all optimized sections into one ATS-compliant resume. Code then replaces the assembler's experience section with the verified upstream entries, so no claim the LLM wrote can sneak past the guardrails.

**Stage 5 — Quality evaluation.** Three deterministic engines score the resume on accuracy (are claims supported by the original resume?), relevance (do the requirements match the job?), and ATS structure (does the rendered document pass automated checks?). An LLM provides advisory narrative feedback, but the scores and the pass/fail gate are owned by code — the agent cannot self-certify.

**Stage 5b — ATS section recovery.** If the ATS structure check finds an empty essential section (experience, education, or skills), a deterministic patch restores it from the canonical upstream state. No LLM is involved. If the restore does not fix the failure, the run escalates to human review.

**Stage 6 — PII rehydration.** Placeholders that masked personal data during LLM processing are swapped back to real values. This runs on every path out of quality evaluation, so the returned resume always carries real names, phone numbers, and addresses.

**Stage 7 — Rendering.** The final resume is written to disk as Markdown and DOCX (always) and PDF (best-effort, if a LaTeX toolchain is installed). Files land under the configured output directory, organized by candidate name and job designation.

## Quality evaluation

The quality gate is deterministic, not LLM-judged. Three dimensions are scored by code-owned engines:

- **Accuracy (40%)** — Checks whether every skill and claim in the tailored resume is supported by evidence in the original resume. Skills with no trace in the source are flagged as unsupported.
- **Relevance (35%)** — Measures how many job requirements are addressed by the tailored resume, using keyword coverage against the job description.
- **ATS compliance (25%)** — Renders the resume to LaTeX and inspects the output for structural issues like missing section headers or empty required sections.

The blended score is compared against a configurable threshold. A non-passing ATS structure check overrides any passing blended score — the rendered document is authoritative over self-assessment.

Each run writes a full JSON artifact (including the original resume, the tailored resume, and the quality report) to the output directory. This survives caller crashes and lets you inspect gate-fail runs without re-running the paid pipeline.

## Prerequisites

- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/) package manager
- An LLM provider API key (OpenAI, Anthropic, or any LiteLLM-compatible provider)
- [Tectonic](https://tectonic-typesetting.github.io/) LaTeX engine (optional; only needed for PDF output)

## Quick start

```bash
# Clone and install
git clone <repo-url>
cd resume_tailor
uv sync

# Set your API key
cp env_example .env
# Edit .env — add your LLM_API_KEY and preferred model

# Run the pipeline
uv run python src/agent_orchestrator.py
```

The orchestrator script prompts for resume and job description file paths. Output lands in the configured `output_dir`.

## Configuration

All settings live in `src/config/settings.yaml` and can be overridden via environment variables or a `.env` file.

Key settings:

- `llm.model` — the model used by every agent (any LiteLLM-supported model string)
- `llm.temperature` — creativity control for generated content
- `quality.threshold` — minimum blended score to pass the quality gate (default: 65)
- `feature_flags.enable_pii_redaction` — mask personal data before LLM processing
- `feature_flags.render_draft_on_gate_fail` — write output files even when the gate fails (useful during development)
- `file_paths.output_dir` — where rendered resumes and run artifacts are written

## Observability

The pipeline emits structured logs at every stage boundary, agent call, routing decision, and quality evaluation. Logs are key-value events (not prose), so they are searchable and chartable in any log platform.

[LangSmith](https://smith.langchain.com) integration is available for distributed tracing. When enabled, every agent run appears as a traced span with token usage, latency, and cost metadata. Set `LANGSMITH_API_KEY` in your environment to enable it — the pipeline runs normally without it.

## Design principles

- **Code owns the scores.** LLM agents generate content; deterministic engines evaluate it. The quality gate cannot be passed by an agent claiming its own output is good.
- **Guardrails, not suggestions.** The experience node rejects any LLM output that changes source bullet text. The skills node re-adds any truthful skill the agent dropped. The ATS patch restores empty sections from typed state without calling an LLM.
- **One model string in config.** Every agent reads the same model setting. Changing providers means changing one YAML value.
- **PII never reaches an LLM.** Personal data is redacted before extraction, stored in a run-scoped Redis mapping, and rehydrated after all agent work is complete.
- **Every run is auditable.** The full pipeline state — inputs, strategy, scores, and final resume — is persisted as a JSON artifact whether the gate passes or fails.

## License

MIT
