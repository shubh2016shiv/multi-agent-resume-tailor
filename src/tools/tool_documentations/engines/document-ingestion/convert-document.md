# Engine — `document_converter`

**File:** `src/tools/document_ingestion/document_converter.py`
**Main function:** `convert_document_to_markdown(file_path: str) -> str`
**Type:** Mechanical (a library call — no LLM)
**Runs in:** both modes — it is the **very first** step of the whole pipeline
**Used by:** the orchestrator directly (not handed to an agent)

> This is a pipeline-stage engine. Unlike the review engines, it does **not** return a
> `ReviewResult` — it returns plain Markdown text. See `concepts/03`, Part C.

---

## 1. Purpose (one sentence)

Turn whatever file the candidate uploaded (PDF, Word, etc.) into clean Markdown text that
the rest of the pipeline can work with.

## 2. Why it exists

Every later step — quality checking, PII redaction, the LLM that extracts a structured
`Resume` — needs **text**, not a binary `.pdf` or `.docx`. So the very first thing the
pipeline does is normalise the upload into one consistent text format: Markdown. Doing
this once, up front, means nothing downstream has to know or care what the original file
type was.

## 3. How it works

It is a thin, reliable wrapper around the `markitdown` library (the same engine used in
many document-to-text pipelines). The only logic it adds is format validation and a
shortcut for files that are already text.

```
convert_document_to_markdown(file_path)
        │
        ▼
  does the file exist?  ── no ──► raise FileNotFoundError
        │ yes
        ▼
  is the extension supported?  ── no ──► raise ValueError (lists supported types)
        │ yes
        ▼
  is it already text (.md / .txt)?  ── yes ──► just read the file and return it
        │ no  (.pdf .docx .pptx .xlsx)
        ▼
  hand it to markitdown ──► returns extracted Markdown
        │
        ▼
  log "Converted X (N chars)" and return the text
```

Supported formats live in one constant, `SUPPORTED_FORMATS` (`.pdf .docx .md .txt .pptx
.xlsx`). Two small helpers expose it: `get_supported_formats()` and
`is_format_supported(path)`.

## 4. Inputs and outputs

| | |
|---|---|
| **Input** | `file_path` — a path to the uploaded document. |
| **Output** | A Markdown `str` of the document's text content. |
| **Raises** | `FileNotFoundError` (no such file), `ValueError` (unsupported type), `OSError` (markitdown failed to convert). |

It deliberately **raises** rather than returning an error object, because a failed
conversion means the pipeline genuinely cannot continue — there is nothing to review.

## 5. Who calls it

The orchestrator, as step 1. Its output (Markdown) feeds straight into
`audit_extraction_quality` (did the conversion work?) and then `redact_pii`.

## 6. Gotchas

- **Conversion quality varies by source file.** A scanned/image PDF, or a heavily
  multi-column layout, can convert to garbled or near-empty text. This engine does **not**
  judge that — it just returns whatever markitdown produced. Catching a bad conversion is
  the *next* engine's job (`audit_extraction_quality`). Always run that check after this.
- **It is stateless and reusable.** One module-level `MarkItDown()` instance is shared;
  that's fine because it holds no per-file state.

## 7. The same idea, in one line

*The pipeline's front door: validate the file type, then turn the upload into Markdown
text (reading `.txt`/`.md` directly, running everything else through markitdown).*
