# Engine — `resume_renderer` (the `document_rendering/` package)

**Files:** `src/tools/document_rendering/` — `resume_renderer.py`, `latex_escape.py`,
`section_policy.py`, `templates/resume.tex.j2`, and `sidecar/latex_compiler.py`
**Main functions:** `build_resume_tex(resume, profile=None) -> str` and
`render_resume_document(resume, output_path, profile=None) -> Path`
**Type:** Mechanical (templating + LaTeX compilation — no LLM)
**Runs in:** both modes — the **final** step of the pipeline
**Used by:** the orchestrator directly

> This is one *tool* made of several files, because it has several distinct concerns. It does
> **not** return a `ReviewResult` — it produces a finished PDF file.

---

## 1. Purpose (one sentence)

Turn the finished, improved `Resume` object into a polished, ATS-safe **PDF**.

## 2. Why it exists

Everything upstream produces and improves *structured data*. But the candidate needs an actual
document to send. This package lays the `Resume` out into a proven, ATS-friendly LaTeX template
and compiles it to PDF. It is the only tool that produces a file the end user downloads.

It carries one hard requirement: **reliability.** A resume can contain characters that are
poison to LaTeX (`%`, `&`, `$`, `_`, `#`), and the LaTeX toolchain is an external program that
can fail. The package's structure exists to make both of those safe.

## 3. How it works — four cooperating pieces

```
render_resume_document(resume, output_path)
        │
        ▼
  build_resume_tex(resume)                         ◄── PURE PYTHON, no toolchain needed
     │
     ├─ section_policy.py : pick the section ORDER for this candidate
     │      EXPERIENCED  -> Summary, Skills, Experience, Education, ...
     │      ENTRY/grad    -> Education and Skills float UP (thin experience)
     │      STUDENT/intern-> Education first
     │      and group skills by category
     │
     ├─ templates/resume.tex.j2 : the LaTeX template, filled with Jinja2
     │      (uses LaTeX-safe delimiters  << >> , <% %>  so Jinja doesn't clash with LaTeX braces)
     │
     └─ latex_escape.py : escape EVERY user value before it enters the template
            "5% R&D on C#" -> "5\% R\&D on C\#"      ◄── the #1 reliability safeguard
     │
     ▼
  a complete .tex string
     │
     ▼
  sidecar/latex_compiler.py : compile the .tex to PDF with `tectonic`   ◄── the only step that
     │   (runs tectonic in a temp dir, raises with the log on failure)      needs the external tool
     ▼
  a PDF written to output_path
```

### Why split `build_resume_tex` from `render_resume_document`?

`build_resume_tex` is **pure** — it produces the `.tex` text with no external tool. That means
all the risky logic (escaping, ordering, templating) can be unit-tested with **no LaTeX
installed**. Only the final compile needs the `tectonic` binary. So ~90% of the engine is
verifiable anywhere.

### Why a "sidecar"?

Compiling LaTeX needs `tectonic`, an external binary — an *infrastructure* dependency, not
application logic. Following the sidecar pattern, that concern is isolated behind a tiny
interface (`is_render_available()`, `compile_tex_to_pdf(...)`) in `sidecar/`. The pure renderer
never imports the toolchain, so the compile backend could later become a separate service
without touching the rendering code.

## 4. Inputs and outputs

| | |
|---|---|
| **Inputs** | A **final, PII-rehydrated** `Resume` (the orchestrator restores real names before this runs); an `output_path`; an optional layout `profile`. |
| **Outputs** | `build_resume_tex` → a `.tex` string. `render_resume_document` → a `Path` to the produced PDF. |
| **Raises** | `RuntimeError` if `tectonic` isn't installed or compilation fails (with the LaTeX log). `is_render_available()` lets a caller check first. |

## 5. Who calls it

The orchestrator, as the very last step, after the resume has been improved and its PII
rehydrated.

## 6. Gotchas

- **Escaping is non-negotiable.** It's wired as a Jinja filter so it can't be forgotten. A
  single unescaped `%` silently comments out the rest of a LaTeX line. URLs are escaped
  differently from body text (a separate helper), because `%`/`#` are legal in URLs.
- **`tectonic` is a system binary, not a pip install.** In production it's baked into the
  Docker image with a pre-warmed package cache so renders need no network. If it's absent,
  `render_resume_document` raises a clear, actionable error.
- **v1 renders only what the `Resume` model holds.** Projects, a headline, and multiple social
  links aren't in the model yet, so they're gracefully omitted (a planned v2 extends the model).
- **The renderer never *cuts* content to fit a page.** Trimming is the optimizer's job; the
  renderer lays out whatever it's given.

## 7. The same idea, in one line

*Order the sections for this candidate, escape every value, fill a proven LaTeX template, and
compile it to PDF with `tectonic` (isolated in a sidecar) — with the pure text-building step
split out so all the risky logic is testable without LaTeX installed.*
