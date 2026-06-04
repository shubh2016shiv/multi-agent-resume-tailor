# Agent-Facing Tool — `validate_ats_compliance`

**File:** `src/tools/agent_facing_tools.py`
**Function:** `validate_ats_compliance(resume_text: str) -> str`
**CrewAI tool name:** "Validate ATS Compliance"
**Type:** Mechanical (it bundles two mechanical engines — no LLM)
**Runs in:** both modes
**Used by:** the ATS Optimization agent and the Quality Assurance agent

---

## 1. Purpose (one sentence)

Give the ATS Optimization (and QA) agent a single tool that checks the resume will survive an
automated parser: no formatting hazards, and the standard section headings present.

## 2. Why it exists

"Will the machine read this?" has two parts — bad *structure* (tables, tabs, exotic characters) and
missing *section headings*. Two engines cover them. This tool runs both and returns one combined
ATS report, so the agent gets the whole machine-readability picture in one call. Note it takes the
resume as **text** (not a `Resume` object), because formatting and headings are properties of the
*document*, which a parsed object has already discarded.

## 3. How it works

```
validate_ats_compliance(resume_text)
        │
        ▼
1. (no JSON parse — the input is already plain text)
        │
        ▼
2. RUN     audit_ats_formatting(resume_text)   (mechanical: tables/tabs/symbols/columns/...)
           audit_section_headers(resume_text)  (mechanical: are experience/skills/education present?)
        │
        ▼
3. MERGE   _merge([...]) -> one ReviewResult
        │
        ▼
4. RENDER  _render_review_result(merged, "ATS Compliance") -> a readable string
```

Because both backing engines are mechanical, this tool makes **no LLM call** and every finding is
`high` confidence.

## 4. Inputs and outputs

| | |
|---|---|
| **Input** | `resume_text` — the resume as plain text or Markdown (often the *raw converted* document, so it reflects the real layout). |
| **Output** | A string titled "ATS Compliance". Empty findings = ATS-safe. |
| **Cost** | Zero LLM calls. |

## 5. What the agent sees (example)

```
=== ATS Compliance ===
1 ATS formatting issue(s); 3 essential section(s) missing
[major/high] (other)      Incompatible pattern '\t': 1 instance(s)
    advice: Remove the flagged element so an ATS parser can read the resume.
[major/high] (experience) No ATS-recognized 'experience' header found
    advice: Add a section titled 'Work Experience'.
```

## 6. Gotchas

- **Feed it the document text, not the structured `Resume`.** Once text is parsed into a `Resume`,
  the formatting hazards are gone and the section headings are normalised — so this check is only
  meaningful on the raw/rendered document text.
- **It's about machine-readability, not content quality.** A resume can be perfectly ATS-compliant
  and still have weak bullets — those are the diagnostics tools' department.

## 7. The same idea, in one line

*Run the two mechanical ATS engines (formatting hazards + standard headings) on the document text and
merge them into one machine-readability report — no LLM, all high confidence.*
