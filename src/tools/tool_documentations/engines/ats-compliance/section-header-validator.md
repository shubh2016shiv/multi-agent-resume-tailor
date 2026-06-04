# Engine — `section_header_validator`

**File:** `src/tools/ats_compliance/section_header_validator.py`
**Main function:** `audit_section_headers(resume_text: str) -> ReviewResult`
**Type:** Mechanical (case-insensitive string matching — no LLM)
**Runs in:** both modes
**Used by:** the `validate_ats_compliance` agent-facing tool → the ATS Optimization and QA agents

---

## 1. Purpose (one sentence)

Check that the resume's essential sections (experience, education, skills) are present under
names an ATS will recognise.

## 2. Why it exists

An ATS doesn't *read* a resume the way a person does — it classifies blocks of text by
spotting standard section headings. If a candidate gets creative ("My Journey" instead of
"Work Experience", "What I Know" instead of "Skills"), the parser may fail to file that
content under the right field, and it effectively disappears. This engine confirms the
standard, recognised headings are there. Like the formatting check, it's about machine
readability, and it's a domain-neutral convention — so it's mechanical.

## 3. How it works

```
audit_section_headers(resume_text)
        │
        ▼
  pull out the "header-like" lines (short, standalone, Title-Case; markdown '#' stripped)
        │
        ▼
  for each standard section type, see if any known alias appears among those lines:
     summary        -> "Professional Summary" / "Summary" / "Profile" / ...
     experience     -> "Work Experience" / "Professional Experience" / ...   (ESSENTIAL)
     skills         -> "Skills" / "Technical Skills" / ...                    (ESSENTIAL)
     education      -> "Education" / "Academic Background" / ...              (ESSENTIAL)
     certifications -> "Certifications" / ...
        │
        ▼
  classify each section as present / missing-essential / missing-optional
        │
        ▼
  build findings:
     missing ESSENTIAL section (experience/skills/education) ──► MAJOR
     missing OPTIONAL section (summary/certifications)       ──► SUGGESTION
        │
        ▼
  ReviewResult (HIGH confidence; each finding anchored to that section)
```

The recognised names live in one dictionary, `STANDARD_SECTION_HEADERS`, and the three
must-have sections in `ESSENTIAL_SECTIONS`. These are domain-neutral resume conventions, not
professional knowledge, so freezing them as a list is appropriate (unlike, say, "good verbs").

## 4. Inputs and outputs

| | |
|---|---|
| **Input** | `resume_text` — the resume as text or Markdown. |
| **Output** | A `ReviewResult`. Missing **essential** sections are `major`; missing optional ones are `suggestion`. Empty means every essential section is present under a recognised name. All `high` confidence. |
| **Note** | The file also keeps an older `check_section_headers(...) -> str` `@tool`; new code uses `audit_section_headers`. |

## 5. Who calls it

The other engine (with `formatting_validator`) behind `validate_ats_compliance`.

## 6. Gotchas

- **It checks names, not content.** It confirms a "Skills" heading exists; it doesn't judge
  whether the skills under it are any good (that's the diagnostics/skills engines).
- **Creative-header detection is deliberately not attempted** (a `TODO`): flagging "My
  Journey" as a non-standard experience header is hard to do without false-positiving on real
  job titles and project names, so for now the engine only checks whether a *recognised* name
  is present, not whether an *unrecognised* one might be a disguised section.

## 7. The same idea, in one line

*Confirm the three essential section headings (experience, skills, education) are present under
ATS-recognised names — a major finding if one is missing — so the parser can file the content
correctly.*
