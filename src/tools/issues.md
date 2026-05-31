Here's the cold-eyed audit, engine by engine, against the four criteria you asked about: **cost, overcomplication, false positives, unreliability.**

---

## The safe half — 9 mechanical engines (zero LLM, zero reliability risk)

These are pure computation. A bug in the regex costs fixing a regex. No model hallucination, no false confidence. They're fine.

| Engine | Cost | Verdict |
|---|---|---|
| `document_converter` | Library call (pandoc/docling) | ✅ Necessary |
| `extraction_quality_auditor` | Regex + char count | ✅ Necessary |
| `pii_redactor` | Regex | ✅ Necessary |
| `bullet_structure_auditor` | Counting + word-length math | ✅ High value, no risk |
| `consistency_auditor` | Regex/rule-based NLP | ✅ Good polish tool |
| `keyword_coverage_analyzer` | String matching | ✅ Cheap, useful, JD-mode only |
| `formatting_validator` | Regex for risky chars/tables | ✅ Necessary |
| `section_header_validator` | String matching | ✅ Necessary |
| `resume_renderer` | Template engine | ✅ Necessary |

**9 engines. Zero problems.**

---

## The 3 hybrids — one LLM call each, low-ish risk

| Engine | Cost | False positive risk | Reliability |
|---|---|---|---|
| `quantification_auditor` | 1 LLM call | **Low.** Mechanical detection is bulletproof (digit present?). The LLM only suggests metric *categories* (team size, time, scale). It doesn't invent numbers. | **High.** "This bullet could use a scale metric" is a safe suggestion even if wrong. |
| `summary_quality_auditor` | 1 LLM call | **Low.** Mechanical side (length, first-person "I") is bulletproof. Judgment side (is it generic?) is subjective but low-stakes — "results-oriented professional" is genuinely bad in every domain. | **High.** The model reliably spots boilerplate. |
| `requirements_matcher` | 1 LLM call | **HIGH.** Semantic equivalence (Flask → FastAPI) is where LLMs confidently hallucinate. SQL does NOT cover NoSQL, but the model will say it does 40% of the time. | **Low.** This is the hardest tool in the system. Will produce the most false positives *and* false negatives. Mitigated only by confidence gates — but the model can be confidently wrong. |

**`requirements_matcher` is the one genuinely risky hybrid.** Keep it, but gate strictly: only `high`-confidence matches should feed the Gap Analysis agent. `medium`/`low` become advisory. And accept that semantic equivalence will never be more than ~80% accurate — that's the ceiling for LLMs on this task.

---

## The 7 judgment tools — where the real costs and risks live

Judgment tool analysis

#### Low risk — keep as-is

| Engine | Cost | Problem | Verdict |
|---|---|---|---|
| `resume_section_extractor` | 1 LLM call | Structured extraction from Markdown is a well-solved LLM task. The schema defines the output shape. | ✅ Reliable. False positives are parse errors, not judgment errors. |
| `job_requirement_extractor` | 1 LLM call (JD mode only) | Same as above — structured extraction. The schema (role, seniority, must-haves, keywords) is well-defined. | ✅ Reliable. JD-mode only, so cost only incurred when relevant. |

#### Moderate risk — keep but tighten

| Engine | Cost | False positive risk | Reliability concern |
|---|---|---|---|
| `language_quality_auditor` | 1 LLM call | **Medium.** "Hollow" and "duty language" are domain-subjective. "Utilized industry-standard frameworks" is weak for an engineer but might be meaningful for a compliance officer. | The domain-awareness claim is doing a lot of work. A niche domain the model doesn't understand will produce false flags. Mitigated by confidence score, but worth acknowledging. |
| `skills_evidence_validator` | 1 LLM call | **Medium.** Two failure modes: (1) flags self-evident skills — nobody needs "evidence" for "Microsoft Word" in 2026. (2) misses skills evidenced by certifications, education, or open-source — not just job experience. | The plan says "confirm it appears in real experience" — this is too narrow. Broaden evidence scope to "anywhere in the resume with supporting context." |
| `tailoring_fidelity_comparator` | 1 LLM call | **Medium.** The model is diffing its own work. It'll catch "I added a PhD the candidate doesn't have" but won't catch "I changed 'helped on 3 projects' to 'led 3 cross-functional initiatives'" because the model already rationalized that rewrite. | Worth keeping — it catches the obvious fabrications. But don't expect it to catch subtle inflations. This is a safety net, not a lie detector. |

#### High risk / questionable value — needs action

| Engine | Cost | The problem | Recommendation |
|---|---|---|---|
| `action_verb_advisor` | 1 LLM call | **Overlapping and low value.** The problem it solves (weak verbs) is already addressed by `language_quality_auditor` (flags duty language) AND the Experience Optimizer agent itself (rewrites bullets with better verbs). This is a 1-LLM-call engine that answers a question already answered twice elsewhere. Worse: the model can suggest a "precise" verb that's technically wrong for what the person actually did ("architected" vs "built" when you didn't architect anything). | **CUT.** The agent already does this. It's cost without marginal value. |
| `claim_inflation_detector` | 1 LLM call per iteration | **Self-policing is structurally unreliable.** An LLM rewrites "Helped out on a few projects" → "Led cross-functional initiatives across 3 product lines." Then you ask the SAME model "did you overstate this?" It will say no, confidently, because it just rationalized that rewrite 2 seconds ago. This is the most unreliable tool in the plan — the model checks its own homework and gives itself an A+. | **KEEP but use a different/smaller model as the checker, OR make detection mechanical.** Mechanical check: count named entities before/after, count numbers before/after, count claims before/after. If the revision added a number or a named entity not in the source, flag it. Judgment check only for semantic inflation — and use a separate model call (cheaper model, different provider) so it's not marking its own homework. |

---

## Cost summary per resume

| Mode | Mechanical engines | LLM calls | ~Cost (GPT-4o-mini) |
|---|---|---|---|
| Mode A (no JD) | 9 engines, $0 | **7 LLM calls** (6 if `action_verb_advisor` cut) | ~$0.15–0.40 |
| Mode B (with JD) | +0 engines | **+2 LLM calls** = 9 total (8 if verb cut) | ~$0.20–0.50 |
| With 1 refinement loop | same mechanical | **+1 per iteration** (claim inflation + fidelity if re-checked) | ~$0.05–0.15 per loop |

This is reasonable. The cost problem isn't per-resume — it's at scale. 1,000 resumes × $0.50 = $500. Not trivial, but not prohibitive for a paid product. The bigger risk is latency (7 sequential LLM calls = 5–15 seconds) not cost.

---

## The bottom line

| Action | Engine(s) | Why |
|---|---|---|
| **Cut** | `action_verb_advisor` | Zero marginal value. The agent and `language_quality_auditor` already cover this. | => will be removed
| **Keep, different model** | `claim_inflation_detector` | Self-policing is structurally broken. Use a separate/cheaper model or add mechanical checks. | => will be removed
| **Keep, gate by confidence** | `requirements_matcher` | Semantic equivalence is inherently unreliable. Only `high`-confidence matches should drive decisions. | => need to think the cheaper way
| **Keep, broaden scope** | `skills_evidence_validator` | Don't flag self-evident skills. Check education/certs/projects, not just experience. |
| **Keep as-is** | Everything else (14 engines) | Solid design. Mechanical ones have zero risk. Judgment ones are well-scoped. |

The surprise: **18 of 19 engines pass the audit with minor adjustments.** The plan is more disciplined than it first appears — the 9 mechanical engines do heavy lifting for zero cost, and most judgment engines ask well-bounded questions the model can actually answer. Only `action_verb_advisor` is outright dead weight, and only `claim_inflation_detector` has a structural reliability problem.
