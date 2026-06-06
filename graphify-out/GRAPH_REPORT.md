# Graph Report - resume_tailor  (2026-06-05)

## Corpus Check
- 112 files · ~126,176 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1549 nodes · 4265 edges · 83 communities (76 shown, 7 thin omitted)
- Extraction: 74% EXTRACTED · 26% INFERRED · 0% AMBIGUOUS · INFERRED: 1105 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `9e1272b2`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 53|Community 53]]
- [[_COMMUNITY_Community 54|Community 54]]
- [[_COMMUNITY_Community 55|Community 55]]
- [[_COMMUNITY_Community 56|Community 56]]
- [[_COMMUNITY_Community 57|Community 57]]
- [[_COMMUNITY_Community 58|Community 58]]
- [[_COMMUNITY_Community 59|Community 59]]
- [[_COMMUNITY_Community 60|Community 60]]
- [[_COMMUNITY_Community 62|Community 62]]
- [[_COMMUNITY_Community 63|Community 63]]
- [[_COMMUNITY_Community 64|Community 64]]
- [[_COMMUNITY_Community 65|Community 65]]
- [[_COMMUNITY_Community 66|Community 66]]
- [[_COMMUNITY_Community 67|Community 67]]
- [[_COMMUNITY_Community 68|Community 68]]
- [[_COMMUNITY_Community 70|Community 70]]
- [[_COMMUNITY_Community 71|Community 71]]
- [[_COMMUNITY_Community 72|Community 72]]
- [[_COMMUNITY_Community 73|Community 73]]
- [[_COMMUNITY_Community 74|Community 74]]
- [[_COMMUNITY_Community 75|Community 75]]
- [[_COMMUNITY_Community 76|Community 76]]
- [[_COMMUNITY_Community 77|Community 77]]
- [[_COMMUNITY_Community 78|Community 78]]
- [[_COMMUNITY_Community 79|Community 79]]
- [[_COMMUNITY_Community 80|Community 80]]
- [[_COMMUNITY_Community 81|Community 81]]

## God Nodes (most connected - your core abstractions)
1. `Resume` - 187 edges
2. `JobDescription` - 139 edges
3. `Experience` - 118 edges
4. `AlignmentStrategy` - 115 edges
5. `ReviewResult` - 100 edges
6. `OptimizedSkillsSection` - 82 edges
7. `Location` - 63 edges
8. `ReviewComment` - 63 edges
9. `Section` - 57 edges
10. `Severity` - 56 edges

## Surprising Connections (you probably didn't know these)
- `str` --uses--> `OptimizedResume`  [INFERRED]
  examples/agents/example_ats_optimization.py → src/agents/ats_optimization_agent.py
- `int` --uses--> `OptimizedResume`  [INFERRED]
  examples/agents/example_ats_optimization.py → src/agents/ats_optimization_agent.py
- `str` --uses--> `AlignmentStrategy`  [INFERRED]
  examples/agents/example_gap_analysis.py → src/data_models/strategy.py
- `int` --uses--> `AlignmentStrategy`  [INFERRED]
  examples/agents/example_gap_analysis.py → src/data_models/strategy.py
- `str` --uses--> `JobDescription`  [INFERRED]
  examples/agents/example_job_analyzer.py → src/data_models/job.py

## Import Cycles
- None detected.

## Communities (83 total, 7 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.06
Nodes (70): AccuracyMetrics, create_mock_job_description(), create_mock_tailored_resume(), main(), print_section(), Example: Quality Assurance Reviewer Agent ======================================, Helper to print formatted sections with truncation., Create mock tailored resume data for the example.      In a real workflow, this (+62 more)

### Community 1 - "Community 1"
Cohesion: 0.08
Nodes (37): audit_ats_formatting(), _build_validation_report(), _find_formatting_issues(), _find_incompatible_patterns(), _find_masked_hyperlinks(), _find_multi_column_layout(), _find_pdf_extraction_artifacts(), _find_problematic_characters() (+29 more)

### Community 2 - "Community 2"
Cohesion: 0.11
Nodes (29): analyze_action_verbs(), analyze_verb_noun_pairs(), assess_impact_level(), calculate_relevance_score(), check_experience_quality(), check_keyword_integration(), check_specificity(), count_quantified_achievements() (+21 more)

### Community 3 - "Community 3"
Cohesion: 0.06
Nodes (73): _assess_keyword_coverage(), _assess_presentation_quality(), _assess_truthfulness(), _build_skill_inference_prompt(), _build_validation_prompt(), _calculate_confidence_score(), categorize_skills(), check_skills_quality() (+65 more)

### Community 4 - "Community 4"
Cohesion: 0.24
Nodes (10): get_resume_md(), Returns the sample resume markdown string., main(), print_section(), Example: Professional Summary Writer Agent =====================================, Helper to print formatted sections with truncation., create_summary_writer_agent(), Create and configure the Professional Summary Writer agent.      This is the mai (+2 more)

### Community 5 - "Community 5"
Cohesion: 0.07
Nodes (42): F, Observability Module ====================  This module provides centralized o, get_weave_project_url(), init_observability(), is_weave_enabled(), _lazy_import_weave(), log_iteration_metrics(), Weave Observability Integration ================================  This module pr (+34 more)

### Community 6 - "Community 6"
Cohesion: 0.04
Nodes (46): log_iteration_progress(), Experience Section Optimizer Agent - Advanced Iterative Improvement System =====, # WHY: Tool receives strings from agent; need structured data, # NOTE: CrewAI sometimes passes lists directly instead of JSON strings,, # WHY: Per-bullet feedback allows targeted improvements, # WHY: Single metric for agent to decide continue/stop, # WHY: Agent needs specific guidance, not just a score, # WHY: Agent uses this boolean to decide iteration continuation (+38 more)

### Community 7 - "Community 7"
Cohesion: 0.06
Nodes (31): BaseSettings, AgentDefaults, ApplicationConfig, FeatureFlags, FilePathsConfig, LLMConfig, LLMGoogleConfig, LLMResilienceConfig (+23 more)

### Community 8 - "Community 8"
Cohesion: 0.17
Nodes (15): audit_quantification(), _find_unquantified_bullets(), _format_bullets_for_prompt(), _has_number(), Quantification auditing: which experience bullets lack a metric, and what could, Flag experience bullets with no number and suggest what metric each could add., Return achievement bullets that contain no digit (mechanical detection)., True if the bullet contains any digit.      Note: counts only digits, so spelled (+7 more)

### Community 9 - "Community 9"
Cohesion: 0.09
Nodes (38): assemble_resume_components(), ATSValidationResult, generate_json_resume(), generate_markdown_resume(), Generate JSON representation of the resume.      This function creates a structu, Validate, format, and finalize the resume for ATS compatibility.      This tool, Comprehensive ATS compatibility validation result.      This model represents th, Perform comprehensive ATS compatibility validation.      This function conducts (+30 more)

### Community 10 - "Community 10"
Cohesion: 0.23
Nodes (16): _extract_ats_validation_data_only(), _extract_contact_and_education_only(), _extract_experience_list_only(), _extract_skills_list_only(), _extract_summary_text_only(), format_ats_optimization_context(), ATS Optimization Agent Formatter =================================  PURPOSE: ---, Extract ONLY the optimized_experiences list from OptimizedExperienceSection mode (+8 more)

### Community 11 - "Community 11"
Cohesion: 0.18
Nodes (13): create_experience_optimizer_agent(), Create and configure the Experience Section Optimizer agent.      This is the ma, Agent, main(), COMPREHENSIVE LLM TEST: Experience Optimizer Agent - Iterative Improvement  Th, Verify the evaluate_experience_bullets tool is properly attached to agent., Test the tool works correctly when invoked directly (control test)., Run all tests to verify iterative improvement mechanism. (+5 more)

### Community 12 - "Community 12"
Cohesion: 0.16
Nodes (30): BulletDraft, IterativeExperienceOptimization, A single iteration of bullet point improvement.      This model captures the sta, Extended output model with iteration history for self-improvement tracking., Experience, Represents a single work experience entry in a resume.      This model captures, AlignmentStrategy, A comprehensive, structured plan for tailoring the resume.      This model is (+22 more)

### Community 13 - "Community 13"
Cohesion: 0.11
Nodes (24): A single draft version of the professional summary., DEPRECATED: This function is no longer needed with output_pydantic.      When us, SummaryDraft, validate_summary_output(), Config, JobLevel, JobRequirement, Data Models for Job Description Structure ------------------------------------- (+16 more)

### Community 14 - "Community 14"
Cohesion: 0.11
Nodes (29): _load_mock_experience(), _load_mock_professional_summary(), _load_mock_resume_extraction(), _load_mock_skills(), main(), str, ATS Optimization Specialist Agent Integrity Check  This script tests the ATS Opt, Load or create mock professional summary output. (+21 more)

### Community 15 - "Community 15"
Cohesion: 0.10
Nodes (24): CircuitBreaker, create_rate_limiter(), create_retry_decorator(), get_circuit_breaker(), get_resilience_stats(), Centralized Resilience Module ------------------------------  This module pro, Get or create a circuit breaker for the specified LLM provider.      Circuit b, Create a retry decorator with exponential backoff using tenacity.      This ha (+16 more)

### Community 16 - "Community 16"
Cohesion: 0.10
Nodes (20): analyze_keyword_integration(), check_summary_quality(), count_words(), DraftEvaluationTool, Professional Summary Writer Agent - Narrative Crafting Specialist ==============, # WHY: Provides the foundation for narrative generation and quality assessment, # WHY: Produces a fully functional agent ready for narrative generation and qual, # WHY: Enables the agent to improve its own output through iterative evaluation (+12 more)

### Community 17 - "Community 17"
Cohesion: 0.18
Nodes (18): _apply_redaction(), _build_placeholder_map(), _detect_pii_spans(), PII redaction: mask personal data before any text reaches an external LLM.  Dete, True if the two character spans share any range., Map each unique (entity_type, value) to a stable placeholder like '[EMAIL_ADDRES, Replace each detected span with its placeholder, right-to-left so offsets stay v, # TODO: Education completion year. (+10 more)

### Community 18 - "Community 18"
Cohesion: 0.15
Nodes (36): OptimizedResume, Final optimized resume with comprehensive metadata.      This model represents t, OptimizedExperienceSection, Structured output containing optimized work experience entries.      This model, ProfessionalSummary, Structured output containing multiple strategic drafts of the professional summa, QualityReport, A comprehensive, structured report on the quality of the tailored resume. (+28 more)

### Community 19 - "Community 19"
Cohesion: 0.23
Nodes (14): _filter_job_for_qa(), _filter_resume_for_qa(), format_quality_assurance_context(), Quality Assurance Agent Formatter ==================================  Formats an, Format context data for Quality Assurance Agent using specified format (TOON or, Filter resume dictionary to include only fields needed for QA evaluation.      Q, Filter job description dictionary to include only fields needed for QA evaluatio, FormatType (+6 more)

### Community 20 - "Community 20"
Cohesion: 0.17
Nodes (18): LogContext, Thread-safe context manager for correlation IDs and structured logging., Result of skill validation check, Check if evidence quotes actually exist in experience text, SkillValidationResult, _verify_evidence_in_experience(), Skill requirement criticality levels for prioritization.      This enum is vit, SkillImportance (+10 more)

### Community 21 - "Community 21"
Cohesion: 0.22
Nodes (10): Review-specific wrapper over the structured-output harness.  Judgment review eng, _parse_into_model(), Generic structured-output harness: call the LLM and get a validated Pydantic mod, Call the LLM and return a validated instance of output_model.      Args:, # TODO: Accept an optional model= override so bounded tool calls can use a, Validate the model's output into output_model.      Accepts an already-parsed in, request_structured_output(), object (+2 more)

### Community 22 - "Community 22"
Cohesion: 0.17
Nodes (22): estimate_tokens(), filter_nested_dict(), format_data(), _needs_quotes(), Base Formatter Module =====================  Provides core formatting utilities, Convert a Python dictionary to TOON (Token-Oriented Object Notation) format., Recursively filter a dictionary to only include specified fields.      This func, Convert a Python value to Markdown format string.      Args:         value: Pyth (+14 more)

### Community 23 - "Community 23"
Cohesion: 0.22
Nodes (19): extract_job_requirements(), Extract job requirements into a normalized dictionary with importance levels., _extract_job_requirements_for_alignment(), _extract_strategy_gaps_for_context(), _extract_work_experience_for_optimization(), format_experience_optimizer_context(), Experience Optimizer Agent Formatter ====================================  Fo, Extract job requirements and ATS keywords for experience alignment.      The E (+11 more)

### Community 24 - "Community 24"
Cohesion: 0.20
Nodes (22): _extract_additional_resume_context(), _extract_ats_keywords_for_optimization(), _extract_education_and_certifications(), _extract_experience_summaries_for_comparison(), _extract_job_metadata_for_context(), _extract_job_requirements_for_comparison(), _extract_skills_for_comparison(), format_gap_analysis_context() (+14 more)

### Community 25 - "Community 25"
Cohesion: 0.11
Nodes (27): Language, audit_consistency(), _check_repeated_opening_verbs(), _check_tense_consistency(), _get_nlp(), _make_finding(), _opening_verb_tense(), Consistency auditing: do experience bullets read consistently within a role?  Me (+19 more)

### Community 26 - "Community 26"
Cohesion: 0.13
Nodes (18): calculate_coverage_stats(), check_analysis_quality(), extract_resume_skills(), normalize_skill(), Gap Analysis Specialist Agent - Strategic Alignment Engine =====================, # WHY: Ensures consistent agent behavior across different environments, DEPRECATED: This function is no longer needed with output_pydantic.      When us, # WHY: Enables accurate skill matching and gap identification (+10 more)

### Community 27 - "Community 27"
Cohesion: 0.13
Nodes (18): evaluate_experience_bullets(), get_iteration_tracker_state(), parse_json_robust(), Robustly parse JSON that may contain extra text after the JSON object.      PURP, Reset the iteration tracker for a new optimization session.      STAGE: 2.2.1 -, Get current state of iteration tracker for debugging/observability.      STAGE:, STAGE 2A: Self-Evaluation Tool - Agent's Quality Assessment Interface      =====, reset_iteration_tracker() (+10 more)

### Community 28 - "Community 28"
Cohesion: 0.19
Nodes (18): audit_summary_quality(), _check_first_person(), _check_length(), _make_finding(), Summary quality auditing: is the professional summary the right length, person,, Flag first-person pronouns via exact token match (so 'academy' never matches 'my, Build a SUMMARY-section comment for this engine's mechanical checks (HIGH confid, # TODO: Calibrate MIN/MAX_SUMMARY_WORDS on real summaries. (+10 more)

### Community 29 - "Community 29"
Cohesion: 0.11
Nodes (23): create_ats_optimization_agent(), Create and configure the ATS Optimization Specialist agent.      This is the mai, create_gap_analysis_agent(), Create the Gap Analysis Specialist agent.      Returns:         Agent: A configu, create_resume_extractor_agent(), get_agent_info(), _get_default_config(), _load_agent_config() (+15 more)

### Community 30 - "Community 30"
Cohesion: 0.13
Nodes (20): DEPRECATED: This function is no longer needed with output_pydantic.      When us, validate_job_output(), DEPRECATED: This function is no longer needed with output_pydantic.      When us, validate_resume_output(), extract_resume(), Resume extraction: turn privacy-redacted resume Markdown into a structured Resum, # TODO: Remove the prompt from tasks.yaml once the Resume Extractor agent is, Extract a structured Resume from privacy-redacted resume Markdown.      Args: (+12 more)

### Community 31 - "Community 31"
Cohesion: 0.19
Nodes (16): _condense_requirement_text(), _extract_job_targets_for_alignment(), _extract_skills_for_optimization(), format_skills_optimizer_context(), Skills Optimizer Agent Formatter =================================  Formats a, Condense verbose job requirement text to reduce token usage.      OPTIMIZATION, Extract job requirements and ATS keywords for skills targeting.      The Skill, Format and filter data for the Skills Optimizer Agent.      This function orch (+8 more)

### Community 32 - "Community 32"
Cohesion: 0.16
Nodes (14): _load_agent_config(), Load the agent configuration from agents.yaml.      This function provides a sin, get_agent_info(), _load_agent_config(), Load the agent configuration from agents.yaml.      This function provides a sin, Get information about this agent for debugging or monitoring.      Returns:, get_agent_info(), _get_default_config() (+6 more)

### Community 33 - "Community 33"
Cohesion: 0.19
Nodes (18): audit_extraction_quality(), _check_extraction_artifacts(), _check_fragmentation(), _check_text_volume(), _make_finding(), Extraction quality auditing: did the document-to-Markdown conversion succeed?  R, Build a document-level ReviewComment for this engine (mechanical, so HIGH confid, # TODO: Tune these thresholds against real converted resumes. (+10 more)

### Community 34 - "Community 34"
Cohesion: 0.17
Nodes (19): audit_bullet_structure(), _check_bullet_counts(), _check_bullet_lengths(), _make_finding(), Bullet structure auditing: are experience bullets balanced by recency and length, Build an EXPERIENCE-section comment for this engine (mechanical, so HIGH confide, # TODO: Tune the count thresholds against real resumes., Flag recency-imbalanced bullet counts and over-long bullets across roles.      A (+11 more)

### Community 35 - "Community 35"
Cohesion: 0.12
Nodes (18): _get_default_config(), NO DEFAULTS - Configuration must come from agents.yaml.          Raises:, # WHY: Prevents downstream errors and ensures strategic data consistency, check_analysis_quality(), get_agent_info(), _get_default_config(), _load_agent_config(), Job Description Analyst Agent - Requirements Extraction System ================= (+10 more)

### Community 36 - "Community 36"
Cohesion: 0.05
Nodes (64): CompletedProcess, A computed property that returns a list of all "must-have" skill names., A computed property that returns a list of "should-have" and "nice-to-have" skil, escape_latex(), escape_latex_url(), LaTeX escaping -- the reliability core of the renderer.  Any resume field can co, Escape LaTeX-special characters in body text so it compiles verbatim.      Args:, Escape only the characters that are special inside a \\href{...} target.      UR (+56 more)

### Community 37 - "Community 37"
Cohesion: 0.23
Nodes (15): _extract_job_for_summary_writing(), _extract_resume_for_summary_writing(), _extract_strategy_for_summary_writing(), format_professional_summary_context(), Professional Summary Agent Formatter =====================================  P, Extract ONLY the job fields needed for professional summary generation.      T, Extract ONLY the strategy fields needed for professional summary generation., Format and filter resume, job, and strategy data for the Summary Writer Agent. (+7 more)

### Community 38 - "Community 38"
Cohesion: 0.10
Nodes (34): The full output of one tool: its comments plus an optional verdict., ReviewResult, Render the claim-bearing sections (summary, experience, skills, education, certs, render_resume(), ReviewResult, Resume, float, Language (+26 more)

### Community 39 - "Community 39"
Cohesion: 0.31
Nodes (8): _load_mock_alignment_strategy(), _load_mock_resume_extraction(), main(), str, Experience Section Optimizer Agent Integrity Check  This script tests the Experi, Load or create mock alignment strategy., Load or create mock resume extraction output., Run integrity check for Experience Section Optimizer agent.

### Community 40 - "Community 40"
Cohesion: 0.31
Nodes (8): _load_mock_job_analysis(), _load_mock_resume_extraction(), main(), str, Gap & Alignment Strategist Agent Integrity Check  This script tests the Gap & Al, Load or create mock job analysis output., Load or create mock resume extraction output., Run integrity check for Gap & Alignment Strategist agent.

### Community 41 - "Community 41"
Cohesion: 0.31
Nodes (8): _load_mock_alignment_strategy(), _load_mock_resume_extraction(), main(), str, Skills Section Strategist Agent Integrity Check  This script tests the Skills Se, Load or create mock alignment strategy., Load or create mock resume extraction output., Run integrity check for Skills Section Strategist agent.

### Community 42 - "Community 42"
Cohesion: 0.25
Nodes (8): get_agent_info(), Get information about this agent for debugging or monitoring.      Returns:, get_agent_info(), _get_default_config(), _load_agent_config(), Load the agent configuration from agents.yaml.      This function provides a sin, Provide default configuration as a fallback.      IMPORTANT: This fallback shoul, Get information about this agent for debugging or monitoring.      Returns:

### Community 43 - "Community 43"
Cohesion: 0.15
Nodes (20): evaluate_single_bullet(), generate_bullet_critique(), optimize_bullets_iteratively(), STAGE 3A: Single Bullet Evaluation - Comprehensive Quality Assessment      Evalu, STAGE 3B: Critique Generation - Actionable Improvement Instructions      Generat, STAGE 4: Iterative Optimization Loop - Self-Improvement Framework      Iterative, Test the evaluate_experience_bullets CrewAI tool., Test the iterative improvement framework (evaluation loop). (+12 more)

### Community 44 - "Community 44"
Cohesion: 0.27
Nodes (9): get_job_desc_md(), parse_json_output(), Validates data using the provided validator function.     Returns True if valid, Returns the sample job description markdown string., Parses JSON from agent output, handling markdown code blocks., validate_output(), Any, bool (+1 more)

### Community 45 - "Community 45"
Cohesion: 0.15
Nodes (16): main(), print_section(), Example: Skills Section Optimizer Agent =======================================, Helper to print formatted sections., # IMPORTANT: We use output_pydantic=OptimizedSkillsSection to enforce structured, Validate and parse skills optimization data into Pydantic model.      Args:, validate_skills_output(), str (+8 more)

### Community 46 - "Community 46"
Cohesion: 0.33
Nodes (6): _load_mock_alignment_strategy(), main(), str, Professional Summary Writer Agent Integrity Check  This script tests the Profess, Load or create mock alignment strategy., Run integrity check for Professional Summary Writer agent.

### Community 47 - "Community 47"
Cohesion: 0.29
Nodes (4): Returns the total number of skills in the optimized section., Returns the number of skill categories., Returns the number of skills added through inference., int

### Community 48 - "Community 48"
Cohesion: 0.21
Nodes (12): KeywordDensityReport, Validation result for a single resume section.      This model captures the vali, Analysis of keyword usage and density in resume content.      This model provide, SectionValidation, Education, Represents an educational background entry in a resume., BaseModel, create_sample_data() (+4 more)

### Community 49 - "Community 49"
Cohesion: 0.11
Nodes (30): audit_section_headers(), _build_header_report(), check_section_headers(), _classify_sections(), _extract_header_lines(), _find_matching_header(), get_standard_headers(), _make_finding() (+22 more)

### Community 50 - "Community 50"
Cohesion: 0.40
Nodes (3): Calculates the duration of this work experience in years.          This is a com, Calculates the total years of professional experience from all jobs.          Th, float

### Community 51 - "Community 51"
Cohesion: 0.50
Nodes (3): main(), Job Description Analyst Agent Integrity Check  This script tests the Job Descrip, Run integrity check for Job Description Analyst agent.

### Community 52 - "Community 52"
Cohesion: 0.50
Nodes (3): main(), Resume Content Extractor Agent Integrity Check  This script tests the Resume Con, Run integrity check for Resume Content Extractor agent.

### Community 53 - "Community 53"
Cohesion: 0.12
Nodes (28): analyze_keyword_coverage(), _build_coverage_comments(), calculate_keyword_density(), _compute_density_metrics(), _count_keywords(), _count_whole_token(), _format_density_report(), get_optimal_keyword_density_range() (+20 more)

### Community 54 - "Community 54"
Cohesion: 0.50
Nodes (3): main(), Example: Resume Tailoring with Iterative Quality Enforcement  This example dem, Run resume tailoring with quality enforcement.      This example shows:     1

### Community 55 - "Community 55"
Cohesion: 0.50
Nodes (3): main(), Quick Start Example: Resume Tailoring Workflow  This example demonstrates how, Demonstrate basic resume tailoring workflow.      WORKFLOW EXPLAINED:     1.

### Community 56 - "Community 56"
Cohesion: 0.15
Nodes (25): Manual smoke test for the agent-facing tools — drives each from multiple angles, ReviewResult, str, analyze_jd_keyword_coverage(), audit_experience_quality(), audit_summary(), audit_truthfulness(), check_skills_evidence() (+17 more)

### Community 59 - "Community 59"
Cohesion: 0.09
Nodes (33): log_iteration_tracker_summary(), Log a comprehensive summary of all tool calls for observability.      STAGE: 2.2, Validate that iteration is actually happening and progressing.      STAGE: 2.3.1, validate_iteration_progress(), create_test_strategy(), print_test_header(), Comprehensive Test Suite for Experience Optimizer Agent Enhancements  This test, Run all tests in the suite. (+25 more)

### Community 62 - "Community 62"
Cohesion: 0.11
Nodes (17): BoundLogger, configure_structlog(), get_logger(), Structured Logging with structlog ==================================  Industr, # WHY: Prevents debug logs from cluttering production, # WHY: Simple, works everywhere, integrates with standard logging, # WHY: Avoid recreating loggers for the same name, # WHY: structlog integrates with Python's logging module for file handling (+9 more)

### Community 67 - "Community 67"
Cohesion: 0.12
Nodes (15): check_ats_quality(), _get_default_config(), ATS Optimization Specialist Agent - Final Quality Assurance System =============, # WHY: Final checkpoint before resume is considered ready for submission, Validate that the agent's output conforms to the OptimizedResume model.      Thi, Perform quality checks on the optimized resume output.      This function valida, # WHY: Offloads heavy processing from the LLM to Python code for reliability, # WHY: Enables debugging, monitoring, and validation of agent functionality (+7 more)

### Community 68 - "Community 68"
Cohesion: 0.20
Nodes (14): Experience, Resume, ReviewResult, str, _build_payload(), _format_evidence(), _format_experience(), Skills-evidence validation: is every listed skill actually backed by the resume? (+6 more)

### Community 70 - "Community 70"
Cohesion: 0.21
Nodes (9): get_supported_formats(), is_format_supported(), Document conversion: any supported file format → clean Markdown text.  Supported, Return the file extensions this module can convert., Return True if the file extension is supported for conversion., Job requirement extraction: turn job-description Markdown into a structured JobD, # TODO: Flag valid-but-empty extraction (no requirements / no ats_keywords). The, bool (+1 more)

### Community 71 - "Community 71"
Cohesion: 0.18
Nodes (11): audit_language_quality(), _collect_bullets(), Language quality auditing: are experience bullets achievements, or duties and fi, Flag duty-language and hollow phrasing in experience bullets, judged for the fie, # TODO: Also review role description text, not just achievements., # TODO: Domain coverage is uneven -- unfamiliar fields yield mostly low-confiden, Gather every achievement bullet across all roles., Experience (+3 more)

### Community 72 - "Community 72"
Cohesion: 0.24
Nodes (10): main(), print_section(), Example: Job Analyzer Agent ===========================  OBJECTIVE: ---------- T, Helper to print formatted sections with truncation., create_job_analyzer_agent(), Create and configure the Job Description Analyst agent.      This is the main en, get_tasks_config(), Loads and returns the task definitions from tasks.yaml. (+2 more)

### Community 73 - "Community 73"
Cohesion: 0.24
Nodes (10): convert_document_to_markdown(), Convert a supported document to clean Markdown text.      Args:         file_pat, extract_job_requirements(), Extract a structured JobDescription from job-description Markdown.      Mode B o, banner(), main(), str, End-to-end pipeline run on a REAL resume PDF, feeding each stage's intermediate (+2 more)

### Community 74 - "Community 74"
Cohesion: 0.24
Nodes (9): main(), print_section(), Example: Experience Section Optimizer Agent ====================================, Helper to print formatted sections with truncation., # IMPORTANT: We use output_pydantic=OptimizedExperienceSection to enforce struct, Validate that the agent's output conforms to experience section models.      Thi, validate_experience_output(), int (+1 more)

### Community 75 - "Community 75"
Cohesion: 0.29
Nodes (7): main(), print_section(), Example: ATS Optimization Specialist Agent =====================================, # IMPORTANT: We use output_pydantic=OptimizedResume to enforce structured output, Helper to print formatted sections with truncation., int, str

### Community 76 - "Community 76"
Cohesion: 0.29
Nodes (7): main(), print_section(), Example: Resume Extractor Agent ================================  OBJECTIVE: ---, # IMPORTANT: We use output_pydantic=Resume to enforce structured output., Helper to print formatted sections with truncation., int, str

### Community 77 - "Community 77"
Cohesion: 0.29
Nodes (8): AnalyzerEngine, _build_age_recognizer(), _build_date_of_birth_recognizer(), _get_analyzer(), Detect a date only when birth-context words are nearby, not any date., Detect an age written with an explicit suffix, e.g. '32 years old'., Lazily build and cache the Presidio analyzer (loads the spaCy model once)., PatternRecognizer

### Community 78 - "Community 78"
Cohesion: 0.25
Nodes (5): This __init__.py file serves two purposes:  1.  It marks the `data_models` dir, Config, Data Models for Resume Structure --------------------------------  This module d, Config, Data Models for Resume-Job Alignment Strategy ---------------------------------

### Community 79 - "Community 79"
Cohesion: 0.33
Nodes (6): main(), print_section(), Example: Gap Analysis Specialist Agent =======================================, Helper to print formatted sections with truncation., int, str

### Community 80 - "Community 80"
Cohesion: 0.33
Nodes (5): Shared resume-to-text rendering.  Why it lives in shared/: this generic resume->, Render each role's title, company, description, and achievement bullets., render_experience(), Experience, str

### Community 81 - "Community 81"
Cohesion: 0.47
Nodes (6): Ask the model to review resume_text against a rubric, as a ReviewResult.      Ar, Tag every comment with the calling engine's id (the model cannot know it)., request_review(), _stamp_engine_id(), ReviewResult, str

## Knowledge Gaps
- **31 isolated node(s):** `str`, `bool`, `str`, `setup-pre-commit.sh script`, `str` (+26 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Resume` connect `Community 9` to `Community 8`, `Community 10`, `Community 12`, `Community 13`, `Community 18`, `Community 19`, `Community 20`, `Community 23`, `Community 24`, `Community 25`, `Community 26`, `Community 28`, `Community 29`, `Community 30`, `Community 31`, `Community 34`, `Community 36`, `Community 37`, `Community 38`, `Community 45`, `Community 48`, `Community 50`, `Community 56`, `Community 67`, `Community 68`, `Community 71`, `Community 76`, `Community 78`, `Community 80`?**
  _High betweenness centrality (0.201) - this node is a cross-community bridge._
- **Why does `Experience` connect `Community 12` to `Community 2`, `Community 3`, `Community 6`, `Community 8`, `Community 9`, `Community 11`, `Community 13`, `Community 18`, `Community 20`, `Community 25`, `Community 34`, `Community 36`, `Community 38`, `Community 43`, `Community 45`, `Community 48`, `Community 50`, `Community 56`, `Community 59`, `Community 67`, `Community 68`, `Community 71`, `Community 78`, `Community 80`?**
  _High betweenness centrality (0.122) - this node is a cross-community bridge._
- **Why does `get_logger()` connect `Community 62` to `Community 0`, `Community 3`, `Community 4`, `Community 5`, `Community 6`, `Community 10`, `Community 11`, `Community 13`, `Community 15`, `Community 16`, `Community 17`, `Community 18`, `Community 19`, `Community 21`, `Community 22`, `Community 23`, `Community 24`, `Community 26`, `Community 27`, `Community 29`, `Community 30`, `Community 31`, `Community 35`, `Community 37`, `Community 45`, `Community 48`, `Community 51`, `Community 54`, `Community 55`, `Community 67`, `Community 70`, `Community 72`, `Community 74`, `Community 75`, `Community 76`, `Community 79`?**
  _High betweenness centrality (0.105) - this node is a cross-community bridge._
- **Are the 147 inferred relationships involving `Resume` (e.g. with `ATSValidationResult` and `KeywordDensityReport`) actually correct?**
  _`Resume` has 147 INFERRED edges - model-reasoned connections that need verification._
- **Are the 106 inferred relationships involving `JobDescription` (e.g. with `ATSValidationResult` and `KeywordDensityReport`) actually correct?**
  _`JobDescription` has 106 INFERRED edges - model-reasoned connections that need verification._
- **Are the 85 inferred relationships involving `Experience` (e.g. with `ATSValidationResult` and `KeywordDensityReport`) actually correct?**
  _`Experience` has 85 INFERRED edges - model-reasoned connections that need verification._
- **Are the 84 inferred relationships involving `AlignmentStrategy` (e.g. with `BulletDraft` and `IterativeExperienceOptimization`) actually correct?**
  _`AlignmentStrategy` has 84 INFERRED edges - model-reasoned connections that need verification._