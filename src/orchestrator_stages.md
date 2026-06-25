INPUT: resume PDF path + JD path

STAGE 1 — PARALLEL (independent, no shared state)
  Resume Extractor  ->  Resume
  Job Analyzer      ->  JobDescription

STAGE 2 — SEQUENTIAL (needs both above)
  Gap Analysis (Resume + JobDescription)  ->  AlignmentStrategy
                                               (fit score, matches, gaps,
                                                keywords, guidance per agent)

STAGE 3 — PARALLEL (each only needs Resume + Job + Strategy)
  Summary Writer        ->  ProfessionalSummary
  Experience Optimizer  ->  OptimizedExperienceSection
  Skills Optimizer      ->  OptimizedSkillsSection

STAGE 4 — SEQUENTIAL (needs all three above)
  ATS Assembly (Summary + Experience + Skills + Resume + Job)  ->  OptimizedResume

STAGE 5 — SEQUENTIAL (needs assembled resume)
  Quality Assurance (OptimizedResume + original Resume + Job)  ->  QualityReport

OUTPUT: OrchestrationResult
