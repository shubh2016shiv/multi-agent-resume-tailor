"""Trigger the Job Description Analyst tool in isolation.

Pipeline:
  1. Load job description text from file
  2. Call analyze_job_description() tool → JobDescription
  3. Print structured summary

The tool function internally creates a CrewAI agent, runs it, and returns
the typed output — the same pattern LangGraph will use.
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.agents.job_description_analyser import analyze_job_description

# ── paths ─────────────────────────────────────────────────────────────────────

JD_PATH = str(Path("sample_documents/sample_job_description.txt").resolve())


# ── step 1: load job text ─────────────────────────────────────────────────────

print("=== STEP 1: Load Job Description ===")
job_text = Path(JD_PATH).read_text()
print(f"  File : {JD_PATH}")
print(f"  Chars: {len(job_text)}")


# ── step 2: invoke tool ───────────────────────────────────────────────────────

print("\n=== STEP 2: Run analyze_job_description() tool ===")
print("  (internally creates CrewAI agent + runs Crew)")

job = analyze_job_description(job_text)


# ── step 3: print results ─────────────────────────────────────────────────────

print("\n=== JOB DESCRIPTION ===")
print(f"  Title        : {job.job_title}")
print(f"  Company      : {job.company_name}")
print(f"  Level        : {job.job_level}")
print(f"  Requirements : {len(job.requirements)}")
print(f"  Must-haves   : {len(job.must_have_skills)}")
print(f"  Should-haves : {len(job.should_have_skills)}")
print(f"  Nice-to-haves: {len(job.nice_to_have_skills)}")
print(f"  ATS keywords : {len(job.ats_keywords)}")
print(f"  Summary      : {job.summary[:120] if job.summary else 'N/A'}...")
print(f"  CONTRACT     : valid JobDescription ready for downstream agents")
