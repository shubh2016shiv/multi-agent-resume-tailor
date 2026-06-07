"""Trigger the Resume Extractor agent live: PDF in, Resume out."""
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from crewai import Crew, Process, Task
from src.agents.resume_parser import create_resume_extractor_agent
from src.core.config import get_tasks_config
from src.data_models.resume import Resume

PDF_PATH = str(Path("sample_documents/5de9878acaf813644e37cbf8_5de98838caf81311687f9290.pdf").resolve())

agent = create_resume_extractor_agent()
print("Agent  :", agent.role)
print("Model  :", agent.llm.model)
print("Tools  :", [t.name for t in agent.tools])
print("PDF    :", PDF_PATH)

task_config = get_tasks_config().get("extract_resume_content_task", {})

task = Task(
    description=(
        f"Resume file path: {PDF_PATH}\n\n"
        f"{task_config.get('description', '')}"
    ),
    expected_output=task_config.get("expected_output", "A validated Resume object."),
    agent=agent,
    output_pydantic=Resume,
)

result = Crew(
    agents=[agent],
    tasks=[task],
    process=Process.sequential,
    verbose=True,
).kickoff()

resume = result.pydantic
print("\n=== EXTRACTED RESUME ===")
print("Name       :", resume.full_name)
print("Email      :", resume.email)
print("Experience :", len(resume.work_experience), "roles")
print("Skills     :", len(resume.skills), "skills")
print("CONTRACT   : valid Resume object ready for handoff")
