"""Shared fixtures for tests/unit/formatters/.

Fixtures here are scoped to the formatter test tree.
Never put test functions in conftest.
"""

from datetime import date

import pytest

from src.agents.ats_optimizer.models import AtsOptimizedResume
from src.agents.professional_experience.models import (
    ExperienceRelevance,
    OptimizedExperienceSection,
)
from src.agents.professional_summary.models import ProfessionalSummary, SummaryDraft
from src.data_models.job import JobDescription, JobRequirement, SkillImportance
from src.data_models.resume import (
    Education,
    Experience,
    OptimizedSkillsSection,
    Resume,
    Skill,
)
from src.data_models.strategy import AlignmentStrategy, SkillGap, SkillMatch
from src.tools.contracts import (
    Confidence,
    Location,
    ReviewComment,
    ReviewResult,
    Section,
    Severity,
)


@pytest.fixture
def sample_resume() -> Resume:
    """A populated resume with enough signal for every formatter contract."""
    return Resume(
        full_name="Jane Doe",
        email="jane@example.com",
        phone_number="+1-555-0100",
        location="Remote",
        website_or_portfolio="https://example.com",
        professional_summary="Backend engineer with cloud and API experience.",
        work_experience=[
            Experience(
                experience_id="exp-1",
                job_title="Software Engineer",
                company_name="Tech Corp",
                start_date=date(2021, 1, 1),
                end_date=date(2024, 1, 1),
                is_current_position=False,
                location="Remote",
                description="Built backend services for customer-facing products.",
                achievements=[
                    "Improved API latency by 30%.",
                    "Built AWS deployment pipeline.",
                    "Partnered with product on launch readiness.",
                ],
                skills_used=["Python", "AWS", "REST APIs"],
            )
        ],
        skills=[
            Skill(
                skill_name="Python",
                category=None,
                proficiency_level="Expert",
                years_of_experience=5,
                justification=None,
                confidence_score=None,
            ),
            Skill(
                skill_name="AWS",
                category=None,
                proficiency_level="Advanced",
                years_of_experience=4,
                justification=None,
                confidence_score=None,
            ),
            Skill(
                skill_name="REST APIs",
                category=None,
                proficiency_level="Advanced",
                years_of_experience=5,
                justification=None,
                confidence_score=None,
            ),
        ],
        education=[
            Education(
                institution_name="State University",
                degree="BS",
                field_of_study="Computer Science",
                graduation_year=2020,
                gpa=None,
                honors=None,
            )
        ],
        certifications=["AWS Certified Developer"],
        languages=["English"],
    )


@pytest.fixture
def sample_job_description() -> JobDescription:
    """A job description with must-have, should-have, and nice-to-have requirements."""
    return JobDescription(
        job_title="Senior Backend Engineer",
        company_name="Cloud Co",
        location=None,
        summary="Build reliable backend systems and APIs.",
        full_text="Need Python, AWS, and API design experience.",
        requirements=[
            JobRequirement(requirement="Python", importance=SkillImportance.MUST_HAVE, years_required=3),
            JobRequirement(requirement="AWS", importance=SkillImportance.SHOULD_HAVE, years_required=None),
            JobRequirement(
                requirement="GraphQL",
                importance=SkillImportance.NICE_TO_HAVE,
                years_required=None,
            ),
        ],
        ats_keywords=["Python", "AWS", "API", "REST"],
    )


@pytest.fixture
def sample_alignment_strategy() -> AlignmentStrategy:
    """A strategy object with guidance for summary, experience, and skills stages."""
    return AlignmentStrategy(
        overall_fit_score=82.0,
        summary_of_strategy="Lead with backend and cloud impact.",
        identified_matches=[
            SkillMatch(
                resume_skill="Python",
                job_requirement="Python",
                match_score=95.0,
                justification="Direct match.",
            ),
            SkillMatch(
                resume_skill="AWS",
                job_requirement="AWS",
                match_score=88.0,
                justification="Direct match.",
            ),
        ],
        identified_gaps=[
            SkillGap(
                missing_skill="GraphQL",
                importance="must_have",
                suggestion="Only mention if supported by existing work.",
            )
        ],
        keywords_to_integrate=["Python", "AWS", "API"],
        professional_summary_guidance="Open with backend and cloud outcomes.",
        experience_guidance="Prioritize API and cloud achievements.",
        skills_guidance="Put Python and AWS first.",
    )


@pytest.fixture
def sample_professional_summary() -> ProfessionalSummary:
    """A summary response with a recommended draft."""
    return ProfessionalSummary(
        drafts=[
            SummaryDraft(
                version_name="v1",
                strategy_used="direct",
                evidence_used=(
                    "Python backend work supports Python and API keywords. "
                    "AWS deployment pipeline supports AWS keyword."
                ),
                content=(
                    "Senior backend engineer with Python and AWS experience delivering "
                    "reliable APIs, measurable performance gains, and cloud-ready systems "
                    "for product teams."
                ),
                score=90,
            ),
            SummaryDraft(
                version_name="v2",
                strategy_used="alternate",
                evidence_used="Secondary draft evidence.",
                content=(
                    "Backend engineer focused on cloud delivery, API reliability, and "
                    "product execution across cross-functional teams."
                ),
                score=80,
            ),
        ],
        recommended_version="v1",
    )


@pytest.fixture
def sample_optimized_experience(sample_resume: Resume) -> OptimizedExperienceSection:
    """An optimized experience section using the sample resume's single role."""
    return OptimizedExperienceSection(
        optimized_experiences=sample_resume.work_experience,
        optimization_notes="Reordered only.",
        keywords_integrated=[],
        relevance_scores=[
            ExperienceRelevance(company_name="Tech Corp", relevance_score=92.0)
        ],
    )


@pytest.fixture
def sample_optimized_skills(sample_resume: Resume) -> OptimizedSkillsSection:
    """An optimized skills section using the sample resume's current skills."""
    return OptimizedSkillsSection(
        optimized_skills=sample_resume.skills,
        optimization_notes="Prioritized core skills.",
        added_skills=[],
        removed_skills=[],
    )


@pytest.fixture
def sample_optimized_resume(sample_resume: Resume) -> AtsOptimizedResume:
    """An ATS-optimized resume wrapper around the sample resume."""
    return AtsOptimizedResume(
        final_resume=sample_resume,
        section_order=["Professional Summary", "Work Experience", "Skills", "Education"],
        optimization_summary="Assembled from upstream sections.",
        keyword_integration_notes="Used supported keywords only.",
        unresolved_issues=[],
    )


@pytest.fixture
def sample_match_report() -> ReviewResult:
    """A code-owned match report with one major finding."""
    return ReviewResult(
        comments=[
            ReviewComment(
                engine_id="job_matcher",
                message="GraphQL is not evidenced in the resume.",
                quoted_text="",
                location=Location(section=Section.SKILLS),
                severity=Severity.MAJOR,
                confidence=Confidence.HIGH,
                advice="Do not invent GraphQL experience.",
            )
        ],
        summary="Strong backend alignment with one important gap.",
        score=0.78,
    )
