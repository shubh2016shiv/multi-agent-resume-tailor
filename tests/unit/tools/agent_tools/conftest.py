"""Fixtures for agent_tools tests."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.data_models.resume import Resume
from src.tools.contracts import ReviewResult


@pytest.fixture
def sample_resume_json():
    """A minimal valid Resume JSON string."""
    resume_dict = {
        "full_name": "Jane Doe",
        "email": "jane@example.com",
        "professional_summary": "Experienced software engineer",
        "work_experience": [],
        "skills": [{"skill_name": "Python"}],
        "education": [],
        "certifications": [],
    }
    return json.dumps(resume_dict)


@pytest.fixture
def valid_resume(sample_resume_json) -> Resume:
    """A valid Resume object from sample JSON."""
    return Resume.model_validate_json(sample_resume_json)


@pytest.fixture
def sample_markdown():
    """Sample Markdown content (as if from document conversion)."""
    return """
# Jane Doe

## Professional Summary
Experienced software engineer with 10 years of experience.

## Experience
- Senior Engineer at TechCorp (2020-present)
- Engineer at StartupInc (2015-2020)

## Skills
- Python
- JavaScript
- AWS
"""


@pytest.fixture
def mock_get_config():
    """Mock get_config() to return a config with feature flags."""
    with patch("src.tools.agent_tools.ingestion_tools.get_config") as mock:
        config = MagicMock()
        config.feature_flags.enable_pii_redaction = True
        mock.return_value = config
        yield mock


@pytest.fixture
def mock_get_current_run_id():
    """Mock get_current_run_id() to return a test run ID."""
    with patch("src.tools.agent_tools.ingestion_tools.get_current_run_id") as mock:
        mock.return_value = "test-run-id-12345"
        yield mock


@pytest.fixture
def mock_save_pii_mapping():
    """Mock save_pii_mapping() in PII mapping store."""
    with patch("src.tools.agent_tools.ingestion_tools.save_pii_mapping") as mock:
        mock.return_value = None
        yield mock


@pytest.fixture
def mock_assert_extraction_input_redacted():
    """Mock assert_extraction_input_redacted() for PII assertion."""
    with patch("src.tools.agent_tools.ingestion_tools.assert_extraction_input_redacted") as mock:
        mock.return_value = None
        yield mock


@pytest.fixture
def mock_convert_document_to_markdown():
    """Mock the underlying document conversion engine."""
    with patch("src.tools.agent_tools.ingestion_tools.convert_document_to_markdown") as mock:
        # By default, return sample markdown
        mock.return_value = """
# Jane Doe
## Professional Summary
Experienced engineer.
## Skills
- Python
"""
        yield mock


@pytest.fixture
def mock_redact_pii():
    """Mock the PII redaction engine."""
    with patch("src.tools.agent_tools.ingestion_tools.redact_pii") as mock:
        # Returns (redacted_markdown, placeholder_mapping)
        redacted = "[REDACTED_NAME] is an experienced engineer."
        mapping = {"[REDACTED_NAME]": "Jane Doe"}
        mock.return_value = (redacted, mapping)
        yield mock


@pytest.fixture
def mock_extract_resume(sample_resume_json):
    """Mock the resume extraction engine."""
    with patch("src.tools.agent_tools.ingestion_tools.extract_resume") as mock:
        # Create a mock Resume object that returns our sample JSON when model_dump_json is called
        mock_resume = MagicMock()
        mock_resume.model_dump_json.return_value = sample_resume_json
        mock.return_value = mock_resume
        yield mock


@pytest.fixture
def mock_audit_extraction_quality():
    """Mock the extraction quality auditor."""
    with patch("src.tools.agent_tools.ingestion_tools.audit_extraction_quality") as mock:
        result = ReviewResult(
            comments=[],
            summary="Markdown quality is good",
            score=0.95,
        )
        mock.return_value = result
        yield mock


# Review tools fixtures


@pytest.fixture
def mock_get_config_review():
    """Mock get_config() for review tools (different module path)."""
    with patch("src.tools.agent_tools.resume_review_tools.get_config") as mock:
        config = MagicMock()
        config.feature_flags.enable_pii_redaction = True
        mock.return_value = config
        yield mock


@pytest.fixture
def mock_audit_summary_text():
    """Mock audit_summary_text engine."""
    with patch("src.tools.agent_tools.resume_review_tools.audit_summary_text") as mock:
        result = ReviewResult(
            comments=[],
            summary="Summary is well-written",
            score=0.85,
        )
        mock.return_value = result
        yield mock


@pytest.fixture
def mock_validate_skills_evidence():
    """Mock validate_skills_evidence engine."""
    with patch("src.tools.agent_tools.resume_review_tools.validate_skills_evidence") as mock:
        result = ReviewResult(
            comments=[],
            summary="All skills are well-supported",
            score=0.90,
        )
        mock.return_value = result
        yield mock


@pytest.fixture
def mock_detect_claim_inflation():
    """Mock detect_claim_inflation engine."""
    with patch("src.tools.agent_tools.resume_review_tools.detect_claim_inflation") as mock:
        result = ReviewResult(comments=[], summary="No claim inflation detected")
        mock.return_value = result
        yield mock


@pytest.fixture
def mock_detect_rewrite_drift():
    """Mock detect_rewrite_drift engine."""
    with patch("src.tools.agent_tools.resume_review_tools.detect_rewrite_drift") as mock:
        result = ReviewResult(comments=[], summary="No rewrite drift detected")
        mock.return_value = result
        yield mock


@pytest.fixture
def mock_audit_ats_formatting():
    """Mock audit_ats_formatting engine."""
    with patch("src.tools.agent_tools.resume_review_tools.audit_ats_formatting") as mock:
        result = ReviewResult(
            comments=[],
            summary="ATS formatting is compatible",
            score=0.95,
        )
        mock.return_value = result
        yield mock


@pytest.fixture
def mock_audit_section_headers():
    """Mock audit_section_headers engine."""
    with patch("src.tools.agent_tools.resume_review_tools.audit_section_headers") as mock:
        result = ReviewResult(
            comments=[],
            summary="Section headers are ATS-compatible",
            score=0.95,
        )
        mock.return_value = result
        yield mock


@pytest.fixture
def mock_analyze_keyword_coverage():
    """Mock analyze_keyword_coverage engine."""
    with patch("src.tools.agent_tools.resume_review_tools.analyze_keyword_coverage") as mock:
        result = ReviewResult(
            comments=[],
            summary="80% keyword coverage",
            score=0.80,
        )
        mock.return_value = result
        yield mock
