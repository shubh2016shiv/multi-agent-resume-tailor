"""Unit tests for src/tools/agent_tools/resume_review_tools.py

Tests verify that review tool wrappers correctly parse resume data,
invoke underlying analysis engines, and return formatted results for agents.
"""

import json

from src.tools.agent_tools import resume_review_tools
from src.tools.contracts import Confidence, Location, ReviewComment, ReviewResult, Section, Severity


def _call_tool(tool_obj, *args, **kwargs):
    """Helper to call a @tool-wrapped function via its underlying func."""
    return tool_obj.func(*args, **kwargs)


class TestRenderReviewResult:
    """Tests for render_review_result helper."""

    def test_render_review_result_includes_title(self):
        """Helper should include the provided title in output."""
        # Arrange
        result = ReviewResult(comments=[], summary="Test summary")
        title = "Test Audit"

        # Act
        output = resume_review_tools.render_review_result(result, title)

        # Assert
        assert "=== Test Audit ===" in output

    def test_render_review_result_includes_summary(self):
        """Helper should include summary from review result."""
        # Arrange
        result = ReviewResult(comments=[], summary="Well-written and comprehensive")

        # Act
        output = resume_review_tools.render_review_result(result, "Summary Quality")

        # Assert
        assert "Well-written and comprehensive" in output

    def test_render_review_result_includes_score_when_present(self):
        """Helper should format and include score when provided."""
        # Arrange
        result = ReviewResult(comments=[], summary="Good", score=0.85)

        # Act
        output = resume_review_tools.render_review_result(result, "Test")

        # Assert
        assert "Score: 0.85" in output

    def test_render_review_result_omits_score_when_none(self):
        """Helper should not include score line when score is None."""
        # Arrange
        result = ReviewResult(comments=[], summary="Good", score=None)

        # Act
        output = resume_review_tools.render_review_result(result, "Test")

        # Assert
        assert "Score:" not in output

    def test_render_review_result_formats_comments_with_severity_and_confidence(self):
        """Helper should format each comment with severity and confidence."""
        # Arrange
        comment = ReviewComment(
            engine_id="test_engine",
            message="Summary is too generic",
            quoted_text="Software engineer",
            location=Location(section=Section.SUMMARY),
            severity=Severity.MINOR,
            confidence=Confidence.HIGH,
            advice="Add specific accomplishments",
        )
        result = ReviewResult(comments=[comment], summary="Issues found")

        # Act
        output = resume_review_tools.render_review_result(result, "Test")

        # Assert
        assert "[minor/high]" in output
        assert "Summary is too generic" in output
        assert "advice: Add specific accomplishments" in output

    def test_render_review_result_includes_proposed_rewrite_when_present(self):
        """Helper should include proposed rewrite if provided."""
        # Arrange
        comment = ReviewComment(
            engine_id="test_engine",
            message="Needs improvement",
            quoted_text="Old text",
            location=Location(section=Section.SUMMARY),
            severity=Severity.MINOR,
            confidence=Confidence.HIGH,
            advice="Rewrite it",
            proposed_rewrite="New improved text",
        )
        result = ReviewResult(comments=[comment])

        # Act
        output = resume_review_tools.render_review_result(result, "Test")

        # Assert
        assert "rewrite: New improved text" in output

    def test_render_review_result_handles_empty_comments(self):
        """Helper should indicate no issues when comment list is empty."""
        # Arrange
        result = ReviewResult(comments=[], summary="")

        # Act
        output = resume_review_tools.render_review_result(result, "Test")

        # Assert
        assert "No issues found" in output


class TestMergeReviewResults:
    """Tests for merge_review_results helper."""

    def test_merge_review_results_combines_all_comments(self):
        """Helper should merge comments from all result objects."""
        # Arrange
        comment1 = ReviewComment(
            engine_id="engine1",
            message="Issue 1",
            quoted_text="text1",
            location=Location(section=Section.SUMMARY),
            severity=Severity.MINOR,
            confidence=Confidence.HIGH,
            advice="Fix it",
        )
        comment2 = ReviewComment(
            engine_id="engine2",
            message="Issue 2",
            quoted_text="text2",
            location=Location(section=Section.SKILLS),
            severity=Severity.MAJOR,
            confidence=Confidence.MEDIUM,
            advice="Fix that too",
        )
        result1 = ReviewResult(comments=[comment1], summary="First audit")
        result2 = ReviewResult(comments=[comment2], summary="Second audit")

        # Act
        merged = resume_review_tools.merge_review_results([result1, result2])

        # Assert
        assert len(merged.comments) == 2
        assert comment1 in merged.comments
        assert comment2 in merged.comments

    def test_merge_review_results_combines_summaries_with_semicolon(self):
        """Helper should concatenate non-empty summaries with semicolons."""
        # Arrange
        result1 = ReviewResult(comments=[], summary="First finding")
        result2 = ReviewResult(comments=[], summary="Second finding")

        # Act
        merged = resume_review_tools.merge_review_results([result1, result2])

        # Assert
        assert merged.summary == "First finding; Second finding"

    def test_merge_review_results_uses_first_non_none_score(self):
        """Helper should use the first result with a score."""
        # Arrange
        result1 = ReviewResult(comments=[], summary="", score=None)
        result2 = ReviewResult(comments=[], summary="", score=0.85)
        result3 = ReviewResult(comments=[], summary="", score=0.90)

        # Act
        merged = resume_review_tools.merge_review_results([result1, result2, result3])

        # Assert
        assert merged.score == 0.85

    def test_merge_review_results_handles_all_none_scores(self):
        """Helper should result in None score when all inputs are None."""
        # Arrange
        result1 = ReviewResult(comments=[], summary="", score=None)
        result2 = ReviewResult(comments=[], summary="", score=None)

        # Act
        merged = resume_review_tools.merge_review_results([result1, result2])

        # Assert
        assert merged.score is None


class TestAuditSummary:
    """Tests for audit_summary tool."""

    def test_audit_summary_parses_resume_json_and_invokes_engine(
        self, mock_audit_summary_quality, sample_resume_json
    ):
        """
        Contract: Tool parses Resume JSON and invokes audit engine,
        returning formatted result for agent consumption.
        Mocking audit_summary_quality because it contains analysis logic.
        Everything else here uses the real implementation.
        """
        # Arrange
        mock_audit_summary_quality.return_value = ReviewResult(
            comments=[], summary="Summary is well-written"
        )

        # Act
        result = _call_tool(resume_review_tools.audit_summary, sample_resume_json)

        # Assert
        assert isinstance(result, str)
        assert "=== Summary Quality ===" in result
        mock_audit_summary_quality.assert_called_once()

    def test_audit_summary_with_invalid_json_returns_error_message(self):
        """Tool should gracefully handle malformed JSON input."""
        # Arrange
        invalid_json = "not valid json {{"

        # Act
        result = _call_tool(resume_review_tools.audit_summary, invalid_json)

        # Assert
        assert "Error: could not parse resume JSON" in result

    def test_audit_summary_with_missing_required_field_returns_validation_error(self):
        """Tool should handle Resume validation errors gracefully."""
        # Arrange
        incomplete_json = json.dumps(
            {
                "candidate_name": "Jane Doe"
                # missing required fields
            }
        )

        # Act
        result = _call_tool(resume_review_tools.audit_summary, incomplete_json)

        # Assert
        assert "Error: could not parse resume JSON" in result
        assert "validation error" in result


class TestCheckSkillsEvidence:
    """Tests for check_skills_evidence tool."""

    def test_check_skills_evidence_validates_skills_and_returns_formatted_result(
        self, mock_validate_skills_evidence, sample_resume_json
    ):
        """
        Contract: Tool validates that skills in resume are supported by evidence.
        Mocking validate_skills_evidence because it contains analysis logic.
        Everything else here uses the real implementation.
        """
        # Arrange
        mock_validate_skills_evidence.return_value = ReviewResult(
            comments=[], summary="All skills are well-supported"
        )

        # Act
        result = _call_tool(resume_review_tools.check_skills_evidence, sample_resume_json)

        # Assert
        assert isinstance(result, str)
        assert "=== Skills Evidence ===" in result
        assert "All skills are well-supported" in result

    def test_check_skills_evidence_with_invalid_json_returns_error(self):
        """Tool should handle invalid JSON gracefully."""
        # Arrange
        bad_json = "{"

        # Act
        result = _call_tool(resume_review_tools.check_skills_evidence, bad_json)

        # Assert
        assert "Error: could not parse resume JSON" in result


class TestAuditTruthfulness:
    """Tests for audit_truthfulness tool."""

    def test_audit_truthfulness_compares_resumes_for_claim_drift(
        self,
        mock_detect_claim_inflation,
        mock_detect_rewrite_drift,
        sample_resume_json,
    ):
        """
        Contract: Tool compares original and revised resumes, detecting
        invented or drifted claims. Result merges both analyses.
        Mocking engines because they contain analysis logic.
        Everything else here uses the real implementation.
        """
        # Arrange
        mock_detect_claim_inflation.return_value = ReviewResult(
            comments=[], summary="No inflation detected"
        )
        mock_detect_rewrite_drift.return_value = ReviewResult(
            comments=[], summary="No drift detected"
        )

        # Act
        result = _call_tool(
            resume_review_tools.audit_truthfulness, sample_resume_json, sample_resume_json
        )

        # Assert
        assert isinstance(result, str)
        assert "=== Truthfulness ===" in result
        assert "No inflation detected" in result
        assert "No drift detected" in result

    def test_audit_truthfulness_with_invalid_original_json_returns_error(self, sample_resume_json):
        """Tool should handle invalid original resume JSON."""
        # Arrange
        bad_json = "{invalid"

        # Act
        result = _call_tool(resume_review_tools.audit_truthfulness, bad_json, sample_resume_json)

        # Assert
        assert "Error: could not parse resume JSON" in result

    def test_audit_truthfulness_with_invalid_revised_json_returns_error(self, sample_resume_json):
        """Tool should handle invalid revised resume JSON."""
        # Arrange
        bad_json = "{invalid"

        # Act
        result = _call_tool(resume_review_tools.audit_truthfulness, sample_resume_json, bad_json)

        # Assert
        assert "Error: could not parse resume JSON" in result


class TestValidateAtsCompliance:
    """Tests for validate_ats_compliance tool."""

    def test_validate_ats_compliance_checks_formatting_and_headers(
        self, mock_audit_ats_formatting, mock_audit_section_headers
    ):
        """
        Contract: Tool checks both ATS formatting and section headers,
        merging results for agent.
        Mocking audit engines because they contain analysis logic.
        Everything else here uses the real implementation.
        """
        # Arrange
        resume_text = "# Jane Doe\n## Experience\nSenior Engineer"
        mock_audit_ats_formatting.return_value = ReviewResult(
            comments=[], summary="Formatting is compatible"
        )
        mock_audit_section_headers.return_value = ReviewResult(
            comments=[], summary="Headers are compatible"
        )

        # Act
        result = _call_tool(resume_review_tools.validate_ats_compliance, resume_text)

        # Assert
        assert isinstance(result, str)
        assert "=== ATS Compliance ===" in result
        assert "Formatting is compatible" in result
        assert "Headers are compatible" in result

    def test_validate_ats_compliance_with_empty_text(
        self, mock_audit_ats_formatting, mock_audit_section_headers
    ):
        """Tool should handle empty resume text."""
        # Arrange
        mock_audit_ats_formatting.return_value = ReviewResult(comments=[])
        mock_audit_section_headers.return_value = ReviewResult(comments=[])

        # Act
        result = _call_tool(resume_review_tools.validate_ats_compliance, "")

        # Assert
        assert isinstance(result, str)
        assert "=== ATS Compliance ===" in result


class TestAnalyzeJdKeywordCoverage:
    """Tests for analyze_jd_keyword_coverage tool."""

    def test_analyze_jd_keyword_coverage_splits_keywords_and_analyzes(
        self, mock_analyze_keyword_coverage
    ):
        """
        Contract: Tool parses comma-separated keywords, analyzes coverage,
        and returns formatted result.
        Mocking analyze_keyword_coverage because it contains analysis logic.
        Everything else here uses the real implementation.
        """
        # Arrange
        resume_text = "Python, AWS, Kubernetes experience"
        keywords_csv = "Python, AWS, Kubernetes, Docker"
        mock_analyze_keyword_coverage.return_value = ReviewResult(
            comments=[], summary="75% keyword coverage"
        )

        # Act
        result = _call_tool(
            resume_review_tools.analyze_jd_keyword_coverage, resume_text, keywords_csv
        )

        # Assert
        assert isinstance(result, str)
        assert "=== Keyword Coverage ===" in result
        assert "75% keyword coverage" in result
        # Verify it split keywords correctly
        mock_analyze_keyword_coverage.assert_called_once()
        call_args = mock_analyze_keyword_coverage.call_args
        assert "Python" in call_args[0][1]
        assert "Docker" in call_args[0][1]

    def test_analyze_jd_keyword_coverage_strips_whitespace_from_keywords(
        self, mock_analyze_keyword_coverage
    ):
        """Tool should trim whitespace from keywords."""
        # Arrange
        resume_text = "Some text"
        keywords_csv = "  Python  ,  AWS  , Kubernetes  "
        mock_analyze_keyword_coverage.return_value = ReviewResult(comments=[])

        # Act
        _call_tool(resume_review_tools.analyze_jd_keyword_coverage, resume_text, keywords_csv)

        # Assert
        call_keywords = mock_analyze_keyword_coverage.call_args[0][1]
        # Verify no extra whitespace
        assert "  Python  " not in call_keywords
        assert "Python" in call_keywords

    def test_analyze_jd_keyword_coverage_ignores_empty_keywords(
        self, mock_analyze_keyword_coverage
    ):
        """Tool should skip empty entries from split."""
        # Arrange
        resume_text = "Test"
        keywords_csv = "Python, , AWS, , Docker"
        mock_analyze_keyword_coverage.return_value = ReviewResult(comments=[])

        # Act
        _call_tool(resume_review_tools.analyze_jd_keyword_coverage, resume_text, keywords_csv)

        # Assert
        call_keywords = mock_analyze_keyword_coverage.call_args[0][1]
        # Should have exactly 3 keywords (no empty ones)
        assert len(call_keywords) == 3
        assert "" not in call_keywords

    def test_analyze_jd_keyword_coverage_with_single_keyword(self, mock_analyze_keyword_coverage):
        """Tool should handle single keyword without comma."""
        # Arrange
        resume_text = "Python experience"
        keywords_csv = "Python"
        mock_analyze_keyword_coverage.return_value = ReviewResult(comments=[])

        # Act
        result = _call_tool(
            resume_review_tools.analyze_jd_keyword_coverage, resume_text, keywords_csv
        )

        # Assert
        assert isinstance(result, str)
        mock_analyze_keyword_coverage.assert_called_once()
        call_keywords = mock_analyze_keyword_coverage.call_args[0][1]
        assert call_keywords == ["Python"]
