"""Unit tests for src/tools/agent_tools/ingestion_tools.py

Tests verify that tool wrappers correctly transform inputs through underlying
engines and return appropriately formatted outputs for agents.
"""

# Tool functions are wrapped by @tool decorator, so we extract the underlying
# callback function for testing
from src.tools.agent_tools import ingestion_tools


def _call_tool(tool_obj, *args, **kwargs):
    """Helper to call a @tool-wrapped function via its underlying func."""
    return tool_obj.func(*args, **kwargs)


class TestConvertResumeDocumentToMarkdown:
    """Tests for convert_resume_document_to_markdown tool."""

    def test_convert_resume_document_to_markdown_calls_underlying_engine_and_returns_markdown(
        self, mock_convert_document_to_markdown, sample_markdown
    ):
        """
        Contract: Tool returns Markdown string from document conversion engine.
        Mocking convert_document_to_markdown because it is disk I/O (file reading).
        Everything else here uses the real implementation.
        """
        # Arrange
        mock_convert_document_to_markdown.return_value = sample_markdown
        file_path = "/path/to/resume.pdf"

        # Act
        result = _call_tool(ingestion_tools.convert_resume_document_to_markdown, file_path)

        # Assert
        assert result == sample_markdown
        mock_convert_document_to_markdown.assert_called_once_with(file_path)

    def test_convert_resume_document_to_markdown_passes_file_path_to_engine(
        self, mock_convert_document_to_markdown
    ):
        """Verify tool correctly forwards file path to underlying engine."""
        # Arrange
        file_path = "/tmp/test_resume.docx"
        mock_convert_document_to_markdown.return_value = "# Test"

        # Act
        _call_tool(ingestion_tools.convert_resume_document_to_markdown, file_path)

        # Assert
        mock_convert_document_to_markdown.assert_called_once_with(file_path)

    def test_convert_resume_document_to_markdown_handles_empty_markdown_output(
        self, mock_convert_document_to_markdown
    ):
        """Tool should handle empty Markdown (e.g., corrupted or blank document)."""
        # Arrange
        mock_convert_document_to_markdown.return_value = ""

        # Act
        result = _call_tool(
            ingestion_tools.convert_resume_document_to_markdown, "/path/to/empty.pdf"
        )

        # Assert
        assert result == ""


class TestRedactPiiFromResumeMarkdown:
    """Tests for redact_pii_from_resume_markdown tool."""

    def test_redact_pii_from_resume_markdown_with_flag_enabled_redacts_and_saves_mapping(
        self,
        mock_get_config,
        mock_get_current_run_id,
        mock_redact_pii,
        mock_save_pii_mapping,
    ):
        """
        Contract: When PII redaction is enabled, tool redacts Markdown
        and saves placeholder mapping to storage.
        Mocking redact_pii because it uses pattern matching (third-party logic).
        Mocking get_config/get_current_run_id because they read runtime state.
        Everything else here uses the real implementation.
        """
        # Arrange
        mock_get_config.return_value.feature_flags.enable_pii_redaction = True
        markdown = "Jane Doe, jane@example.com, 555-1234"
        redacted = "[REDACTED_NAME], [REDACTED_EMAIL], [REDACTED_PHONE]"
        mapping = {
            "[REDACTED_NAME]": "Jane Doe",
            "[REDACTED_EMAIL]": "jane@example.com",
        }
        mock_redact_pii.return_value = (redacted, mapping)

        # Act
        result = _call_tool(ingestion_tools.redact_pii_from_resume_markdown, markdown)

        # Assert
        assert result == redacted
        mock_save_pii_mapping.assert_called_once_with("test-run-id-12345", mapping)

    def test_redact_pii_from_resume_markdown_with_flag_disabled_returns_unmodified(
        self, mock_get_config, mock_redact_pii, mock_save_pii_mapping
    ):
        """
        Contract: When PII redaction is disabled by feature flag,
        tool returns Markdown unchanged without calling redaction engine.
        """
        # Arrange
        mock_get_config.return_value.feature_flags.enable_pii_redaction = False
        markdown = "Jane Doe, jane@example.com"

        # Act
        result = _call_tool(ingestion_tools.redact_pii_from_resume_markdown, markdown)

        # Assert
        assert result == markdown
        mock_redact_pii.assert_not_called()
        mock_save_pii_mapping.assert_not_called()

    def test_redact_pii_from_resume_markdown_with_empty_mapping_still_saves(
        self, mock_get_config, mock_get_current_run_id, mock_redact_pii, mock_save_pii_mapping
    ):
        """Tool should handle redaction that produces empty mapping."""
        # Arrange
        mock_get_config.return_value.feature_flags.enable_pii_redaction = True
        markdown = "No PII here"
        mock_redact_pii.return_value = (markdown, {})

        # Act
        result = _call_tool(ingestion_tools.redact_pii_from_resume_markdown, markdown)

        # Assert
        assert result == markdown
        mock_save_pii_mapping.assert_called_once_with("test-run-id-12345", {})


class TestExtractStructuredResumeFromMarkdown:
    """Tests for extract_structured_resume_from_markdown tool."""

    def test_extract_structured_resume_from_markdown_returns_resume_json(
        self, mock_extract_resume, sample_resume_json
    ):
        """
        Contract: Tool extracts Resume from Markdown and returns JSON string.
        Mocking extract_resume because it contains LLM calls (not pure logic).
        Everything else here uses the real implementation.
        """
        # Arrange
        markdown = "# Jane Doe\nExperienced engineer."

        # Act
        result = _call_tool(ingestion_tools.extract_structured_resume_from_markdown, markdown)

        # Assert
        assert result == sample_resume_json
        mock_extract_resume.assert_called_once_with(markdown)

    def test_extract_structured_resume_from_markdown_with_pii_redaction_enabled_asserts_redaction(
        self,
        mock_get_config,
        mock_get_current_run_id,
        mock_extract_resume,
        mock_assert_extraction_input_redacted,
        sample_resume_json,
    ):
        """
        Contract: When PII redaction is enabled, tool verifies that input
        Markdown has been redacted before processing.
        Mocking assert_extraction_input_redacted because it reads state from store.
        Everything else here uses the real implementation.
        """
        # Arrange
        mock_get_config.return_value.feature_flags.enable_pii_redaction = True
        redacted_markdown = "[REDACTED_NAME] is an engineer."

        # Act
        result = _call_tool(
            ingestion_tools.extract_structured_resume_from_markdown, redacted_markdown
        )

        # Assert
        assert result == sample_resume_json
        mock_assert_extraction_input_redacted.assert_called_once_with(
            "test-run-id-12345", redacted_markdown
        )

    def test_extract_structured_resume_from_markdown_with_pii_redaction_disabled_skips_assertion(
        self,
        mock_get_config,
        mock_extract_resume,
        mock_assert_extraction_input_redacted,
        sample_resume_json,
    ):
        """
        Contract: When PII redaction is disabled, tool skips redaction assertion.
        """
        # Arrange
        mock_get_config.return_value.feature_flags.enable_pii_redaction = False
        markdown = "Jane Doe is an engineer."

        # Act
        result = _call_tool(ingestion_tools.extract_structured_resume_from_markdown, markdown)

        # Assert
        assert result == sample_resume_json
        mock_assert_extraction_input_redacted.assert_not_called()


class TestCheckResumeMarkdownQuality:
    """Tests for check_resume_markdown_quality tool."""

    def test_check_resume_markdown_quality_returns_formatted_review_result(
        self, mock_audit_extraction_quality, sample_markdown
    ):
        """
        Contract: Tool formats extraction quality audit result into plain text
        that agents can read.
        Mocking audit_extraction_quality because it contains analysis logic.
        Everything else here uses the real implementation.
        """
        # Arrange
        result_from_engine = mock_audit_extraction_quality.return_value

        # Act
        result = _call_tool(ingestion_tools.check_resume_markdown_quality, sample_markdown)

        # Assert
        # Verify the tool renders content from the engine's result
        assert isinstance(result, str)
        assert "=== Resume Markdown Quality ===" in result
        assert result_from_engine.summary in result
        assert f"{result_from_engine.score:.2f}" in result
        mock_audit_extraction_quality.assert_called_once_with(sample_markdown)

    def test_check_resume_markdown_quality_includes_summary_in_output(
        self, mock_audit_extraction_quality, sample_markdown
    ):
        """Tool output should include summary from quality audit."""
        # Arrange
        mock_audit_extraction_quality.return_value.summary = "Markdown is clean and well-formatted"

        # Act
        result = _call_tool(ingestion_tools.check_resume_markdown_quality, sample_markdown)

        # Assert
        assert "Markdown is clean and well-formatted" in result

    def test_check_resume_markdown_quality_includes_score_when_present(
        self, mock_audit_extraction_quality, sample_markdown
    ):
        """Tool should include quality score when audit provides one."""
        # Arrange
        mock_audit_extraction_quality.return_value.score = 0.92

        # Act
        result = _call_tool(ingestion_tools.check_resume_markdown_quality, sample_markdown)

        # Assert
        assert "Score:" in result
        assert "0.92" in result

    def test_check_resume_markdown_quality_handles_no_comments(
        self, mock_audit_extraction_quality, sample_markdown
    ):
        """Tool should gracefully handle audit with no issues."""
        # Arrange
        mock_audit_extraction_quality.return_value.comments = []
        mock_audit_extraction_quality.return_value.summary = "No issues found"

        # Act
        result = _call_tool(ingestion_tools.check_resume_markdown_quality, sample_markdown)

        # Assert
        assert "No issues found" in result
