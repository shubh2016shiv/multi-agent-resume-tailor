"""Unit tests for src/agents/quality_assessment/engines.py

Contracts under test:
  apply_quality_gate(report, threshold) — overwrites passed_quality_threshold with
                                          overall_quality_score >= threshold, regardless
                                          of what the agent set. Returns a new report.
  should_render_resume(report)          — returns report.passed_quality_threshold.
                                          Does NOT recompute — assumes gate was applied.
"""

from src.agents.quality_assessment.engines import (
    QUALITY_PASS_THRESHOLD,
    apply_quality_gate,
    should_render_resume,
)

# ── apply_quality_gate tests ──────────────────────────────────────────────────


class TestApplyQualityGate:
    """Tests for apply_quality_gate."""

    def test_apply_quality_gate_with_score_above_threshold_sets_passed_true(
        self, passing_quality_report
    ):
        """
        Contract: overall_quality_score >= threshold → passed_quality_threshold = True.
        passing_quality_report has score=85.0, threshold=80.0.
        Expected value: 85 >= 80 is True — derived from the stated threshold, not code.
        """
        result = apply_quality_gate(passing_quality_report)

        assert result.passed_quality_threshold is True

    def test_apply_quality_gate_with_score_below_threshold_sets_passed_false(
        self, failing_quality_report
    ):
        """
        Contract: overall_quality_score < threshold → passed_quality_threshold = False.
        failing_quality_report has score=72.0, threshold=80.0.
        Expected value: 72 < 80 is False — derived from the stated threshold, not code.
        """
        result = apply_quality_gate(failing_quality_report)

        assert result.passed_quality_threshold is False

    def test_apply_quality_gate_at_exact_threshold_sets_passed_true(self, passing_quality_report):
        """
        Contract: score exactly equal to threshold passes (>=, not >).
        Boundary case derived from the docstring: "overall_quality_score >= threshold".
        """
        report_at_boundary = passing_quality_report.model_copy(
            update={"overall_quality_score": QUALITY_PASS_THRESHOLD}
        )

        result = apply_quality_gate(report_at_boundary)

        assert result.passed_quality_threshold is True

    def test_apply_quality_gate_one_point_below_threshold_sets_passed_false(
        self, passing_quality_report
    ):
        """
        Contract: score one point below threshold fails.
        Boundary case: QUALITY_PASS_THRESHOLD - 0.1 must fail.
        """
        report_just_below = passing_quality_report.model_copy(
            update={"overall_quality_score": QUALITY_PASS_THRESHOLD - 0.1}
        )

        result = apply_quality_gate(report_just_below)

        assert result.passed_quality_threshold is False

    def test_apply_quality_gate_overwrites_agent_set_value_regardless_of_original(
        self, failing_quality_report
    ):
        """
        Contract: the agent's value of passed_quality_threshold is NOT trusted.
        failing_quality_report starts with passed_quality_threshold=True (intentionally wrong).
        apply_quality_gate must overwrite it to False based on score=72.0 < 80.0.
        """
        assert failing_quality_report.passed_quality_threshold is True  # pre-condition

        result = apply_quality_gate(failing_quality_report)

        assert result.passed_quality_threshold is False

    def test_apply_quality_gate_returns_new_report_not_mutating_original(
        self, passing_quality_report
    ):
        """
        Contract: returns a copy (model_copy) — the original report is not mutated.
        """
        original_value = passing_quality_report.passed_quality_threshold

        result = apply_quality_gate(passing_quality_report)

        assert passing_quality_report.passed_quality_threshold == original_value
        assert result is not passing_quality_report

    def test_apply_quality_gate_accepts_custom_threshold(self, passing_quality_report):
        """
        Contract: custom threshold overrides the default QUALITY_PASS_THRESHOLD.
        passing_quality_report has score=85.0; with threshold=90.0 it should fail.
        """
        result = apply_quality_gate(passing_quality_report, threshold=90.0)

        assert result.passed_quality_threshold is False


# ── should_render_resume tests ────────────────────────────────────────────────


class TestShouldRenderResume:
    """Tests for should_render_resume."""

    def test_should_render_resume_with_passed_report_returns_true(self, passing_quality_report):
        """
        Contract: returns True when passed_quality_threshold is True.
        """
        gated_report = apply_quality_gate(passing_quality_report)

        assert should_render_resume(gated_report) is True

    def test_should_render_resume_with_failed_report_returns_false(self, failing_quality_report):
        """
        Contract: returns False when passed_quality_threshold is False.
        """
        gated_report = apply_quality_gate(failing_quality_report)

        assert should_render_resume(gated_report) is False
