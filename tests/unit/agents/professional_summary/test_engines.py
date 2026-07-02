"""Unit tests for src/agents/professional_summary/engines.py

Contracts under test:
  check_summary_quality(summary, strategy) — scores each draft in a ProfessionalSummary
                                              for word count and supported-role relevance.
                                              Returns per-draft evaluations.
  analyze_keyword_integration(text, keywords) — measures how many required keywords appear
                                                in a summary text. Returns counts and rate.
"""

from src.agents.professional_summary.engines import (
    analyze_keyword_integration,
    check_summary_quality,
)
from src.agents.professional_summary.models import ProfessionalSummary, SummaryDraft
from src.data_models.strategy import AlignmentStrategy

# ── helpers ───────────────────────────────────────────────────────────────────


def _make_draft(content: str, version_name: str = "test-draft") -> SummaryDraft:
    """Build a SummaryDraft with the given content."""
    return SummaryDraft(
        version_name=version_name,
        strategy_used="Test strategy.",
        evidence_used="Test evidence: resume fact -> keyword.",
        content=content,
        score=80,
    )


def _make_summary(draft: SummaryDraft) -> ProfessionalSummary:
    return ProfessionalSummary(
        drafts=[draft],
        recommended_version=draft.version_name,
    )


# A good draft: 80–110 words, thesis-led, contains supported role vocabulary, no clichés.
_GOOD_DRAFT_CONTENT = (
    "Platform engineer trusted to steady production Python systems when cloud delivery starts to fray. "
    "Built AWS and CI/CD workflows that kept data pipelines reliable under real-time operational pressure, "
    "and translated brittle release paths into repeatable runtime practices for product teams. "
    "Comfortable moving between infrastructure decisions, incident debugging, release planning, and engineering handoffs, "
    "bringing delivery judgment to environments where uptime and change safety matter. "
    "Known for simplifying failure-prone systems before they become operational drag, "
    "especially when ownership boundaries and release habits are still maturing."
)

# ── check_summary_quality tests ───────────────────────────────────────────────


class TestCheckSummaryQuality:
    """Tests for check_summary_quality."""

    def test_check_summary_quality_with_good_draft_returns_good_or_excellent_quality(
        self, complete_alignment_strategy
    ):
        """
        Contract: a thesis-led draft with supported role vocabulary and no clichés
        scores at least 'good' (>=75).
        Expected value derived from the quality band contract: >=75 → 'good' or 'excellent'.
        """
        draft = _make_draft(_GOOD_DRAFT_CONTENT)
        summary = _make_summary(draft)

        result = check_summary_quality(summary, complete_alignment_strategy)

        evaluation = result["evaluations"][0]
        assert evaluation["quality"] in ("excellent", "good")
        assert evaluation["issues"] == []

    def test_check_summary_quality_with_too_short_draft_adds_issue_and_deducts_score(
        self, complete_alignment_strategy
    ):
        """
        Contract: draft with fewer than 80 words adds an issue and deducts 30 points.
        Expected value derived from module constants: "Too short (<80 words) → issue, -30."
        SummaryDraft.content has a min_length=50 char constraint, so we use a short
        sentence that meets the character floor but falls below 80 words.
        """
        short_content = (
            "Platform engineer with Python, AWS, and CI/CD experience improving cloud operations "
            "for product teams under production deadlines."
        )
        draft = _make_draft(short_content)
        summary = _make_summary(draft)

        result = check_summary_quality(summary, complete_alignment_strategy)

        evaluation = result["evaluations"][0]
        assert any("short" in issue.lower() for issue in evaluation["issues"])
        assert evaluation["score"] <= 70  # 100 - 30 at minimum

    def test_check_summary_quality_with_too_long_draft_adds_issue(
        self, complete_alignment_strategy
    ):
        """
        Contract: draft exceeding 110 words adds an issue and deducts 25 points.
        """
        long_content = " ".join(["word"] * 120)
        draft = _make_draft(long_content)
        summary = _make_summary(draft)

        result = check_summary_quality(summary, complete_alignment_strategy)

        evaluation = result["evaluations"][0]
        assert any("long" in issue.lower() for issue in evaluation["issues"])

    def test_check_summary_quality_with_nearly_all_supported_terms_missing_adds_warning(self):
        """
        Contract: weak supported-vocabulary coverage adds a warning, not an automatic issue,
        when the draft still retains some role relevance.
        """
        from src.data_models.strategy import SkillGap, SkillMatch

        strategy_with_five_keywords = AlignmentStrategy(
            overall_fit_score=75.0,
            summary_of_strategy="Focus on cloud.",
            identified_matches=[
                SkillMatch(
                    resume_skill="Python",
                    job_requirement="Python",
                    match_score=90.0,
                    justification="Direct match.",
                )
            ],
            identified_gaps=[
                SkillGap(
                    missing_skill="Terraform",
                    importance="should_have",
                    suggestion="Reframe IaC experience.",
                )
            ],
            keywords_to_integrate=["Python", "AWS", "Kubernetes", "Terraform", "Kafka"],
            professional_summary_guidance="Open with cloud leadership and quantified outcomes.",
            experience_guidance="Lead with impact metrics on the most recent role.",
            skills_guidance="Prioritise AWS services and infrastructure tooling.",
        )
        low_coverage_content = (
            "Platform engineer brought production discipline to complex service delivery for product teams operating under change pressure. "
            "Built Python workflows that reduced operational friction and kept releases steady, while translating messy delivery paths into cleaner runtime habits. "
            "Trusted to tighten execution when systems or handoffs became brittle, balancing hands-on engineering with practical coordination across cloud-heavy environments. "
            "Known for making unstable work easier to run without adding heavy process overhead, "
            "especially when ownership lines are blurred and teams need steadier execution before they need more tooling."
        )
        draft = _make_draft(low_coverage_content)
        summary = _make_summary(draft)

        result = check_summary_quality(summary, strategy_with_five_keywords)

        evaluation = result["evaluations"][0]
        assert evaluation["issues"] == []
        assert any("supported role vocabulary" in warning.lower() for warning in evaluation["warnings"])

    def test_check_summary_quality_with_no_supported_terms_present_adds_issue(
        self, complete_alignment_strategy
    ):
        """
        Contract: a polished summary that drops all supported role vocabulary loses role relevance
        and should receive an issue.
        """
        keyword_free_content = (
            "Turnaround-minded operator who brings structure to messy delivery environments. "
            "Known for calming ambiguous programs, tightening execution, and keeping teams aligned "
            "through changing priorities without creating unnecessary process drag."
        )
        draft = _make_draft(keyword_free_content)
        summary = _make_summary(draft)

        result = check_summary_quality(summary, complete_alignment_strategy)

        evaluation = result["evaluations"][0]
        assert any("role relevance" in issue.lower() for issue in evaluation["issues"])

    def test_check_summary_quality_returns_metadata_for_each_draft(
        self, complete_alignment_strategy, good_summary_draft
    ):
        """
        Contract: result includes draft_count, recommended version, and per-draft evaluations.
        """
        summary = ProfessionalSummary(
            drafts=[good_summary_draft],
            recommended_version=good_summary_draft.version_name,
        )

        result = check_summary_quality(summary, complete_alignment_strategy)

        assert result["overall_status"] == "complete"
        assert result["draft_count"] == 1
        assert result["recommended"] == good_summary_draft.version_name
        assert len(result["evaluations"]) == 1
        assert result["evaluations"][0]["version"] == good_summary_draft.version_name


# ── analyze_keyword_integration tests ─────────────────────────────────────────


class TestAnalyzeKeywordIntegration:
    """Tests for analyze_keyword_integration."""

    def test_analyze_keyword_integration_with_all_keywords_present_returns_rate_1(self):
        """
        Contract: integration_rate = 1.0 when all keywords appear in the text.
        Expected value derived from formula: total_integrated / total_required.
        """
        text = "This resume covers Python, AWS, and CI/CD experience."
        keywords = ["Python", "AWS", "CI/CD"]

        result = analyze_keyword_integration(text, keywords)

        assert result["integration_rate"] == 1.0
        assert result["total_integrated"] == 3
        assert result["missing_keywords"] == []

    def test_analyze_keyword_integration_with_no_keywords_present_returns_rate_0(self):
        """
        Contract: integration_rate = 0.0 when none of the keywords appear.
        """
        text = "Experienced professional with diverse skills."
        keywords = ["Python", "Kubernetes", "Terraform"]

        result = analyze_keyword_integration(text, keywords)

        assert result["integration_rate"] == 0.0
        assert result["total_integrated"] == 0
        assert set(result["missing_keywords"]) == {"Python", "Kubernetes", "Terraform"}

    def test_analyze_keyword_integration_with_partial_coverage_returns_correct_rate(self):
        """
        Contract: integration_rate = 2/3 when 2 of 3 keywords are present.
        Expected value: round(2/3, 2) = 0.67.
        """
        text = "Experienced Python engineer with strong AWS background."
        keywords = ["Python", "AWS", "Terraform"]

        result = analyze_keyword_integration(text, keywords)

        assert result["integration_rate"] == 0.67
        assert result["total_integrated"] == 2
        assert result["missing_keywords"] == ["Terraform"]

    def test_analyze_keyword_integration_is_case_insensitive(self):
        """
        Contract: keyword matching ignores case — 'python' matches 'Python' in the keyword list.
        """
        text = "expert in python and aws."
        keywords = ["Python", "AWS"]

        result = analyze_keyword_integration(text, keywords)

        assert result["integration_rate"] == 1.0

    def test_analyze_keyword_integration_with_empty_keyword_list_returns_zero_rate(self):
        """
        Contract: integration_rate = 0.0 when the required keyword list is empty
        (guards division by zero; 0/0 is defined as 0.0).
        """
        result = analyze_keyword_integration("Any text here.", [])

        assert result["integration_rate"] == 0.0
        assert result["total_required"] == 0
