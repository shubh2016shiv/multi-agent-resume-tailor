"""Unit tests for src/agents/professional_summary/engines.py

Contracts under test:
  check_summary_quality(summary, strategy) — scores each draft in a ProfessionalSummary
                                              for word count, keyword presence, and cliché
                                              avoidance. Returns per-draft evaluations.
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


# A good draft: 50–150 words, contains top keywords, no clichés.
_GOOD_DRAFT_CONTENT = (
    "Senior Python engineer with 6 years building AWS data pipelines. "
    "Led a team of 5 to deliver a real-time fraud detection system reducing losses by 30%. "
    "Expert in CI/CD practices, Kubernetes, and distributed systems. "
    "Seeking a staff engineering role where I can drive platform architecture."
)

# A draft that contains a cliché.
_CLICHE_DRAFT_CONTENT = (
    "I am a results-driven engineer with 6 years of Python and AWS experience. "
    "I have built multiple data pipelines and enjoy working on complex problems. "
    "Looking for a role where I can grow and make an impact on the team."
)


# ── check_summary_quality tests ───────────────────────────────────────────────


class TestCheckSummaryQuality:
    """Tests for check_summary_quality."""

    def test_check_summary_quality_with_good_draft_returns_good_or_excellent_quality(
        self, complete_alignment_strategy
    ):
        """
        Contract: a well-formed draft (correct word count, keywords present, no clichés)
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
        Contract: draft with fewer than 40 words adds an issue and deducts 30 points.
        Expected value derived from docstring: "Too short (<40 words) → issue, -30."
        SummaryDraft.content has a min_length=50 char constraint, so we use a short
        sentence that meets the character floor but falls below 40 words.
        """
        # 14 words, 74 characters — passes Pydantic's char constraint but fails the word-count check
        short_content = (
            "Python engineer with six years of AWS experience in cloud infrastructure and DevOps."
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
        Contract: draft exceeding 150 words adds an issue and deducts 25 points.
        """
        long_content = " ".join(["word"] * 160)  # 160 words — clearly over 150
        draft = _make_draft(long_content)
        summary = _make_summary(draft)

        result = check_summary_quality(summary, complete_alignment_strategy)

        evaluation = result["evaluations"][0]
        assert any("long" in issue.lower() for issue in evaluation["issues"])

    def test_check_summary_quality_with_cliche_adds_warning(self, complete_alignment_strategy):
        """
        Contract: a draft containing a known cliché (e.g. 'results-driven') adds a warning.
        Cliché list is defined in the docstring of check_summary_quality.
        """
        draft = _make_draft(_CLICHE_DRAFT_CONTENT)
        summary = _make_summary(draft)

        result = check_summary_quality(summary, complete_alignment_strategy)

        evaluation = result["evaluations"][0]
        assert any("clich" in w.lower() for w in evaluation["warnings"])

    def test_check_summary_quality_with_more_than_3_missing_keywords_adds_issue(self):
        """
        Contract: missing more than 3 of the top 5 strategy keywords adds an issue.
        Expected value derived from docstring: "> 3 missing → issue, -25."
        Uses a strategy with 5 keywords so the top-5 check is meaningful.
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
            keywords_to_integrate=["Kubernetes", "Terraform", "GraphQL", "Rust", "Kafka"],
            professional_summary_guidance="Open with cloud leadership and quantified outcomes.",
            experience_guidance="Lead with impact metrics on the most recent role.",
            skills_guidance="Prioritise AWS services and infrastructure tooling.",
        )
        # Content with none of the five keywords above
        keyword_free_content = (
            "Experienced professional with extensive background in project delivery "
            "and team management. Strong track record of success in technical roles "
            "across multiple industries and domains with diverse skill sets."
        )
        draft = _make_draft(keyword_free_content)
        summary = _make_summary(draft)

        result = check_summary_quality(summary, strategy_with_five_keywords)

        evaluation = result["evaluations"][0]
        assert any("keyword" in issue.lower() for issue in evaluation["issues"])

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
