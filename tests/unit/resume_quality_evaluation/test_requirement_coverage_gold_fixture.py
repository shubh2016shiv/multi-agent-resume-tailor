"""Contract tests for the requirement-coverage gold fixture."""

import json
from pathlib import Path

FIXTURE_PATH = (
    Path(__file__).parents[2]
    / "fixtures"
    / "resume_quality_evaluation"
    / "requirement_coverage_gold.json"
)

REQUIRED_KEYS = {
    "domain",
    "requirement",
    "expected_groups",
    "covered_groups",
    "missed_groups",
    "needs_judgment",
    "expected_coverage",
    "resume_span",
    "entailment",
}


def _load_gold_records() -> list[dict]:
    """Return the labeled requirement-coverage fixture records."""
    return json.loads(FIXTURE_PATH.read_text())


def test_gold_fixture_has_records_across_three_domains() -> None:
    """The Stage 0 fixture spans the required smoke domains."""
    records = _load_gold_records()
    domains = {record["domain"] for record in records}

    assert len(records) >= 20
    assert "ai_engineering" in domains
    assert "backend_engineering" in domains
    assert len(domains - {"ai_engineering", "backend_engineering"}) >= 1


def test_gold_fixture_records_have_stage_specific_expected_fields() -> None:
    """Each fixture row can diagnose extraction, presence, generic guard, and scoring."""
    for record in _load_gold_records():
        assert REQUIRED_KEYS <= set(record)
        assert isinstance(record["requirement"], str) and record["requirement"]
        assert isinstance(record["resume_span"], str) and record["resume_span"]
        assert record["entailment"] in {"ENTAILED", "NOT"}
        assert 0.0 <= record["expected_coverage"] <= 1.0
        _assert_group_list(record["expected_groups"])
        _assert_group_list(record["covered_groups"])
        _assert_group_list(record["missed_groups"])
        _assert_group_list(record["needs_judgment"])


def test_gold_fixture_groups_partition_expected_groups() -> None:
    """Covered, missed, and judgment groups are all declared expected groups."""
    for record in _load_gold_records():
        expected = {tuple(group) for group in record["expected_groups"]}
        classified = {
            tuple(group)
            for field in ("covered_groups", "missed_groups", "needs_judgment")
            for group in record[field]
        }

        assert classified <= expected


def _assert_group_list(groups: object) -> None:
    """Assert a fixture group field is a list of non-empty string lists."""
    assert isinstance(groups, list)
    for group in groups:
        assert isinstance(group, list)
        assert group
        assert all(isinstance(term, str) and term for term in group)
