from __future__ import annotations

import pytest

from benchmarks.run_banner_optimizer import (
    KEEP_MARGIN,
    TUNABLE_FIELDS,
    _feedback,
    _offline_proposal,
    apply_proposal,
    objective_from_record,
)


def test_objective_averages_all_candidates_and_winners():
    record = {"placements": [
        {"placement": "a", "winner": "c1", "score_means": {"c1": 9.0, "c2": 7.0}},
        {"placement": "b", "winner": "c2", "score_means": {"c1": 6.0, "c2": 8.0}},
    ]}
    obj, win = objective_from_record(record)
    assert obj == pytest.approx(7.5)   # mean of 9,7,6,8
    assert win == pytest.approx(8.5)   # mean of winners 9.0, 8.0


def test_objective_falls_back_to_winner_criteria_without_score_means():
    record = {"placements": [
        {"placement": "a", "winner": "c1", "judge": {"c1": {
            "focal_clarity": 8, "photo_quality": 8,
            "tonal_harmony": 9, "cta_craft": 9}}},
    ]}
    obj, win = objective_from_record(record)
    assert obj == pytest.approx(8.5)
    assert win == pytest.approx(8.5)


def test_objective_empty_record_is_none():
    assert objective_from_record({"placements": []}) == (None, None)


def test_apply_proposal_sets_field_without_mutating_original():
    brand = {"name": "X", "brief": "old brief", "creative_worldview": "w"}
    candidate = apply_proposal(
        brand, {"field": "brief", "value": "new brief", "rationale": "r"})
    assert candidate["brief"] == "new brief"
    assert brand["brief"] == "old brief"  # deep-copied, original untouched


@pytest.mark.parametrize("field", TUNABLE_FIELDS)
def test_apply_proposal_accepts_every_tunable_field(field):
    brand = {f: "seed" for f in TUNABLE_FIELDS} | {"name": "X", "brief": "b"}
    out = apply_proposal(brand, {"field": field, "value": "replacement"})
    assert out[field] == "replacement"


def test_apply_proposal_rejects_untunable_field():
    with pytest.raises(ValueError, match="not tunable"):
        apply_proposal({"brief": "b"}, {"field": "name", "value": "hacked"})


def test_apply_proposal_rejects_empty_value():
    with pytest.raises(ValueError, match="empty replacement"):
        apply_proposal({"brief": "b"}, {"field": "brief", "value": "   "})


def test_feedback_surfaces_weak_criteria_and_defects():
    record = {"placements": [
        {"placement": "leaderboard", "winner": "c1", "judge": {"c1": {
            "focal_clarity": 9, "photo_quality": 7, "tonal_harmony": 9,
            "cta_craft": 9, "defects": ["lens flare", "None"]}}},
    ]}
    fb = _feedback(record)
    assert "leaderboard" in fb
    assert "photo_quality=7" in fb
    assert "lens flare" in fb


def test_feedback_reports_ceiling_when_nothing_weak():
    record = {"placements": [
        {"placement": "a", "winner": "c1", "judge": {"c1": {
            "focal_clarity": 9, "photo_quality": 9, "tonal_harmony": 9,
            "cta_craft": 9, "defects": ["none"]}}},
    ]}
    assert "ceiling" in _feedback(record).lower()


def test_offline_proposal_is_deterministic_and_valid():
    brand = {"brief": "base brief"}
    p = _offline_proposal(brand, 3)
    assert p["field"] == "brief"
    assert p["value"].endswith("(offline variant 3)")
    apply_proposal(brand, p)  # must not raise


def test_keep_margin_is_a_positive_noise_guard():
    assert KEEP_MARGIN > 0
