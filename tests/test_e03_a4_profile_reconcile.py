from __future__ import annotations

import pytest

from src.evaluation.profile_reconcile import (
    classify_historical_difference,
    compare_profile_lists,
    empty_profile,
    profile_difference_dimensions,
    validate_profile,
)


ENUMS = {
    "emotion_label": ["neutral"],
    "personality_traits": ["open", "sensitive"],
    "interests": ["games"],
    "style": ["direct"],
}


def _profile(*, personality=(), interests=(), style=()):
    return {
        "personality_traits": list(personality),
        "interests": list(interests),
        "style": list(style),
    }


def test_pairwise_profile_metrics_keep_fixed_denominator_and_report_tp_fp_fn_deltas():
    gold = [_profile(personality=("open",), interests=("games",)), _profile(style=("direct",))]
    submission = [empty_profile(), _profile(style=("direct",))]
    raw_prefix = [_profile(personality=("open",), interests=("games",)), _profile()]

    result = compare_profile_lists(
        ids=["a", "b"],
        gold_profiles=gold,
        left_profiles=submission,
        right_profiles=raw_prefix,
        enums=ENUMS,
        left_name="S",
        right_name="R",
    )

    assert result["sample_count"] == 2
    assert result["different_sample_count"] == 2
    assert result["different_sample_count_by_dimension"] == {
        "personality_traits": 1,
        "interests": 1,
        "style": 1,
    }
    assert result["left_metrics"]["per_dimension"]["style"]["sample_count"] == 2
    assert result["left_metrics"]["overall_33_label_fn"] == 2
    assert result["right_metrics"]["overall_33_label_tp"] == 2
    assert result["right_minus_left"]["overall_33_label_tp"] == 1
    assert result["right_minus_left"]["overall_33_label_fn"] == -1


def test_profile_set_comparison_rejects_id_count_mapping_errors():
    with pytest.raises(ValueError, match="counts must match"):
        compare_profile_lists(
            ids=["a", "b"],
            gold_profiles=[_profile()],
            left_profiles=[_profile()],
            right_profiles=[_profile()],
            enums=ENUMS,
            left_name="S",
            right_name="R",
        )

    with pytest.raises(ValueError, match="duplicate IDs"):
        compare_profile_lists(
            ids=["a", "a"],
            gold_profiles=[_profile(), _profile()],
            left_profiles=[_profile(), _profile()],
            right_profiles=[_profile(), _profile()],
            enums=ENUMS,
            left_name="S",
            right_name="R",
        )


def test_profile_rows_reject_duplicate_and_unknown_labels():
    with pytest.raises(ValueError, match="invalid labels for interests"):
        validate_profile(_profile(interests=("games", "games")), ENUMS, label="fixture")
    with pytest.raises(ValueError, match="invalid labels for style"):
        validate_profile(_profile(style=("unknown",)), ENUMS, label="fixture")


def test_historical_attribution_separates_downstream_json_failure_from_prefix_failure():
    empty = empty_profile()
    predicted = _profile(personality=("open",))
    assert classify_historical_difference(
        submission_profile=empty,
        prefix_profile=predicted,
        prefix_status="valid",
        full_json_status="invalid_json",
    ) == "downstream_full_json_parse_failure"
    assert classify_historical_difference(
        submission_profile=predicted,
        prefix_profile=empty,
        prefix_status="incomplete",
        full_json_status="valid",
    ) == "prefix_unusable_submission_differs"
    assert classify_historical_difference(
        submission_profile=empty,
        prefix_profile=predicted,
        prefix_status="valid",
        full_json_status="valid",
    ) == "submission_normalization_or_mapping_difference"
    assert classify_historical_difference(
        submission_profile=predicted,
        prefix_profile=predicted,
        prefix_status="valid",
        full_json_status="valid",
    ) == "profile_match"


def test_profile_dimension_comparison_ignores_label_order():
    left = _profile(personality=("open", "sensitive"))
    right = _profile(personality=("sensitive", "open"))
    assert profile_difference_dimensions(left, right) == []
