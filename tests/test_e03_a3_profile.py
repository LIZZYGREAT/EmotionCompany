"""Synthetic CPU regressions for E03-A3 profile aggregation."""
from __future__ import annotations

import pytest

from scripts.e01_b_diagnose import _safe_parse
from src.diagnostics.e03_a3_profile import (
    DIMENSIONS,
    _profile_counts,
    cluster_bootstrap_recall_gap,
    freeze_conversation_subset,
    join_by_id,
    length_bucket,
    opportunity_space,
    retained_message_count,
    transition_counts,
    validate_history,
    validate_profile,
)


ENUMS = {
    "personality_traits": ["p1", "p2"],
    "interests": ["i1"],
    "style": ["s1"],
}


def profile(**values):
    return {
        "personality_traits": values.get("personality_traits", []),
        "interests": values.get("interests", []),
        "style": values.get("style", []),
    }


def row(conversation, history_count, gold, pred, **extra):
    return {
        "conversation_id": conversation,
        "history_length": history_count,
        "retained_history_length": extra.pop("retained", history_count),
        "removed_history_messages": extra.pop("removed", 0),
        "prompt_tokens": extra.pop("prompt_tokens", 100),
        "target_turn_index": extra.pop("target_turn_index", 1),
        "gold": validate_profile(gold, ENUMS),
        "pred": validate_profile(pred, ENUMS),
        **extra,
    }


def test_length_buckets_cover_boundaries_without_collapse():
    assert [length_bucket(n) for n in (1, 4, 5, 8, 9, 16, 17, 41)] == [
        "1-4", "1-4", "5-8", "5-8", "9-16", "9-16", "17+", "17+"
    ]
    with pytest.raises(ValueError):
        length_bucket(0)


def test_history_must_include_current_user_and_t1_keeps_it():
    history = [
        {"role": "assistant", "content": "synthetic"},
        {"role": "user", "content": "synthetic"},
    ]
    assert validate_history(history) == 2
    with pytest.raises(ValueError, match="current user"):
        validate_history([{"role": "user", "content": "synthetic"}, {"role": "assistant", "content": "synthetic"}])
    assert retained_message_count(10, 7) == 3
    with pytest.raises(ValueError, match="retain the current user"):
        retained_message_count(10, 10)


def test_join_uses_ids_when_file_order_differs_and_rejects_mismatch():
    left = [{"eval_id": "b", "sample": 2}, {"eval_id": "a", "sample": 1}]
    right = [{"sample_id": "a", "target": "gold-a"}, {"sample_id": "b", "target": "gold-b"}]
    joined = join_by_id(left, right, left_key="eval_id", right_key="sample_id")
    assert [(key, first["sample"], second["target"]) for key, first, second in joined] == [
        ("b", 2, "gold-b"), ("a", 1, "gold-a")
    ]
    with pytest.raises(ValueError, match="ID sets differ"):
        join_by_id(left, right[:1], left_key="eval_id", right_key="sample_id")


def test_profile_validation_rejects_non_lists_duplicates_and_unknown_labels():
    with pytest.raises(ValueError, match="must be a list"):
        validate_profile({**profile(), "style": "s1"}, ENUMS)
    with pytest.raises(ValueError, match="duplicate"):
        validate_profile(profile(personality_traits=["p1", "p1"]), ENUMS)
    with pytest.raises(ValueError, match="outside the fixed enum"):
        validate_profile(profile(interests=["unsupported"]), ENUMS)


def test_multiple_points_in_one_conversation_conserve_gold_and_prediction_counts():
    first = row(
        "conversation-a", 3,
        profile(personality_traits=["p1"]),
        profile(personality_traits=["p1"], style=["s1"]),
    )
    second = row(
        "conversation-a", 5,
        profile(personality_traits=["p1", "p2"]),
        profile(),
        retained=3,
        removed=2,
    )
    third = row("conversation-b", 18, profile(), profile())
    result = _profile_counts([first, second, third], ENUMS)
    assert result["point_count"] == 3
    assert result["conversation_count"] == 2
    assert result["gold_nonempty_points"] == 2
    assert result["gold_all_empty_points"] == 1
    assert result["gold_nonempty_pred_all_empty_points"] == 1
    assert result["gold_nonempty_pred_all_empty_rate"] == 0.5
    assert result["profile_tp"] == 1
    assert result["profile_fp"] == 1
    assert result["profile_fn"] == 2
    assert result["profile_tp"] + result["profile_fn"] == 3
    assert result["profile_tp"] + result["profile_fp"] == 2


def test_invalid_raw_is_counted_as_a_parse_failure_not_silently_scored_as_empty():
    parsed, failure = _safe_parse('{"profile": []} trailing garbage')
    assert parsed is None
    assert failure == "json_decode:Extra data"
    assert "trailing garbage" not in failure


def test_e01b_to_a1_transition_types_are_mutually_exclusive():
    base = profile(personality_traits=["p1"], interests=["i1"], style=["s1"])
    a1 = profile(personality_traits=["p2"], style=["s1"])
    item = {
        "gold": validate_profile(profile(personality_traits=["p1"], interests=["i1"]), ENUMS),
        "before": validate_profile(base, ENUMS),
        "after": validate_profile(a1, ENUMS),
        "bucket": "1-4",
    }
    result = transition_counts([item], ENUMS, before_key="before", after_key="after", group_field="bucket")
    assert result["overall"]["personality_traits"]["p1"]["tp_to_fn"] == 1
    assert result["overall"]["personality_traits"]["p2"]["tn_to_fp"] == 1
    assert result["overall"]["interests"]["i1"]["tp_to_fn"] == 1
    assert result["overall"]["style"]["s1"]["fp_to_tn"] == 0
    assert result["by_group"]["1-4"]["point_count"] == 1


def test_optimistic_opportunity_and_fp_sensitivity_use_global_denominators():
    result = opportunity_space(total_tp=1130, total_fp=1271, total_fn=1821, group_fn=100)
    assert result["total_gold_positive_labels"] == 2951
    assert result["required_additional_tp_for_micro_f1_0_50_if_fp_fixed"] == 278
    clean_full_recovery = next(
        item for item in result["scenarios"]
        if item["fraction_of_group_fn_recovered"] == 1 and item["added_fp_per_recovered_tp"] == 0
    )
    one_fp_per_tp = next(
        item for item in result["scenarios"]
        if item["fraction_of_group_fn_recovered"] == 1 and item["added_fp_per_recovered_tp"] == 1
    )
    assert clean_full_recovery["micro_f1"] > 0.42227
    assert one_fp_per_tp["micro_f1"] < clean_full_recovery["micro_f1"]


def test_cluster_bootstrap_is_reproducible_and_groups_by_conversation():
    rows = [
        row("c1", 3, profile(personality_traits=["p1"]), profile()),
        row("c1", 5, profile(personality_traits=["p1"]), profile(personality_traits=["p1"])),
        row("c2", 4, profile(interests=["i1"]), profile(interests=["i1"])),
        row("c3", 9, profile(style=["s1"]), profile()),
    ]
    first = cluster_bootstrap_recall_gap(rows, repetitions=50, seed=7)
    second = cluster_bootstrap_recall_gap(rows, repetitions=50, seed=7)
    assert first == second
    assert first["conversation_count"] == 3
    assert first["estimand"] == "micro_label_recall_short_minus_long"
    assert first["interpretation"] == "descriptive_uncertainty_only_not_a_causal_test"


def test_frozen_eval_subset_is_deterministic_gold_stratified_and_keeps_conversations_whole():
    rows = []
    for conversation_index in range(30):
        conversation_id = f"conversation-{conversation_index:02d}"
        for point_index in range(10):
            if point_index == 0:
                gold = profile()
            elif point_index == 1:
                gold = profile(personality_traits=["p1"])
            elif point_index == 2:
                gold = profile(interests=["i1"])
            elif point_index == 3:
                gold = profile(style=["s1"])
            else:
                gold = profile(personality_traits=["p1"])
            item = row(
                conversation_id,
                3 if point_index % 2 == 0 else 7,
                gold,
                profile(),
            )
            item["eval_id"] = f"eval-{conversation_index:02d}-{point_index:02d}"
            rows.append(item)
    selected, summary = freeze_conversation_subset(rows, seed=20261104)
    selected_again, summary_again = freeze_conversation_subset(list(reversed(rows)), seed=20261104)
    assert [item["eval_id"] for item in selected] == [item["eval_id"] for item in selected_again]
    assert summary == summary_again
    assert 192 <= summary["selected_points"] <= 256
    assert summary["selection_uses_predictions"] is False
    assert summary["conversation_groups_split"] is False
    assert summary["gold_all_empty_points"] > 0
    assert summary["gold_nonempty_points"] > 0
    assert all(count > 0 for count in summary["dimension_nonempty_point_counts"].values())
    selected_by_conversation = {}
    for item in selected:
        selected_by_conversation.setdefault(item["conversation_id"], 0)
        selected_by_conversation[item["conversation_id"]] += 1
    assert set(selected_by_conversation.values()) == {10}
