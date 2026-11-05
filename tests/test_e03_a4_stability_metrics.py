from __future__ import annotations

import copy

import pytest

from src.evaluation.e03_a4_stability_metrics import paired_stability_summary


def _inputs():
    ids = [f"eval:{index}" for index in range(8)]
    conversations = [f"conversation:{index // 2}" for index in range(8)]
    history_lengths = [2, 3, 2, 4, 5, 8, 12, 17]
    gold = []
    for index in range(8):
        gold.append({
            "personality_traits": ["kind"] if index % 2 else [],
            "interests": ["music"] if index % 3 == 0 else [],
            "style": ["brief"] if index % 4 == 0 else [],
        })
    emotions = ["joy" if index % 2 else "sadness" for index in range(8)]
    enums = {
        "personality_traits": ["kind", "open"],
        "interests": ["music", "travel"],
        "style": ["brief", "formal"],
    }
    return ids, conversations, history_lengths, gold, emotions, enums


def _summarize(candidate_profiles=None, candidate_emotions=None, **overrides):
    ids, conversations, history_lengths, gold, emotions, enums = _inputs()
    return paired_stability_summary(
        ids=overrides.get("ids", ids),
        conversation_ids=conversations,
        history_lengths=history_lengths,
        gold_profiles=gold,
        baseline_profiles=gold,
        candidate_profiles=candidate_profiles if candidate_profiles is not None else gold,
        gold_emotions=emotions,
        baseline_emotions=emotions,
        candidate_emotions=candidate_emotions if candidate_emotions is not None else emotions,
        profile_enums=enums,
        emotion_labels=["joy", "sadness"],
        seed=42,
        bootstrap_repetitions=80,
        short_history_max_messages=4,
    )


def test_paired_stability_bootstrap_is_deterministic_and_groups_conversations():
    left = _summarize()
    right = _summarize()
    assert left == right
    assert left["status"] == "stability_preliminarily_passed"
    assert left["sample_count"] == 8
    assert left["conversation_count"] == 4
    assert left["history_groups"]["short_le_4_messages"]["baseline"]["sample_count"] == 4
    assert left["history_groups"]["long_gt_4_messages"]["candidate"]["sample_count"] == 4
    assert left["candidate_minus_baseline"]["profile"]["overall_33_label_micro_f1"] == 0.0
    assert left["conversation_cluster_bootstrap"]["confidence_intervals"]["micro_f1"] == {
        "lower_95": 0.0,
        "upper_95": 0.0,
        "replicates": 80,
        "unit": "conversation_cluster",
    }


def test_paired_stability_applies_predeclared_engineering_stop_thresholds():
    _ids, _conversations, _history, gold, emotions, _enums = _inputs()
    degraded_profiles = copy.deepcopy(gold)
    degraded_emotions = list(emotions)
    for index in range(len(gold)):
        degraded_profiles[index]["personality_traits"] = list(gold[index]["personality_traits"]) + ["open"]
        degraded_emotions[index] = "joy" if emotions[index] == "sadness" else "sadness"
    result = _summarize(degraded_profiles, degraded_emotions)
    assert result["status"] == "stability_failed"
    assert "false_positives_increased_over_10_percent" in result["threshold_failures"]
    assert "emotion_accuracy_decline_over_0.02" in result["threshold_failures"]
    assert result["full_json_and_reply_smoke"] == "not_run"


def test_paired_stability_rejects_duplicate_ids_and_out_of_schema_labels():
    ids, _conversations, _history, gold, _emotions, enums = _inputs()
    with pytest.raises(ValueError, match="unique IDs"):
        _summarize(ids=[*ids[:-1], ids[0]])
    invalid = copy.deepcopy(gold)
    invalid[0]["style"] = ["unknown"]
    with pytest.raises(ValueError, match="invalid style"):
        _summarize(invalid)
