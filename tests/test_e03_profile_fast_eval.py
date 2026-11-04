from __future__ import annotations

import pytest

from src.evaluation.profile_fast_eval import (
    compare_generated_prefix,
    profile_nonempty_recall,
    select_unique_conversation_samples,
)
from src.evaluation.source_integrity import (
    audit_prediction_point_sources,
    assign_dev_cluster_partitions,
    build_source_manifest,
    validate_history_matches_turns,
    validate_target_free_inputs,
    verify_prompt_token_ids,
)


def _inputs():
    return [
        {"id": "p1", "history": [{"role": "user", "content": "one"}]},
        {"id": "p2", "history": [{"role": "user", "content": "two"}]},
        {"id": "p3", "history": [{"role": "user", "content": "three"}]},
        {"id": "p4", "history": [{"role": "user", "content": "four"}]},
    ]


def _id_map():
    return [
        {"eval_id": "p1", "conversation_id": "c1", "sample_id": "s1"},
        {"eval_id": "p2", "conversation_id": "c1", "sample_id": "s2"},
        {"eval_id": "p3", "conversation_id": "c2", "sample_id": "s3"},
        {"eval_id": "p4", "conversation_id": "c3", "sample_id": "s4"},
    ]


def test_fixed_selection_is_repeatable_session_unique_and_target_free():
    selected = select_unique_conversation_samples(_inputs(), _id_map(), count=3, seed=42)
    repeated = select_unique_conversation_samples(_inputs(), _id_map(), count=3, seed=42)
    assert selected == repeated
    assert len({row["id"] for row in selected}) == 3
    assert all(set(row) == {"id", "history"} for row in selected)


def test_selection_rejects_gold_leak_duplicates_and_invalid_budgets():
    with pytest.raises(ValueError, match="only id and history"):
        validate_target_free_inputs([{"id": "p1", "history": _inputs()[0]["history"], "target": {}}])
    with pytest.raises(ValueError, match="duplicate id"):
        validate_target_free_inputs([_inputs()[0], _inputs()[0]])
    with pytest.raises(ValueError, match="between 1 and 32"):
        select_unique_conversation_samples(_inputs(), _id_map(), count=33, seed=42)


def test_current_user_is_verified_against_explicit_source_turn_id():
    turns = [
        {"turn_id": 10, "role": "user", "content": "prior"},
        {"turn_id": 11, "role": "assistant", "content": "answer"},
        {"turn_id": 20, "role": "user", "content": "current"},
        {"turn_id": 21, "role": "assistant", "content": "target response"},
    ]
    history = [
        {"role": "user", "content": "prior"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "current"},
    ]
    assert validate_history_matches_turns(turns, history, 20) == 2
    with pytest.raises(ValueError, match="omits or adds"):
        validate_history_matches_turns(turns, history[:-1], 20)
    with pytest.raises(ValueError, match="does not reference a user"):
        validate_history_matches_turns(turns, history, 11)


def test_prediction_point_preflight_blocks_missing_current_user_before_generation():
    raw = [{
        "conversation_id": "c1",
        "turns": [
            {"turn_id": 1, "role": "assistant", "content": "opening"},
            {"turn_id": 2, "role": "user", "content": "current"},
            {"turn_id": 3, "role": "assistant", "content": "response"},
        ],
        "prediction_points": [{"target_user_turn_id": 2}],
    }]
    normalized = [{
        "sample_id": "s1",
        "conversation_id": "c1",
        "source": {
            "split": "train", "source_line": 1, "prediction_point_index": 0,
            "target_user_turn_id": 2, "target_turn_index": 1,
        },
        "history": [{"role": "assistant", "content": "opening"}],
    }]
    with pytest.raises(ValueError, match="omits or adds"):
        audit_prediction_point_sources(normalized, raw)


def test_duplicate_sample_id_and_session_grouping_are_checked():
    rows = [
        {"sample_id": "s1", "conversation_id": "c1"},
        {"sample_id": "s2", "conversation_id": "c1"},
        {"sample_id": "s3", "conversation_id": "c2"},
    ]
    assignment = assign_dev_cluster_partitions(rows, seed=7, holdout_percent=35)
    assert assignment["s1"] == assignment["s2"]
    with pytest.raises(ValueError, match="duplicate sample_id"):
        assign_dev_cluster_partitions([rows[0], rows[0]], seed=7)


def test_equal_prompt_token_counts_do_not_imply_token_identity():
    assert verify_prompt_token_ids([1, 2, 3], [1, 2, 3])
    with pytest.raises(ValueError, match="despite matching token counts"):
        verify_prompt_token_ids([1, 2, 3], [3, 2, 1])


def test_source_manifest_contains_hashes_and_explicit_non_repair_status():
    manifest = build_source_manifest(
        source_path="outputs/dev.jsonl",
        source_sha256="a" * 64,
        input_rows=_inputs()[:2],
        id_map_rows=_id_map()[:2],
        prompt_version="P0",
        prompt_sha256="b" * 64,
        format_version="F2",
        truncation_version="T1",
        exporter_source_sha256="c" * 64,
    )
    assert manifest["selected_point_count"] == 2
    assert manifest["selected_conversation_count"] == 1
    assert manifest["history_repaired"] is False
    assert manifest["prompt_token_ids_sha256"] is None


def test_prefix_equivalence_requires_exact_token_ids_and_profile_values():
    same = compare_generated_prefix(
        [1, 2, 3, 4], [1, 2], "joy", "joy",
        {"personality_traits": [], "interests": ["music"], "style": []},
        {"personality_traits": [], "interests": ["music"], "style": []},
    )
    assert same["token_prefix_match"]
    assert same["emotion_and_profile_match"]
    changed = compare_generated_prefix(
        [1, 2, 3, 4], [2, 1], "joy", "joy",
        {"personality_traits": [], "interests": [], "style": []},
        {"personality_traits": [], "interests": [], "style": []},
    )
    assert not changed["token_prefix_match"]


def test_nonempty_dimension_recall_keeps_three_dimensions_separate():
    gold = [
        {"personality_traits": ["kind"], "interests": [], "style": ["warm"]},
        {"personality_traits": [], "interests": ["music"], "style": []},
    ]
    predicted = [
        {"personality_traits": [], "interests": [], "style": ["warm"]},
        {"personality_traits": [], "interests": ["music"], "style": []},
    ]
    recall = profile_nonempty_recall(gold, predicted)
    assert recall["personality_traits"]["recall"] == 0.0
    assert recall["interests"]["recall"] == 1.0
    assert recall["style"]["recall"] == 1.0
