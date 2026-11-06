from __future__ import annotations

import json

import pytest

from src.data.e03_a5_labels import (
    DIMENSIONS,
    LABEL_ORDER_VERSION,
    build_supervision,
    freeze_conversation_splits,
    profile_to_target,
    schema_label_order,
    validate_sft_row,
)


def _enums():
    return {
        "personality_traits": [f"p{i}" for i in range(10)],
        "interests": [f"i{i}" for i in range(12)],
        "style": [f"s{i}" for i in range(11)],
    }


def _schema(enums):
    return {"$defs": {"profile": {"properties": {
        dimension: {"items": {"enum": list(values)}} for dimension, values in enums.items()
    }}}}


def _output_schema(enums):
    return {"properties": {"user_profile": {"properties": {
        dimension: {"items": {"enum": list(values)}} for dimension, values in enums.items()
    }}}}


def _normalized(sample_id, conversation_id, *, index=0, profile=None, history=None):
    return {
        "sample_id": sample_id,
        "conversation_id": conversation_id,
        "history": history if history is not None else [{"role": "user", "content": "synthetic"}],
        "target": {"user_profile": profile or {dimension: [] for dimension in DIMENSIONS}},
        "source": {
            "split": "train",
            "source_line": index + 1,
            "prediction_point_index": index,
            "target_user_turn_id": index * 2 + 1,
            "assistant_response_turn_id": index * 2 + 2,
            "target_turn_index": index * 2,
            "response_turn_index": index * 2 + 1,
        },
    }


def _sft(row):
    return {
        "sample_id": row["sample_id"],
        "conversation_id": row["conversation_id"],
        "source_split": "train",
        "source": dict(row["source"]),
        "input_ids": [10, 11, 20, 21, 99],
        "labels": [-100, -100, 20, 21, 99],
        "prompt_length": 2,
        "completion_length": 3,
    }


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def test_schema_order_uses_official_dimension_and_enum_order():
    enums = _enums()
    order, loaded = schema_label_order(_schema(enums), _output_schema(enums))
    assert order[:2] == ["personality_traits:p0", "personality_traits:p1"]
    assert order[10:12] == ["interests:i0", "interests:i1"]
    assert order[22:24] == ["style:s0", "style:s1"]
    assert len(order) == 33
    assert loaded == enums


def test_schema_order_rejects_output_order_drift():
    train = _schema(_enums())
    output_enums = _enums()
    output_enums["interests"] = list(reversed(output_enums["interests"]))
    with pytest.raises(ValueError, match="enum order differs"):
        schema_label_order(train, _output_schema(output_enums))


def test_closed_world_empty_profile_has_zero_targets_and_full_mask():
    y, mask, _ = profile_to_target(
        {dimension: [] for dimension in DIMENSIONS}, _enums(), sample_id="sample-empty"
    )
    assert len(y) == len(mask) == 33
    assert sum(y) == 0
    assert mask == [1] * 33


def test_target_rejects_unknown_or_duplicate_labels():
    profile = {dimension: [] for dimension in DIMENSIONS}
    profile["style"] = ["unknown"]
    with pytest.raises(ValueError, match="outside official enum"):
        profile_to_target(profile, _enums(), sample_id="bad-label")
    profile["style"] = ["s0", "s0"]
    with pytest.raises(ValueError, match="duplicate"):
        profile_to_target(profile, _enums(), sample_id="duplicate-label")


def test_sft_row_validates_prompt_mask_and_completion_suffix():
    row = {
        "sample_id": "x",
        "source_split": "train",
        "input_ids": [1, 2, 3],
        "labels": [-100, 2, 3],
        "prompt_length": 1,
        "completion_length": 2,
    }
    assert validate_sft_row(row)["prompt_length"] == 1
    malformed = {**row, "labels": [1, -100, 3]}
    with pytest.raises(ValueError, match="Prompt labels"):
        validate_sft_row(malformed)
    malformed = {**row, "labels": [-100, 2, 4]}
    with pytest.raises(ValueError, match="Completion labels"):
        validate_sft_row(malformed)


def test_join_uses_sample_id_and_keeps_empty_gold_rows(tmp_path):
    enums = _enums()
    first = _normalized("id-a", "conv-a", profile={
        "personality_traits": ["p1"], "interests": [], "style": ["s0"]
    })
    second = _normalized("id-b", "conv-b")
    normalized_path = tmp_path / "normalized.jsonl"
    sft_path = tmp_path / "sft.jsonl"
    _write_jsonl(normalized_path, [first, second])
    _write_jsonl(sft_path, [_sft(second), _sft(first)])
    order = schema_label_order(_schema(enums))[0]

    rows, audit = build_supervision(normalized_path, sft_path, order, enums)
    by_id = {row["sample_id"]: row for row in rows}
    assert list(by_id) == ["id-a", "id-b"]
    assert by_id["id-a"]["y"][1] == 1
    assert by_id["id-a"]["y"][22] == 1
    assert by_id["id-b"]["y"] == [0] * 33
    assert by_id["id-b"]["mask"] == [1] * 33
    assert audit["join"]["join_key"] == "sample_id"
    assert audit["join"]["join_exact"] is True
    assert audit["target"]["profile_stats"]["all_three_dimensions_empty_count"] == 1


def test_join_rejects_source_metadata_mismatch(tmp_path):
    enums = _enums()
    normalized = _normalized("id-a", "conv-a")
    sft = _sft(normalized)
    sft["source"]["prediction_point_index"] = 9
    normalized_path = tmp_path / "normalized.jsonl"
    sft_path = tmp_path / "sft.jsonl"
    _write_jsonl(normalized_path, [normalized])
    _write_jsonl(sft_path, [sft])
    with pytest.raises(ValueError, match="source metadata differs"):
        build_supervision(normalized_path, sft_path, schema_label_order(_schema(enums))[0], enums)


def test_conversation_splits_are_deterministic_complete_and_isolated():
    rows = []
    for index in range(20):
        rows.append({
            "sample_id": f"sample-{index:02d}",
            "conversation_id": f"conv-{index // 2:02d}",
            "y": [int(index % 3 == label % 3) for label in range(33)],
        })
    split = freeze_conversation_splits(rows, seed=42)
    again = freeze_conversation_splits(rows, seed=42)
    assert split == again
    assert split["no_conversation_leakage"] is True
    ids = [sample for part in split["splits"].values() for sample in part["sample_ids"]]
    assert sorted(ids) == sorted(row["sample_id"] for row in rows)
    conv_groups = [set(part["conversation_ids"]) for part in split["splits"].values()]
    assert all(not left & right for i, left in enumerate(conv_groups) for right in conv_groups[i + 1:])
