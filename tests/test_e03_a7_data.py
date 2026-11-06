import json

import pytest

from scripts.e03_a7_train import _frozen_dev_conversation_count
from src.data.e03_a7_classification import (
    SFTPromptIndex,
    build_schedule,
    build_supervision,
    emotion_label_order,
    sft_prompt_train_ids,
    validate_conversation_splits,
    verify_schedule,
)


def test_a7_frozen_dev_conversation_count_uses_and_validates_a5_id_map(tmp_path):
    id_map = tmp_path / "frozen_eval_id_map.jsonl"
    id_map.write_text(
        '{"eval_id":"eval-1","conversation_id":"conv-a"}\n'
        '{"eval_id":"eval-2","conversation_id":"conv-a"}\n'
        '{"eval_id":"eval-3","conversation_id":"conv-b"}\n',
        encoding="utf-8",
    )
    assets = {"frozen_ids": ["eval-1", "eval-2", "eval-3"], "source_paths": {"id_map": id_map}}
    assert _frozen_dev_conversation_count(assets) == 2

    assets["frozen_ids"] = ["eval-1", "eval-2"]
    with pytest.raises(ValueError, match="IDs differ"):
        _frozen_dev_conversation_count(assets)


def _schemas():
    profile = {
        "personality_traits": [f"p{i}" for i in range(10)],
        "interests": [f"i{i}" for i in range(12)],
        "style": [f"s{i}" for i in range(11)],
    }
    emotions = [f"emotion_{i}" for i in range(16)]
    train = {
        "$defs": {
            "profile": {"properties": {key: {"items": {"enum": values}} for key, values in profile.items()}},
            "emotion": {"enum": emotions},
        }
    }
    output = {
        "properties": {
            "user_profile": {"properties": {key: {"items": {"enum": values}} for key, values in profile.items()}},
            "emotion_label": {"enum": ["", *emotions]},
        }
    }
    return train, output, profile, emotions


def test_a7_emotion_order_cross_checks_both_official_schemas():
    train, output, _profile, emotions = _schemas()
    assert emotion_label_order(train, output) == emotions
    output["properties"]["emotion_label"]["enum"][2], output["properties"]["emotion_label"]["enum"][3] = (
        output["properties"]["emotion_label"]["enum"][3], output["properties"]["emotion_label"]["enum"][2]
    )
    with pytest.raises(ValueError, match="order differs"):
        emotion_label_order(train, output)


def test_a7_prompt_index_train_ids_cover_all_three_supervised_partitions():
    splits = {
        "fit": ["fit-1", "fit-2"],
        "calibration": ["cal-1"],
        "internal_holdout": ["holdout-1"],
    }
    assert sft_prompt_train_ids(splits) == ["fit-1", "fit-2", "cal-1", "holdout-1"]
    splits["internal_holdout"] = ["fit-1"]
    with pytest.raises(ValueError, match="unique"):
        sft_prompt_train_ids(splits)


def test_a7_supervision_joins_profile_and_emotion_by_sample_id(tmp_path):
    train_schema, output_schema, enums, emotions = _schemas()
    normalized = tmp_path / "normalized.jsonl"
    profile_path = tmp_path / "profile.jsonl"
    normalized_rows = [
        {
            "sample_id": f"sample-{i}",
            "conversation_id": f"conv-{i}",
            "source": {"split": "train"},
            "history": [{"role": "user", "content": "x"}],
            "target": {"emotion_label": emotions[i]},
        }
        for i in range(2)
    ]
    normalized.write_text("".join(json.dumps(row) + "\n" for row in normalized_rows), encoding="utf-8")
    profile_rows = []
    for i in range(2):
        target = [0] * 33
        target[i] = 1
        profile_rows.append({"sample_id": f"sample-{i}", "y": target, "mask": [1] * 33})
    profile_path.write_text("".join(json.dumps(row) + "\n" for row in profile_rows), encoding="utf-8")
    supervision, summary = build_supervision(
        normalized, profile_path, train_schema, output_schema, expected_rows=2
    )
    assert supervision["sample-0"]["profile_target"][0] == 1
    assert supervision["sample-1"]["emotion_target"] == 1
    assert summary["sample_id_sha256"]


def test_a7_schedule_freezes_epoch_boundary_and_second_pass():
    ids = [f"fit-{i:05}" for i in range(11_304)]
    schedule = build_schedule(ids)
    facts = verify_schedule(schedule, ids)
    assert facts["status"] == "schedule_validated"
    assert facts["draw_count"] == 16_192
    assert schedule["steps"][2825]["epoch"] == 1
    assert schedule["steps"][2826]["epoch"] == 2
    assert schedule["steps"][0]["sample_ids"] == build_schedule(ids)["steps"][0]["sample_ids"]
    assert set(sid for step in schedule["steps"][:2826] for sid in step["sample_ids"]) == set(ids)


def test_a7_schedule_rejects_non_frozen_inputs():
    with pytest.raises(ValueError, match="11,304"):
        build_schedule(["one"])


def test_a7_conversation_split_requires_full_nonoverlapping_coverage():
    supervision = {
        f"sample-{index}": {"conversation_id": f"conversation-{index}"}
        for index in range(3)
    }
    splits = {"fit": ["sample-0"], "calibration": ["sample-1"], "internal_holdout": ["sample-2"]}
    summary = validate_conversation_splits(splits, supervision)
    assert summary["fit"]["conversation_count"] == 1
    splits["internal_holdout"] = ["sample-0"]
    with pytest.raises(ValueError, match="overlap"):
        validate_conversation_splits(splits, supervision)


def test_sft_prompt_index_returns_only_prompt_prefix(tmp_path):
    path = tmp_path / "sft.jsonl"
    row = {
        "sample_id": "sample-0",
        "source_split": "train",
        "input_ids": [10, 11, 12, 13],
        "labels": [-100, -100, 12, 13],
        "prompt_length": 2,
        "completion_length": 2,
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    index = SFTPromptIndex(path, ["sample-0"])
    assert index.prompt_ids("sample-0") == [10, 11]
    assert list(index.iter_prompt_rows(["sample-0"])) == [
        {"sample_id": "sample-0", "input_ids": [10, 11], "prompt_length": 2}
    ]
