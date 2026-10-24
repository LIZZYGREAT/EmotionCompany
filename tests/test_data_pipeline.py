"""Synthetic boundary tests for D0.3; no private source conversations are embedded."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from src.data.prepare import run_pipeline


EMOTIONS = [
    "joy", "gratitude", "relaxed", "care", "pride", "neutral", "surprise", "mixed",
    "sadness", "loneliness", "anxiety", "anger", "fear", "disgust", "shame", "helplessness",
]
PROFILE_ENUMS = {
    "personality_traits": ["sensitive", "introverted"],
    "interests": ["reading_writing", "music"],
    "style": ["rational", "brief"],
}


def _schema(split: str) -> dict:
    prefix = "train" if split == "train" else "val"
    profile = {
        "type": "object",
        "required": list(PROFILE_ENUMS),
        "additionalProperties": False,
        "properties": {
            name: {"type": "array", "items": {"enum": values}, "uniqueItems": True}
            for name, values in PROFILE_ENUMS.items()
        },
    }
    target = {
        "type": "object",
        "required": ["response_text", "emotion_label", "user_profile", "memory_refs"],
        "additionalProperties": False,
        "properties": {
            "response_text": {"type": "string", "minLength": 1},
            "emotion_label": {"enum": EMOTIONS},
            "user_profile": profile,
            "memory_refs": {"type": "array", "items": {"type": "string", "pattern": "^mem_[0-9]{6}$"}, "uniqueItems": True},
        },
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["conversation_id", "turns", "prediction_points"],
        "additionalProperties": False,
        "properties": {
            "conversation_id": {"type": "string", "pattern": rf"^{prefix}_conv_[0-9]{{6}}$"},
            "turns": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["turn_id", "role", "content"],
                    "additionalProperties": False,
                    "properties": {
                        "turn_id": {"type": "integer", "minimum": 1},
                        "role": {"enum": ["user", "assistant"]},
                        "content": {"type": "string", "minLength": 1},
                    },
                },
            },
            "prediction_points": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["target_user_turn_id", "assistant_response_turn_id", "target"],
                    "additionalProperties": False,
                    "properties": {
                        "target_user_turn_id": {"type": "integer", "minimum": 1},
                        "assistant_response_turn_id": {"type": "integer", "minimum": 1},
                        "target": target,
                    },
                },
            },
        },
    }


def _target(
    response: str = "answer",
    emotion: str = "anxiety",
    profile: dict | None = None,
    memory_refs: list[str] | None = None,
) -> dict:
    if profile is None:
        profile = {dim: [] for dim in PROFILE_ENUMS}
    if memory_refs is None:
        memory_refs = []
    return {
        "response_text": response,
        "emotion_label": emotion,
        "user_profile": copy.deepcopy(profile),
        "memory_refs": list(memory_refs),
    }


def _point(target_id: int, response_id: int, target: dict | None = None) -> dict:
    return {
        "target_user_turn_id": target_id,
        "assistant_response_turn_id": response_id,
        "target": copy.deepcopy(target or _target()),
    }


def _row(split: str, number: int, turns: list[tuple[int, str, str]], points: list[dict]) -> dict:
    prefix = "train" if split == "train" else "val"
    return {
        "conversation_id": f"{prefix}_conv_{number:06d}",
        "turns": [{"turn_id": turn_id, "role": role, "content": content} for turn_id, role, content in turns],
        "prediction_points": copy.deepcopy(points),
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _fixture(tmp_path: Path, train_rows: list[dict], val_rows: list[dict] | None = None):
    val_rows = val_rows or []
    (tmp_path / "data/train").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data/val").mkdir(parents=True, exist_ok=True)
    train_path = tmp_path / "data/train/train_public.jsonl"
    val_path = tmp_path / "data/val/val_public.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(val_path, val_rows)
    train_schema = tmp_path / "data/train/train_public.schema.json"
    val_schema = tmp_path / "data/val/val_public.schema.json"
    train_schema.write_text(json.dumps(_schema("train")), encoding="utf-8")
    val_schema.write_text(json.dumps(_schema("val")), encoding="utf-8")
    config = {
        "inputs": {
            "train": {"path": "data/train/train_public.jsonl", "schema": "data/train/train_public.schema.json"},
            "val": {"path": "data/val/val_public.jsonl", "schema": "data/val/val_public.schema.json"},
        },
        "outputs": {
            "normalized_dir": "data/processed/v1",
            "artifacts_dir": "outputs/data_pipeline/v1",
            "report": "docs/data/D0.3_数据构建报告.md",
        },
        "policy": {"version": "D0.2-v1", "short_response_chars": 5},
        "split": {
            "dev_ratio": 0.25,
            "seed": 42,
            "preserve_emotions": ["anxiety", "relaxed"],
            "large_group_fraction": 0.5,
            "max_dev_ratio_deviation": 1.0,
        },
        "validation": {"assert_input_unchanged": True},
    }
    return config, train_path, val_path


def _run(tmp_path: Path, train_rows: list[dict], val_rows: list[dict] | None = None):
    config, train_path, val_path = _fixture(tmp_path, train_rows, val_rows)
    result = run_pipeline(tmp_path, config)
    return result, config, train_path, val_path


def _read_samples(root: Path, split: str) -> list[dict]:
    path = root / "data/processed/v1" / f"{split}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_t01_turn_ids_unordered_use_array_positions(tmp_path: Path):
    row = _row("train", 1, [(90, "assistant", "prior"), (400, "user", "goal"), (2, "assistant", "answer")], [_point(400, 2, _target("answer"))])
    _run(tmp_path, [row])
    sample = (_read_samples(tmp_path, "train") + _read_samples(tmp_path, "dev"))[0]
    assert [turn["content"] for turn in sample["history"]] == ["prior", "goal"]
    assert sample["source"]["target_turn_index"] == 1
    assert sample["source"]["response_turn_index"] == 2


def test_t02_multiple_points_keep_separate_prefixes(tmp_path: Path):
    row = _row("train", 1, [
        (5, "user", "first"), (9, "assistant", "reply1"),
        (2, "user", "second"), (20, "assistant", "reply2"),
        (7, "user", "third"), (8, "assistant", "reply3"),
    ], [_point(2, 20, _target("reply2")), _point(7, 8, _target("reply3"))])
    _run(tmp_path, [row])
    samples = _read_samples(tmp_path, "train") + _read_samples(tmp_path, "dev")
    assert [len(sample["history"]) for sample in samples] == [3, 5]
    assert samples[0]["history"][-1]["content"] == "second"
    assert all(turn["content"] != "third" for turn in samples[0]["history"])


def test_t03_quarantine_only_invalid_point_in_conversation(tmp_path: Path):
    row = _row("train", 1, [(1, "user", "one"), (2, "assistant", "reply1"), (3, "user", "two"), (4, "assistant", "reply2")], [
        _point(1, 2, _target("reply1")), _point(3, 2, _target("reply2")),
    ])
    _run(tmp_path, [row])
    log = _read_jsonl(tmp_path / "outputs/data_pipeline/v1/cleaning_log.jsonl")
    samples = _read_samples(tmp_path, "train") + _read_samples(tmp_path, "dev")
    assert len(samples) == 1
    assert samples[0]["sample_id"].endswith("pp0")
    assert [(item["action"], item["reason"], item["prediction_point_index"]) for item in log if item["action"] == "quarantined"] == [
        ("quarantined", "RESPONSE_NOT_AFTER_TARGET", 1)
    ]


def test_t04_exact_history_and_target_keeps_first_source_point(tmp_path: Path):
    target = _target("answer")
    row = _row("train", 1, [(1, "user", "question"), (2, "assistant", "answer")], [_point(1, 2, target), _point(1, 2, target)])
    _run(tmp_path, [row])
    samples = _read_samples(tmp_path, "train") + _read_samples(tmp_path, "dev")
    log = _read_jsonl(tmp_path / "outputs/data_pipeline/v1/cleaning_log.jsonl")
    assert len(samples) == 1 and samples[0]["sample_id"].endswith("pp0")
    duplicate = next(item for item in log if item["action"] == "deduplicated")
    assert duplicate["sample_id"].endswith("pp1")
    assert duplicate["kept_as"].endswith("pp0")


def test_t05_same_history_different_target_is_preserved_and_reviewed(tmp_path: Path):
    p1 = {"personality_traits": ["sensitive"], "interests": [], "style": []}
    p2 = {"personality_traits": ["introverted"], "interests": [], "style": []}
    first = _row("train", 1, [(1, "user", "same question"), (2, "assistant", "first answer")], [_point(1, 2, _target("first answer", profile=p1))])
    second = _row("train", 2, [(10, "user", "same question"), (11, "assistant", "second answer")], [_point(10, 11, _target("second answer", profile=p2))])
    _run(tmp_path, [first, second])
    samples = _read_samples(tmp_path, "train") + _read_samples(tmp_path, "dev")
    summary = json.loads((tmp_path / "outputs/data_pipeline/v1/summary.json").read_text(encoding="utf-8"))
    log = _read_jsonl(tmp_path / "outputs/data_pipeline/v1/cleaning_log.jsonl")
    assert len(samples) == 2
    assert summary["duplicate_stats"]["same_history_different_target_groups"] == 1
    assert summary["duplicate_stats"]["same_history_label_conflict_groups"] == 1
    assert sum(item["reason"] == "SAME_HISTORY_DIFFERENT_TARGET" for item in log) == 2


def test_t06_same_full_turn_content_with_different_ids_stays_in_one_group(tmp_path: Path):
    target = _target("answer")
    first = _row("train", 1, [(1, "user", "same"), (2, "assistant", "answer")], [_point(1, 2, target)])
    second = _row("train", 2, [(101, "user", "same"), (202, "assistant", "answer")], [_point(101, 202, target)])
    _run(tmp_path, [first, second])
    manifest = json.loads((tmp_path / "outputs/data_pipeline/v1/split_manifest.json").read_text(encoding="utf-8"))
    mapping = manifest["conversation_to_split"]
    assert mapping[first["conversation_id"]] == mapping[second["conversation_id"]]
    assert manifest["max_group_conversation_count"] == 2


def test_t07_same_legal_history_across_conversations_stays_grouped(tmp_path: Path):
    first = _row("train", 1, [(1, "user", "same prompt"), (2, "assistant", "reply one")], [_point(1, 2, _target("reply one"))])
    second = _row("train", 2, [(10, "user", "same prompt"), (20, "assistant", "reply two")], [_point(10, 20, _target("reply two"))])
    _run(tmp_path, [first, second])
    manifest = json.loads((tmp_path / "outputs/data_pipeline/v1/split_manifest.json").read_text(encoding="utf-8"))
    mapping = manifest["conversation_to_split"]
    assert mapping[first["conversation_id"]] == mapping[second["conversation_id"]]
    assert manifest["max_group_conversation_count"] == 2


def test_t08_legal_empty_profile_rare_emotion_short_reply_and_memory_are_kept(tmp_path: Path):
    target = _target("好", emotion="surprise", memory_refs=["mem_000001"])
    row = _row("train", 1, [(1, "user", "prompt"), (2, "assistant", "好")], [_point(1, 2, target)])
    _run(tmp_path, [row])
    sample = (_read_samples(tmp_path, "train") + _read_samples(tmp_path, "dev"))[0]
    assert sample["target"] == target
    summary = json.loads((tmp_path / "outputs/data_pipeline/v1/summary.json").read_text(encoding="utf-8"))
    assert summary["distributions"]["unique_train"]["profile_all_empty_count"] == 1
    assert summary["distributions"]["unique_train"]["memory_refs_nonempty_count"] == 1
    assert summary["distributions"]["unique_train"]["short_response_under_5_count"] == 1
    assert summary["distributions"]["unique_train"]["emotion_counts"] == {"surprise": 1}


def test_t09_empty_prediction_conversation_has_no_sample_but_is_accounted(tmp_path: Path):
    empty = _row("train", 1, [(1, "user", "empty convo")], [])
    valid = _row("train", 2, [(3, "user", "prompt"), (4, "assistant", "answer")], [_point(3, 4)])
    _run(tmp_path, [empty, valid])
    summary = json.loads((tmp_path / "outputs/data_pipeline/v1/summary.json").read_text(encoding="utf-8"))
    samples = _read_samples(tmp_path, "train") + _read_samples(tmp_path, "dev")
    assert summary["train"]["empty_prediction_point_conversations"] == 1
    assert len(samples) == 1
    assert all(sample["conversation_id"] != empty["conversation_id"] for sample in samples)


def test_t10_train_dev_group_separation_and_val_is_not_deduplicated(tmp_path: Path):
    train = [
        _row("train", i, [(i * 10, "user", f"train question {i}"), (i * 10 + 1, "assistant", f"answer {i}")], [_point(i * 10, i * 10 + 1, _target(f"answer {i}"))])
        for i in range(1, 17)
    ]
    val = _row("val", 1, [(1, "user", "validation only"), (2, "assistant", "val answer")], [
        _point(1, 2, _target("val answer")), _point(1, 2, _target("val answer")),
    ])
    result, _, _, _ = _run(tmp_path, train, [val])
    manifest = json.loads((tmp_path / "outputs/data_pipeline/v1/split_manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["conversation_to_split"].values()) == {"train", "dev"}
    assert result["leakage_checks"]["train_dev_full_turn_overlap"] == 0
    assert result["leakage_checks"]["train_dev_legal_history_overlap"] == 0
    val_samples = _read_samples(tmp_path, "val")
    assert len(val_samples) == 2
    assert all(sample["source"]["split"] == "val" for sample in val_samples)


def test_t11_raw_file_hashes_are_unchanged(tmp_path: Path):
    row = _row("train", 1, [(1, "user", "prompt"), (2, "assistant", "answer")], [_point(1, 2)])
    result, _, train_path, val_path = _run(tmp_path, [row])
    actual = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (train_path, val_path)}
    summary = json.loads((tmp_path / "outputs/data_pipeline/v1/summary.json").read_text(encoding="utf-8"))
    assert result["input_sha256_unchanged"] is True
    assert summary["input_sha256_unchanged"] is True
    assert actual[str(train_path)] == summary["inputs"]["train"]["sha256_before"] == summary["inputs"]["train"]["sha256_after"]
    assert actual[str(val_path)] == summary["inputs"]["val"]["sha256_before"] == summary["inputs"]["val"]["sha256_after"]


def test_t12_repeated_build_is_byte_deterministic(tmp_path: Path):
    rows = [
        _row("train", i, [(i, "user", f"question {i}"), (i + 100, "assistant", f"answer {i}")], [_point(i, i + 100, _target(f"answer {i}"))])
        for i in range(1, 8)
    ]
    config, _, _ = _fixture(tmp_path, rows)
    run_pipeline(tmp_path, config)
    paths = [
        tmp_path / "data/processed/v1/train.jsonl",
        tmp_path / "data/processed/v1/dev.jsonl",
        tmp_path / "data/processed/v1/val.jsonl",
        tmp_path / "outputs/data_pipeline/v1/cleaning_log.jsonl",
        tmp_path / "outputs/data_pipeline/v1/split_manifest.json",
    ]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    run_pipeline(tmp_path, config)
    after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    assert before == after


def test_t13_every_raw_train_point_has_exactly_one_outcome(tmp_path: Path):
    first = _row("train", 1, [(1, "user", "question"), (2, "assistant", "answer")], [
        _point(1, 2, _target("answer")), _point(1, 2, _target("answer")),
    ])
    second = _row("train", 2, [(3, "user", "bad timing"), (4, "assistant", "old reply")], [_point(3, 4, _target("old reply"))])
    # Make the only second point deterministically invalid by targeting the later user with the earlier assistant.
    second["turns"].append({"turn_id": 5, "role": "user", "content": "later"})
    second["prediction_points"][0]["target_user_turn_id"] = 5
    result, _, _, _ = _run(tmp_path, [first, second])
    outcomes = result["train"]["outcomes"]
    assert result["train"]["raw_prediction_points"] == outcomes["quarantined"] + outcomes["deduplicated"] + outcomes["train"] + outcomes["dev"]
    assert result["train"]["raw_prediction_points"] == 3
    assert outcomes["quarantined"] == outcomes["deduplicated"] == 1
    assert result["group_stats"]["group_count"] >= 1


def test_t14_unsafe_output_paths_fail_before_publishing(tmp_path: Path):
    row = _row("train", 1, [(1, "user", "prompt"), (2, "assistant", "answer")], [_point(1, 2)])
    config, _, _ = _fixture(tmp_path, [row])
    bad = copy.deepcopy(config)
    bad["outputs"]["normalized_dir"] = "../outside"
    with pytest.raises(ValueError, match="escapes project root"):
        run_pipeline(tmp_path, bad)
    assert not (tmp_path.parent / "outside").exists()
    assert not (tmp_path / "data/processed/v1").exists()

    overlap = copy.deepcopy(config)
    overlap["outputs"]["normalized_dir"] = "data/train/train_public.jsonl"
    with pytest.raises(ValueError, match="overlaps input"):
        run_pipeline(tmp_path, overlap)
    assert not (tmp_path / "outputs/data_pipeline/v1").exists()
    assert not (tmp_path / "docs/data/D0.3_数据构建报告.md").exists()

    blocked_parent = tmp_path / "blocked-parent"
    blocked_parent.write_text("synthetic file", encoding="utf-8")
    blocked = copy.deepcopy(config)
    blocked["outputs"]["normalized_dir"] = "blocked-parent/v1"
    with pytest.raises(OSError):
        run_pipeline(tmp_path, blocked)
    assert not (tmp_path / "outputs/data_pipeline/v1").exists()
    assert not (tmp_path / "docs/data/D0.3_数据构建报告.md").exists()
