from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.e03_a8_replay import (
    assert_safe_replay_checkpoint_path,
    build_replay_schedule,
    score_candidate_difficulty,
    select_candidates,
    set_replay_learning_rate,
    validate_encoded_gold_completion,
)
from src.evaluation.e03_a8_replay_metrics import freeze_val_splits, gate_reasons

SCHEMA = json.loads((Path(__file__).resolve().parents[1] / "data/train/train_public.schema.json").read_text(encoding="utf-8"))


def _gold(index: int) -> dict[str, list[str]]:
    if index % 4 == 0:
        return {"personality_traits": [], "interests": [], "style": []}
    return {
        "personality_traits": ["sensitive"] if index % 2 else [],
        "interests": ["music"] if index % 3 else [],
        "style": ["rational"] if index % 5 else [],
    }


def test_select_candidates_is_deterministic_and_source_stratified():
    sources = {"train": [], "dev": []}
    for source, count in (("train", 80), ("dev", 20)):
        for index in range(count):
            sources[source].append({
                "sample_id": f"{source}:{index}",
                "conversation_id": f"{source}-conversation-{index // 2}",
                "target": {"user_profile": _gold(index)},
                "prompt_length": 300 + index,
                "history_message_count": 1 + index % 12,
            })
    first, audit1 = select_candidates(sources, total_count=30, source_counts={"train": 27, "dev": 3})
    second, audit2 = select_candidates(sources, total_count=30, source_counts={"train": 27, "dev": 3})
    assert [row["sample_id"] for row in first] == [row["sample_id"] for row in second]
    assert audit1 == audit2
    assert audit1["source_counts_actual"] == {"train": 27, "dev": 3}
    assert audit1["overall_gold_nonempty"] + audit1["overall_gold_empty"] == 30
    assert len({row["sample_id"] for row in first}) == 30


def test_difficulty_keeps_dimension_label_keys_and_invalid_separate():
    candidates = [
        {"sample_id": "miss", "source": "train", "conversation_id": "c1", "gold_profile": {"personality_traits": ["sensitive"], "interests": [], "style": []}, "candidate_tie_sha256": "a"},
        {"sample_id": "extra", "source": "dev", "conversation_id": "c2", "gold_profile": {"personality_traits": [], "interests": [], "style": []}, "candidate_tie_sha256": "b"},
        {"sample_id": "mixed", "source": "train", "conversation_id": "c3", "gold_profile": {"personality_traits": ["sensitive"], "interests": [], "style": []}, "candidate_tie_sha256": "c"},
        {"sample_id": "clean", "source": "train", "conversation_id": "c4", "gold_profile": {"personality_traits": [], "interests": [], "style": []}, "candidate_tie_sha256": "d"},
        {"sample_id": "invalid", "source": "train", "conversation_id": "c5", "gold_profile": {"personality_traits": [], "interests": [], "style": []}, "candidate_tie_sha256": "e"},
    ]
    predictions = {
        "miss": {"status": "valid", "parsed": {"user_profile": {"personality_traits": [], "interests": [], "style": []}}},
        "extra": {"status": "valid", "parsed": {"user_profile": {"personality_traits": ["sensitive"], "interests": [], "style": []}}},
        "mixed": {"status": "valid", "parsed": {"user_profile": {"personality_traits": [], "interests": ["music"], "style": []}}},
        "clean": {"status": "valid", "parsed": {"user_profile": {"personality_traits": [], "interests": [], "style": []}}},
        "invalid": {"status": "invalid", "parsed": None},
    }
    hard, stats = score_candidate_difficulty(
        candidates,
        predictions,
        {"personality_traits:sensitive": 2, "interests:music": 1},
    )
    by_id = {row["sample_id"]: row for row in hard}
    assert {key: by_id[key]["group"] for key in by_id} == {
        "miss": "miss", "extra": "extra", "mixed": "mixed", "clean": "clean", "invalid": "invalid",
    }
    assert by_id["invalid"]["fn_count"] is None
    assert by_id["invalid"]["fp_count"] is None
    assert by_id["miss"]["difficulty"] == 2.25
    assert by_id["extra"]["difficulty"] == 1.5
    assert by_id["mixed"]["difficulty"] == 4.25
    assert by_id["mixed"]["false_negatives"] == [{"dimension": "personality_traits", "label": "sensitive"}]
    assert by_id["mixed"]["false_positives"] == [{"dimension": "interests", "label": "music"}]
    assert stats["group_counts"] == {"miss": 1, "extra": 1, "mixed": 1, "clean": 1, "invalid": 1}


def test_schedule_freezes_counts_repeats_source_caps_and_step_uniqueness():
    candidates = []
    hard = []
    for index in range(2400):
        source = "dev" if index % 10 == 0 else "train"
        sample_id = f"{source}:{index}"
        dimension = ("personality_traits", "interests", "style")[index % 3]
        profile = {"personality_traits": [], "interests": [], "style": []}
        profile[dimension] = ["sensitive"]
        candidates.append({
            "sample_id": sample_id,
            "source": source,
            "conversation_id": f"conversation-{index}",
            "gold_profile": profile,
            "gold_nonempty": True,
            "history_message_count": 2 + index % 20,
            "candidate_tie_sha256": f"{index:064x}",
        })
        if index < 1000:
            group = "miss"
        elif index < 2000:
            group = "extra"
        else:
            group = "clean"
        hard.append({
            **candidates[-1],
            "group": group,
            "difficulty": 2.0 if group == "miss" else 1.5 if group == "extra" else 0.0,
            "fn_count": 1 if group == "miss" else 0,
            "fp_count": 1 if group == "extra" else 0,
            "prediction_status": "valid",
        })
    schedule, audit = build_replay_schedule(hard, candidates)
    assert len(schedule) == 1536
    assert audit["pilot_steps"] == 128
    assert audit["extension_steps"] == 256
    assert audit["pilot_repeat_rate"] >= 0.20
    assert max(audit["sample_exposure_counts"].values()) <= 4
    assert sum(value for key, value in audit["source_exposure_counts"].items() if key == "dev") <= 384
    assert set(audit["profile_dimension_exposure_counts"]) == {"personality_traits", "interests", "style"}
    assert all(audit["profile_dimension_exposure_counts"][dimension] > 0 for dimension in ("personality_traits", "interests", "style"))
    for start in range(0, len(schedule), 4):
        step = schedule[start:start + 4]
        assert len({row["sample_id"] for row in step}) == 4
        assert len({row["optimizer_step"] for row in step}) == 1
    counts = audit["sample_exposure_counts"]
    for row in schedule:
        assert 1 <= row["occurrence_in_schedule"] <= counts[row["sample_id"]]


def test_candidate_selection_reallocates_a_short_source_without_changing_total():
    sources = {"train": [], "dev": []}
    for index in range(20):
        sources["train"].append({
            "sample_id": f"train:{index}",
            "conversation_id": f"train-c{index}",
            "target": {"user_profile": _gold(index)},
            "prompt_length": 300 + index,
            "history_message_count": 1 + index % 12,
        })
    for index in range(2):
        sources["dev"].append({
            "sample_id": f"dev:{index}",
            "conversation_id": f"dev-c{index}",
            "target": {"user_profile": _gold(index)},
            "prompt_length": 400 + index,
            "history_message_count": 2 + index,
        })
    selected, audit = select_candidates(sources, total_count=10, source_counts={"train": 7, "dev": 3})
    assert len(selected) == 10
    assert audit["source_counts_allocated"] == {"train": 8, "dev": 2}
    assert audit["source_counts_actual"] == {"train": 8, "dev": 2}
    assert audit["source_audit"]["dev"]["reallocated_from_requested_quota"] == -1


def test_replay_lr_reset_and_checkpoint_path_protection(tmp_path):
    class Optimizer:
        def __init__(self):
            self.param_groups = [{"lr": 1e-4}, {"lr": 1e-4}]
            self.defaults = {"lr": 1e-4}

    class Scheduler:
        def __init__(self):
            self.base_lrs = [1e-4, 1e-4]
            self._last_lr = [1e-4, 1e-4]

    optimizer = Optimizer()
    scheduler = Scheduler()
    facts = set_replay_learning_rate(optimizer, scheduler, 2e-5)
    assert facts["optimizer_group_lrs"] == [2e-5, 2e-5]
    assert scheduler.base_lrs == [2e-5, 2e-5]
    assert optimizer.defaults["lr"] == 2e-5

    parent = tmp_path / "parent"
    local = tmp_path / "baseline"
    archive = tmp_path / "archive"
    with pytest.raises(ValueError, match="protected model path"):
        assert_safe_replay_checkpoint_path(
            local / "checkpoint",
            parent_checkpoint=parent,
            local_baseline=local,
            archive_root=archive,
        )


def test_gold_round_trip_ignores_only_template_text_after_eos():
    completion = "Personality=-\nInterests=-\nStyle=-"

    class Tokenizer:
        eos_token_id = 99

        def decode(self, ids, **_kwargs):
            ids = list(ids)
            if ids == [99]:
                return "<|im_end|>"
            if ids == [2, 99, 3]:
                return completion + "<|im_end|>\n"
            raise AssertionError(f"unexpected decode request: {ids}")

    parsed = validate_encoded_gold_completion(
        Tokenizer(),
        [1, 2, 99, 3],
        1,
        completion,
        SCHEMA,
        sample_id="train:synthetic:pp0",
    )
    assert parsed["user_profile"] == {"personality_traits": [], "interests": [], "style": []}


def test_val_splits_are_conversation_disjoint_and_repeatable():
    inputs = []
    id_map = {}
    for conversation in range(20):
        for point in range(2):
            eval_id = f"val:{conversation}:{point}"
            inputs.append({"id": eval_id, "history": []})
            id_map[eval_id] = {"conversation_id": f"c{conversation}"}
    first = freeze_val_splits(inputs, id_map, {"c0", "c1"})
    second = freeze_val_splits(inputs, id_map, {"c0", "c1"})
    assert first == second
    gate = first["cohorts"]["val_gate"]
    confirm = first["cohorts"]["val_confirm"]
    historical = first["cohorts"]["historical_exposed"]
    assert not (set(gate["conversation_ids"]) & set(confirm["conversation_ids"]))
    assert not (set(gate["conversation_ids"]) & set(historical["conversation_ids"]))
    assert not (set(confirm["conversation_ids"]) & set(historical["conversation_ids"]))


def _metrics(micro_f1: float, precision: float, tp: int, fp: int, f1: float) -> dict:
    dims = {name: {"local_micro_f1": f1} for name in ("personality_traits", "interests", "style")}
    return {
        "profile_33_label": {
            "overall_33_label_micro_f1": micro_f1,
            "overall_33_label_micro_precision": precision,
            "overall_33_label_tp": tp,
            "overall_33_label_fp": fp,
            "per_dimension": dims,
        },
        "strict_parse_rate": 1.0,
        "eos_completion_rate": 1.0,
    }


def test_quality_gate_rejects_fp_spike_and_accepts_bounded_gain():
    parent = _metrics(0.40, 0.50, 100, 100, 0.40)
    good = _metrics(0.405, 0.50, 110, 105, 0.405)
    assert gate_reasons(parent, good)["passed"] is True
    bad = _metrics(0.41, 0.48, 111, 130, 0.38)
    result = gate_reasons(parent, bad)
    assert result["passed"] is False
    assert "new_false_positive_ratio_exceeds_1_5" in result["reasons"]
    assert "personality_traits_f1_drop_exceeds_0_02" in result["reasons"]
