from __future__ import annotations

import pytest

from src.training.experiment_spec import validate_experiment_spec
from src.training.sampling import build_sampling_plan, validate_bce_targets_and_mask


DIMS = ("personality_traits", "interests", "style")


def _train_rows():
    rows = []
    for index in range(8):
        profile = {
            "personality_traits": ["kind"] if index % 2 else [],
            "interests": ["music"] if index % 3 == 0 else [],
            "style": [],
        }
        rows.append({
            "sample_id": f"train:s{index}",
            "conversation_id": f"conversation:{index // 2}",
            "source": {"split": "train"},
            "target": {"user_profile": profile},
        })
    return rows


def _spec():
    return {
        "version": "E03-experiment-spec-v1",
        "experiment_id": "E03-RANK-r16-v1",
        "execution": {"enabled": False},
        "authorization": {"status": "requires_separate_approval"},
        "objective": "sft",
        "sampling": {
            "strategy": "uniform_without_replacement",
            "source_split": "train",
            "source_train_sha256": "a" * 64,
            "seed": 42,
            "draw_count": 8,
            "replay_fraction": 0.0,
        },
        "training": {
            "max_optimizer_steps": 4, "max_training_seconds": 600,
            "per_device_micro_batch_size": 1, "gradient_accumulation_steps": 2,
        },
        "lora": {"r": 16, "alpha": 32, "target_modules": ["q_proj", "v_proj"]},
        "resource_gate": {"max_gpu_memory_bytes": 8_000_000_000},
        "initialization": {
            "source": "base",
            "adapter_path": None,
            "resume_from_checkpoint": None,
            "base_weight_sha256": "a" * 64,
        },
    }


def test_sampling_schedule_is_deterministic_train_only_and_records_replay_exposure():
    first, summary = build_sampling_plan(
        _train_rows(),
        draw_count=8,
        seed=7,
        source_train_sha256="a" * 64,
        replay_sample_ids=["train:s0"],
        replay_fraction=0.25,
    )
    second, repeated = build_sampling_plan(
        _train_rows(),
        draw_count=8,
        seed=7,
        source_train_sha256="a" * 64,
        replay_sample_ids=["train:s0"],
        replay_fraction=0.25,
    )
    assert first == second
    assert summary == repeated
    assert summary["source_split"] == "train"
    assert summary["replay_draw_count"] == 2
    assert summary["draw_count"] == 8
    assert summary["gold_profile_all_empty_draws"] + summary["gold_profile_nonempty_draws"] == 8
    assert sum(summary["conversation_exposures"].values()) == 8
    assert summary["sample_exposures"]["train:s0"] >= 2
    assert summary["repeated_sample_draw_count"] == 8 - summary["unique_sample_count"]
    assert summary["dev_rows_read"] == summary["val_rows_read"] == 0
    assert summary["optimizer_step_count"] == 8


def test_sampling_refuses_dev_val_and_replay_ids_outside_train():
    rows = _train_rows()
    rows[0] = {**rows[0], "source": {"split": "dev"}}
    with pytest.raises(ValueError, match="Dev/Val"):
        build_sampling_plan(rows, draw_count=4, seed=1, source_train_sha256="b" * 64)
    with pytest.raises(ValueError, match="subset"):
        build_sampling_plan(
            _train_rows(), draw_count=4, seed=1, source_train_sha256="b" * 64,
            replay_sample_ids=["dev:s1"], replay_fraction=0.25,
        )
    with pytest.raises(ValueError, match="align"):
        build_sampling_plan(
            _train_rows(), draw_count=7, seed=1, source_train_sha256="b" * 64,
            micro_batch_size=2, gradient_accumulation_steps=2,
        )


def test_bce_mask_keeps_unknown_targets_out_of_negative_class():
    targets = [None] * 33
    mask = [0] * 33
    targets[0], mask[0] = 1, 1
    normalized_targets, normalized_mask = validate_bce_targets_and_mask(targets, mask)
    assert normalized_targets[0] == 1.0
    assert normalized_mask[0] == 1.0
    assert normalized_targets[1:] == [0.0] * 32
    assert normalized_mask[1:] == [0.0] * 32
    with pytest.raises(ValueError, match="mask=0"):
        validate_bce_targets_and_mask([None] + [0] * 32, [1] + [1] * 32)
    with pytest.raises(ValueError, match="mask=1"):
        validate_bce_targets_and_mask([0] * 33, [0] * 33)


def test_experiment_spec_is_validation_only_and_rank_change_starts_from_base():
    observed_adapter = {"r": 8, "base_weight_sha256": "a" * 64}
    result = validate_experiment_spec(_spec(), observed_adapter=observed_adapter)
    assert result["status"] == "validation_passed"
    assert result["lora_r"] == 16
    assert result["initialization_source"] == "base"
    assert result["training_enabled"] is False
    assert result["authorization_status"] == "requires_separate_approval"
    unsafe = _spec()
    unsafe["initialization"]["adapter_path"] = "outputs/training/old/adapter"
    with pytest.raises(ValueError, match="existing Adapter"):
        validate_experiment_spec(unsafe, observed_adapter=observed_adapter)


def test_spec_cannot_enable_training_or_use_old_a1_identity():
    unsafe = _spec()
    unsafe["execution"]["enabled"] = True
    with pytest.raises(ValueError, match="cannot enable"):
        validate_experiment_spec(unsafe)
    old_identity = _spec()
    old_identity["experiment_id"] = "E03-A1"
    with pytest.raises(ValueError, match="new versioned"):
        validate_experiment_spec(old_identity)
