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
            "mode": "new_adapter_from_base",
            "source": "base",
            "adapter_path": None,
            "resume_from_checkpoint": None,
            "optimizer_state": "fresh",
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
    assert result["initialization_mode"] == "new_adapter_from_base"
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


def test_base_route_cannot_mix_observed_adapter_from_another_base():
    spec = _spec()
    with pytest.raises(ValueError, match="Base weight identity"):
        validate_experiment_spec(
            spec,
            observed_adapter={"base_weight_sha256": "b" * 64, "r": 8},
        )


def _same_rank_continuation():
    spec = _spec()
    spec["lora"].update(r=8, alpha=16, target_modules=["q_proj", "v_proj"])
    spec["initialization"] = {
        "mode": "new_run_from_existing_adapter",
        "source": "existing_adapter",
        "adapter_path": "outputs/training/E01-B-FULL/adapter",
        "base_weight_sha256": "a" * 64,
        "adapter_weight_sha256": "b" * 64,
        "adapter_config_sha256": "c" * 64,
        "optimizer_state": "fresh",
        "resume_from_checkpoint": None,
        "quantization": {
            "method": "bitsandbytes_4bit",
            "load_in_4bit": True,
            "quant_type": "nf4",
            "compute_dtype": "bfloat16",
            "use_double_quant": True,
        },
        "parameter_dtype": "bfloat16",
    }
    observed = {
        "base_weight_sha256": "a" * 64,
        "weight_sha256": "b" * 64,
        "config_sha256": "c" * 64,
        "r": 8,
        "alpha": 16,
        "target_modules": ["q_proj", "v_proj"],
        "quantization": {
            "method": "bitsandbytes_4bit",
            "load_in_4bit": True,
            "quant_type": "nf4",
            "compute_dtype": "bfloat16",
            "use_double_quant": True,
        },
        "parameter_dtype": "bfloat16",
    }
    return spec, observed


def test_same_rank_continuation_requires_matching_adapter_and_fresh_optimizer():
    spec, observed = _same_rank_continuation()
    result = validate_experiment_spec(spec, observed_adapter=observed)
    assert result["status"] == "validation_passed"
    assert result["initialization_mode"] == "new_run_from_existing_adapter"
    assert result["initialization_source"] == "existing_adapter"
    assert result["source_adapter_weight_sha256"] == "b" * 64
    assert result["optimizer_state"] == "fresh"
    assert result["resume_from_checkpoint"] is None
    assert result["training_enabled"] is False


@pytest.mark.parametrize(
    ("observed_field", "observed_value", "message"),
    [
        ("base_weight_sha256", "d" * 64, "Base weight identity"),
        ("weight_sha256", "d" * 64, "Adapter weight SHA"),
        ("config_sha256", "d" * 64, "Adapter config SHA"),
        ("r", 16, "rank and alpha"),
        ("alpha", 32, "rank and alpha"),
        ("target_modules", ["q_proj"], "target modules"),
        ("quantization", {
            "method": "bitsandbytes_4bit", "load_in_4bit": True,
            "quant_type": "nf4", "compute_dtype": "float16", "use_double_quant": True,
        }, "quantization settings"),
        ("parameter_dtype", "float32", "parameter dtype"),
    ],
)
def test_same_rank_continuation_rejects_source_identity_or_runtime_mismatch(
    observed_field, observed_value, message,
):
    spec, observed = _same_rank_continuation()
    observed[observed_field] = observed_value
    with pytest.raises(ValueError, match=message):
        validate_experiment_spec(spec, observed_adapter=observed)


def test_same_rank_continuation_rejects_checkpoint_and_old_optimizer_state():
    spec, observed = _same_rank_continuation()
    spec["initialization"]["resume_from_checkpoint"] = "outputs/training/old/checkpoint-128"
    with pytest.raises(ValueError, match="cannot resume"):
        validate_experiment_spec(spec, observed_adapter=observed)
    spec, observed = _same_rank_continuation()
    spec["initialization"]["optimizer_state"] = "restored"
    with pytest.raises(ValueError, match="fresh optimizer"):
        validate_experiment_spec(spec, observed_adapter=observed)
