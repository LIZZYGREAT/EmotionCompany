from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from src.data.sft_collator import SFTDataCollator
from src.training.e03_a4_stability import (
    canonical_sha256,
    finite_nonzero_gradient_facts,
    materialize_a4_schedule,
    schedule_json_rows,
    validate_a4_config,
    validate_experiment_initialization_spec,
    validate_fresh_optimizer_state,
    validate_preencoded_row,
    validate_trainable_lora_names,
    verify_gpu_ticket,
)
from src.training.sampling import build_sampling_plan


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/training/e03_a4_lr_stability.yaml"
EOS = 151645


def _config():
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _records(count: int = 300):
    records = []
    sampling_rows = []
    for index in range(count):
        sample_id = f"train:sample:{index:04d}"
        conversation_id = f"train:conversation:{index // 3:04d}"
        ids = [11 + index % 7, 21, 31, 41, 51, 61, 71, EOS]
        labels = [-100] * 4 + ids[4:]
        records.append({
            "sample_id": sample_id,
            "conversation_id": conversation_id,
            "encoded": {
                "input_ids": ids,
                "labels": labels,
                "prompt_length": 4,
                "completion_length": 4,
            },
        })
        sampling_rows.append({
            "sample_id": sample_id,
            "conversation_id": conversation_id,
            "source_split": "train",
            "target": {"user_profile": {
                "personality_traits": ["kind"] if index % 2 else [],
                "interests": ["music"] if index % 3 == 0 else [],
                "style": [],
            }},
        })
    return records, sampling_rows


def test_a4_config_and_existing_adapter_spec_are_frozen():
    config = _config()
    result = validate_a4_config(config)
    assert result["optimizer_steps"] == 64
    assert result["draw_count"] == 256
    assert result["learning_rate"] == 2e-5
    initialization = config["experiment_spec"]["initialization"]
    observed = {
        "base_weight_sha256": initialization["base_weight_sha256"],
        "weight_sha256": initialization["adapter_weight_sha256"],
        "config_sha256": initialization["adapter_config_sha256"],
        "r": 8,
        "alpha": 16,
        "target_modules": list(config["lora"]["target_modules"]),
        "quantization": dict(initialization["quantization"]),
        "parameter_dtype": "bfloat16",
    }
    assert validate_experiment_initialization_spec(config, observed)["training_enabled"] is False

    changed = copy.deepcopy(config)
    changed["optimizer"]["learning_rate"] = 1e-4
    with pytest.raises(ValueError, match="exactly 2e-5"):
        validate_a4_config(changed)
    changed = copy.deepcopy(config)
    changed["training"]["max_optimizer_steps"] = 128
    with pytest.raises(ValueError, match="budget"):
        validate_a4_config(changed)
    changed = copy.deepcopy(config)
    changed["sampler"]["source_split"] = "dev"
    with pytest.raises(ValueError, match="Train"):
        validate_a4_config(changed)


def test_a4_sampling_is_deterministic_unique_train_only_and_step_aligned():
    records, sampling_rows = _records()
    draws, summary = build_sampling_plan(
        sampling_rows,
        draw_count=256,
        seed=42,
        source_train_sha256="a" * 64,
        micro_batch_size=1,
        gradient_accumulation_steps=4,
    )
    repeated, repeated_summary = build_sampling_plan(
        sampling_rows,
        draw_count=256,
        seed=42,
        source_train_sha256="a" * 64,
        micro_batch_size=1,
        gradient_accumulation_steps=4,
    )
    changed_seed, _ = build_sampling_plan(
        sampling_rows,
        draw_count=256,
        seed=43,
        source_train_sha256="a" * 64,
        micro_batch_size=1,
        gradient_accumulation_steps=4,
    )
    assert draws == repeated
    assert summary == repeated_summary
    assert summary["source_split"] == "train"
    assert summary["optimizer_step_count"] == 64
    assert summary["unique_sample_count"] == 256
    assert summary["repeated_sample_draw_count"] == 0
    assert summary["dev_rows_read"] == summary["val_rows_read"] == 0
    assert draws != changed_seed

    ordered, facts = materialize_a4_schedule(
        records, draws, eos_token_id=EOS, max_sequence_length=2048,
    )
    assert len(ordered) == 256
    assert len({row["sample_id"] for row in ordered}) == 256
    assert [row["optimizer_step"] for row in ordered] == [step for step in range(1, 65) for _ in range(4)]
    assert [row["microbatch_index"] for row in ordered[:8]] == [0, 1, 2, 3, 0, 1, 2, 3]
    assert facts["draw_count"] == facts["unique_sample_count"] == 256
    assert facts["repeated_sample_draw_count"] == 0
    assert facts["dev_rows_read"] == facts["val_rows_read"] == 0

    id_only = schedule_json_rows(ordered)
    assert len(id_only) == 256
    assert "input_ids" not in id_only[0] and "labels" not in id_only[0]


def test_a4_sampling_rejects_dev_and_encoded_identity_mismatches():
    records, sampling_rows = _records()
    sampling_rows[0] = {**sampling_rows[0], "source_split": "dev"}
    with pytest.raises(ValueError, match="Dev/Val"):
        build_sampling_plan(
            sampling_rows, draw_count=256, seed=42, source_train_sha256="a" * 64,
            micro_batch_size=1, gradient_accumulation_steps=4,
        )
    _, sampling_rows = _records()
    draws, _ = build_sampling_plan(
        sampling_rows, draw_count=256, seed=42, source_train_sha256="a" * 64,
        micro_batch_size=1, gradient_accumulation_steps=4,
    )
    records[0]["conversation_id"] = "mismatch"
    with pytest.raises(ValueError, match="does not match"):
        materialize_a4_schedule(records, draws, eos_token_id=EOS, max_sequence_length=2048)


def test_preencoded_completion_only_mask_and_eos_are_exact():
    row = {
        "sample_id": "train:one",
        "input_ids": [1, 2, 3, 4, EOS],
        "labels": [-100, -100, 3, 4, EOS],
        "prompt_length": 2,
        "completion_length": 3,
    }
    with pytest.raises(ValueError, match="Prompt Mask"):
        validate_preencoded_row({**row, "labels": [-100, 2, 3, 4, EOS]}, eos_token_id=EOS, max_sequence_length=8)
    with pytest.raises(ValueError, match="completion labels"):
        validate_preencoded_row({**row, "labels": [-100, -100, 3, 8, EOS]}, eos_token_id=EOS, max_sequence_length=8)
    with pytest.raises(ValueError, match="EOS"):
        validate_preencoded_row(
            {**row, "input_ids": [1, 2, 3, 4, 9], "labels": [-100, -100, 3, 4, 9]},
            eos_token_id=EOS,
            max_sequence_length=8,
        )
    with pytest.raises(ValueError, match="S=4"):
        validate_preencoded_row(row, eos_token_id=EOS, max_sequence_length=4)
    assert validate_preencoded_row(row, eos_token_id=EOS, max_sequence_length=8)["sequence_length"] == 5
    batch = SFTDataCollator(pad_token_id=0, max_sequence_length=8)([row])
    assert batch["input_ids"].tolist() == [[1, 2, 3, 4, EOS]]
    assert batch["labels"].tolist() == [[-100, -100, 3, 4, EOS]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1, 1]]


def test_gpu_preflight_ticket_is_bound_to_config_sources_schedule_and_code():
    binding = {
        "status": "passed",
        "run_id": "run",
        "config_sha256": "a" * 64,
        "cpu_preflight_sha256": "b" * 64,
        "schedule_sha256": "c" * 64,
        "encoded_rows_sha256": "d" * 64,
        "source_hashes": {"train": "e" * 64},
        "implementation_sha256": {"runner": "f" * 64},
        "gpu": {"device": 0},
        "dry_step": {"status": "passed", "optimizer_updates": 0},
    }
    ticket = {
        **binding,
        "status": "passed",
        "ticket_sha256": canonical_sha256(binding),
    }
    verify_gpu_ticket(
        ticket,
        run_id="run",
        config_sha256="a" * 64,
        cpu_preflight_sha256="b" * 64,
        schedule_sha256="c" * 64,
        encoded_rows_sha256="d" * 64,
        implementation_sha256={"runner": "f" * 64},
    )
    with pytest.raises(ValueError, match="implementation_sha256"):
        verify_gpu_ticket(
            ticket,
            run_id="run",
            config_sha256="a" * 64,
            cpu_preflight_sha256="b" * 64,
            schedule_sha256="c" * 64,
            encoded_rows_sha256="d" * 64,
            implementation_sha256={"runner": "0" * 64},
        )
    with pytest.raises(ValueError, match="digest"):
        verify_gpu_ticket(
            {**ticket, "ticket_sha256": "0" * 64},
            run_id="run",
            config_sha256="a" * 64,
            cpu_preflight_sha256="b" * 64,
            schedule_sha256="c" * 64,
            encoded_rows_sha256="d" * 64,
            implementation_sha256={"runner": "f" * 64},
        )
    with pytest.raises(ValueError, match="failed"):
        verify_gpu_ticket(
            {**ticket, "status": "failed"},
            run_id="run",
            config_sha256="a" * 64,
            cpu_preflight_sha256="b" * 64,
            schedule_sha256="c" * 64,
            encoded_rows_sha256="d" * 64,
            implementation_sha256={"runner": "f" * 64},
        )


def test_gradient_contract_requires_finite_nonzero_lora_gradients():
    assert finite_nonzero_gradient_facts([None, 0.5]) == {
        "gradient_tensor_count": 1, "finite": True, "nonzero": True,
    }
    with pytest.raises(ValueError, match="nonzero"):
        finite_nonzero_gradient_facts([0.0, None])
    with pytest.raises(ValueError, match="finite"):
        finite_nonzero_gradient_facts([float("nan")])


def test_optimizer_scope_and_initial_state_are_strict():
    names = [
        "model.layers.0.self_attn.q_proj.lora_A.default.weight",
        "model.layers.0.self_attn.q_proj.lora_B.default.weight",
        "model.layers.0.mlp.down_proj.lora_A.default.weight",
        "model.layers.0.mlp.down_proj.lora_B.default.weight",
    ]
    assert validate_trainable_lora_names(names, ["q_proj", "down_proj"]) == ["down_proj", "q_proj"]
    with pytest.raises(ValueError, match="only LoRA"):
        validate_trainable_lora_names(names + ["model.layers.0.norm.weight"], ["q_proj", "down_proj"])
    with pytest.raises(ValueError, match="target modules"):
        validate_trainable_lora_names(names, ["q_proj"])
    validate_fresh_optimizer_state({})
    with pytest.raises(ValueError, match="empty state"):
        validate_fresh_optimizer_state({"momentum": object()})
