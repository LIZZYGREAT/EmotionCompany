"""CPU-only regression coverage for E01 training isolation and data semantics."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.data.sft_collator import SFTDataCollator
from src.training import training_sft as training


def _selected_fixture_rows():
    emotions = ["anger", "anxiety", "care", "disgust", "fear", "gratitude", "helplessness", "joy"]
    sft = []
    normalized = []
    for index, emotion in enumerate(emotions):
        sample_id = f"train:conv_{index:03d}:pp0"
        conversation_id = f"conv_{index:03d}"
        prompt_len = 5 + index
        completion_len = 3
        sft.append({
            "sample_id": sample_id,
            "conversation_id": conversation_id,
            "source_split": "train",
            "prompt_length": prompt_len,
            "completion_length": completion_len,
        })
        profile_values = [] if index % 2 == 0 else ["one"]
        normalized.append({
            "sample_id": sample_id,
            "conversation_id": conversation_id,
            "target": {
                "emotion_label": emotion,
                "user_profile": {
                    "personality_traits": profile_values,
                    "interests": [],
                    "style": [],
                },
            },
        })
    return sft, normalized


def _row(sample_id="train:conv_000:pp0", *, source_split="train"):
    input_ids = [11, 12, 20, 21]
    return {
        "sample_id": sample_id,
        "conversation_id": "conv_000",
        "source_split": source_split,
        "source": {},
        "prompt_length": 2,
        "completion_length": 2,
        "input_ids": input_ids,
        "labels": [-100, -100, 20, 21],
    }


def test_e01_configs_lock_the_same_base_and_isolate_b():
    config_a, _ = training.load_config("configs/training/e01_a.yaml")
    config_b, _ = training.load_config("configs/training/e01_b.yaml")
    assert config_a["base_model"]["expected_weight_sha256"] == config_b["base_model"]["expected_weight_sha256"]
    assert config_a["data"]["init_adapter"] is config_a["data"]["resume_from_checkpoint"] is None
    assert config_b["data"]["init_adapter"] is config_b["data"]["resume_from_checkpoint"] is None
    assert config_b["training"]["max_optimizer_steps"] is None
    assert config_b["status"] == "draft_manual_gate"
    with pytest.raises(ValueError, match="禁止启动全量训练"):
        training.validate_training_mode(config_b, "train")


def test_e01_training_enables_nf4_runtime_inspection():
    config, _ = training.load_config("configs/training/e01_a.yaml")
    assert training._training_quantization_inspection_config(config) == {
        "quantization": {"enabled": True, "use_double_quant": True},
    }
    invalid = json.loads(json.dumps(config))
    invalid["quantization"]["quant_type"] = "fp4"
    with pytest.raises(ValueError, match="requires bitsandbytes NF4"):
        training._training_quantization_inspection_config(invalid)


def test_e01_selection_is_stable_covers_labels_and_uses_unique_train_conversations():
    sft, normalized = _selected_fixture_rows()
    selected = training.deterministic_select_samples(sft, normalized, count=8)
    assert selected == training.deterministic_select_samples(sft, normalized, count=8)
    assert len({row["conversation_id"] for row in selected}) == 8
    assert len({row["emotion_label"] for row in selected}) >= 4
    assert {row["profile_empty"] for row in selected} == {True, False}
    assert all(row["sample_id"].startswith("train:") for row in selected)


def test_e01_selection_rejects_missing_d03_source_and_duplicate_conversations():
    sft, normalized = _selected_fixture_rows()
    with pytest.raises(ValueError, match="not found in D0.3 Train"):
        training.deterministic_select_samples(sft, normalized[:-1], count=8)
    for row in sft[1:]:
        row["conversation_id"] = sft[0]["conversation_id"]
        for source in normalized:
            if source["sample_id"] == row["sample_id"]:
                source["conversation_id"] = sft[0]["conversation_id"]
    with pytest.raises(ValueError, match="unique conversations"):
        training.deterministic_select_samples(sft, normalized, count=8)


def test_e01_preencoded_labels_eos_and_train_split_are_checked():
    row = _row()
    training.validate_train_row(row, expected_eos_token_id=21)
    # The chat template can append supervised whitespace after its terminal EOS.
    training.validate_train_row({
        **row,
        "input_ids": [11, 12, 20, 21, 22],
        "labels": [-100, -100, 20, 21, 22],
        "completion_length": 3,
    }, expected_eos_token_id=21)
    collator = SFTDataCollator(pad_token_id=0, max_sequence_length=8)
    batch = collator([row, {**row, "sample_id": "train:conv_001:pp0", "conversation_id": "conv_001", "input_ids": [13, 20, 21], "labels": [-100, 20, 21], "prompt_length": 1, "completion_length": 2}])
    assert batch["labels"].tolist() == [[-100, -100, 20, 21], [-100, 20, 21, -100]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1], [1, 1, 1, 0]]
    with pytest.raises(ValueError, match="Train split"):
        training.validate_train_row(_row(source_split="dev"))
    with pytest.raises(ValueError, match="completion labels differ"):
        training.validate_train_row({**row, "labels": [-100, -100, 20, 99]})


class _ToyAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = nn.Linear(4, 4, bias=False)
        self.lora_B = nn.Linear(4, 8, bias=False)

    def forward(self, values):
        return self.lora_B(torch.tanh(self.lora_A(values)))


class _ToyCausalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 4)
        self.adapter = _ToyAdapter()
        self.head = nn.Linear(4, 8, bias=False)
        self.embedding.weight.requires_grad_(False)
        self.head.weight.requires_grad_(False)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, attention_mask, labels):
        hidden = self.embedding(input_ids)
        logits = self.head(hidden) + self.adapter(hidden)
        loss = F.cross_entropy(logits.reshape(-1, 8), labels.reshape(-1), ignore_index=-100)
        return SimpleNamespace(loss=loss)


def test_e01_cpu_synthetic_step_proves_only_lora_changes():
    torch.manual_seed(5)
    model = _ToyCausalModel()
    params = training._trainable_parameters(model)
    optimizer = torch.optim.AdamW([parameter for _name, parameter in params], lr=0.01)
    base_before = model.embedding.weight.detach().clone(), model.head.weight.detach().clone()
    row = {
        "input_ids": [1, 2, 3],
        "labels": [-100, 2, 3],
        "prompt_length": 1,
        "completion_length": 2,
    }
    evidence = training._step_update(
        model, torch, optimizer, [row], SFTDataCollator(pad_token_id=0), accumulation_steps=1,
    )
    assert evidence["finite_gradients"] is True
    assert evidence["nonzero_gradient_parameter_count"] > 0
    assert evidence["updated_parameter_count"] > 0
    assert torch.equal(base_before[0], model.embedding.weight)
    assert torch.equal(base_before[1], model.head.weight)


def test_e01_run_ids_are_immutable_and_cannot_escape_output_root(tmp_path):
    config, _ = training.load_config("configs/training/e01_a.yaml")
    config = json.loads(json.dumps(config))
    config["data"]["output_root"] = str(tmp_path / "training")
    root = training._run_root(config, "E01-A-cpu-test")
    assert root.is_dir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        training._run_root(config, "E01-A-cpu-test")
    with pytest.raises(ValueError, match="run-id"):
        training._run_root(config, "../outside")


def test_e01_train_requires_an_independent_passing_optimizer_proof(tmp_path):
    config, _ = training.load_config("configs/training/e01_a.yaml")
    run_root = tmp_path / "new-run"
    proof = tmp_path / "step_proof.json"
    proof.write_text(json.dumps({
        "status": "passed",
        "optimizer_steps": 1,
        "lora_parameter_changed": True,
        "base_weight_sha256": config["base_model"]["expected_weight_sha256"],
    }), encoding="utf-8")
    assert training._verify_step_proof(proof, run_root, config["base_model"]["expected_weight_sha256"])["status"] == "passed"
    proof.write_text(json.dumps({"status": "blocked_gpu_unavailable"}), encoding="utf-8")
    with pytest.raises(ValueError, match="passing one-step"):
        training._verify_step_proof(proof, run_root, config["base_model"]["expected_weight_sha256"])
