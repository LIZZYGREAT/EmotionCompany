"""Synthetic CPU coverage for B selection, approval and real Trainer resume state."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from src.training import training_sft as training
from src.data.sft_collator import SFTDataCollator


def test_pilot_selection_covers_real_length_deciles_and_labels_without_dev():
    metadata, sources = [], []
    for index in range(2000):
        sample_id = f"train:c{index}:pp0"
        metadata.append({"sample_id": sample_id, "conversation_id": f"c{index}", "source_split": "train", "prompt_length": 50 + index % 1230, "completion_length": 20 + index % 501})
        sources.append({"sample_id": sample_id, "conversation_id": f"c{index}", "target": {"emotion_label": f"e{index % 16}", "user_profile": {"personality_traits": [] if index % 3 else ["tag"], "interests": [], "style": []}}})
    selected = training.deterministic_select_pilot_samples(metadata, sources, count=512, seed=42)
    assert selected == training.deterministic_select_pilot_samples(metadata, sources, count=512, seed=42)
    assert selected != training.deterministic_select_pilot_samples(metadata, sources, count=512, seed=43)
    assert len(selected) == len({row["sample_id"] for row in selected}) == 512
    assert {row["length_bin"] for row in selected} == set(range(10))
    assert len({row["emotion_label"] for row in selected}) == 16
    assert {row["profile_empty"] for row in selected} == {True, False}
    assert min(row["sequence_length"] for row in selected) == min(row["prompt_length"] + row["completion_length"] for row in metadata)
    assert max(row["sequence_length"] for row in selected) == max(row["prompt_length"] + row["completion_length"] for row in metadata)
    metadata[0]["source_split"] = "dev"
    with pytest.raises(ValueError, match="Train identity"):
        training.deterministic_select_pilot_samples(metadata, sources, count=512, seed=42)


def test_full_b_needs_matching_approval_and_separate_launch_flag(tmp_path):
    config, _ = training.load_config("configs/training/e01_b.yaml")
    with pytest.raises(ValueError, match="禁止启动全量训练"):
        training.validate_training_mode(config, "train", allow_full_training=True)
    approved = copy.deepcopy(config)
    approved["status"] = "approved_for_full_training"
    approved["approval"] = {"scope": "E01-B-FULL", "approved_by": "test operator", "approved_at_utc": "2026-10-28T00:00:00Z", "config_sha256": training.approved_config_sha256(approved)}
    with pytest.raises(ValueError, match="allow-full-training"):
        training.validate_training_mode(approved, "train")
    training.validate_training_mode(approved, "train", allow_full_training=True)
    approved["training"]["seed"] += 1
    with pytest.raises(ValueError, match="does not match"):
        training.validate_training_mode(approved, "train", allow_full_training=True)


def test_operator_approval_creates_new_independent_full_config_without_starting_training(tmp_path, monkeypatch):
    import yaml
    from scripts import e01_b_config
    draft, _ = training.load_config("configs/training/e01_b.yaml")
    config_dir = tmp_path / "configs/training"
    config_dir.mkdir(parents=True)
    (config_dir / "e01_b.yaml").write_text(yaml.safe_dump(draft))
    monkeypatch.setattr(e01_b_config, "ROOT", tmp_path)
    monkeypatch.setattr(e01_b_config, "review", lambda root: {"pilot_manifest_sha256": "a" * 64})
    output = tmp_path / "outputs/training/configs/approved.yaml"
    result = e01_b_config.approve(tmp_path / "outputs/training/pilot", output, approved_by="test operator", learning_rate=1e-4, epochs=1, save_steps=256, eval_steps=256, max_training_hours=12)
    approved, _ = training.load_config(output)
    assert approved["training"]["max_optimizer_steps"] == 4048
    assert approved["data"]["init_adapter"] is approved["data"]["resume_from_checkpoint"] is None
    assert result["training_started"] is False
    assert approved["approval"]["config_sha256"] == training.approved_config_sha256(approved)
    with pytest.raises(FileExistsError):
        e01_b_config.approve(tmp_path / "outputs/training/pilot", output, approved_by="test operator", learning_rate=1e-4, epochs=1, save_steps=256, eval_steps=256, max_training_hours=12)


def test_training_cli_cannot_write_run_artifacts_into_base_directory(tmp_path):
    import yaml
    config, _ = training.load_config("configs/training/e01_b_pilot.yaml")
    config["data"]["output_root"] = "participant/model"
    path = tmp_path / "wrong-output.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="output_root must remain under outputs/training"):
        training.load_config(path)


def _row(index: int, split: str = "train"):
    ids = [3, 4 + index % 8, 12 + index % 7, 1]
    return {"sample_id": f"train:c{index}:pp0", "conversation_id": f"c{index}", "source_split": split, "source": {}, "input_ids": ids, "labels": [-100, -100, ids[2], 1], "prompt_length": 2, "completion_length": 2}


def test_sft_dev_keeps_original_train_ids_but_cannot_enter_updates():
    row = _row(0, "dev")
    training.validate_train_row(row, expected_split="dev", expected_eos_token_id=1)
    with pytest.raises(ValueError, match="Train split"):
        training.validate_train_row(row)


def test_checkpoint_integrity_rejects_wrong_run_data_and_modified_optimizer(tmp_path):
    checkpoint = tmp_path / "checkpoint-2"
    checkpoint.mkdir()
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_config.json", "adapter_model.safetensors"):
        (checkpoint / name).write_bytes(b"synthetic checkpoint fixture")
    (checkpoint / "trainer_state.json").write_text('{"global_step":2}')
    facts = {"run_id": "pilot-one", "optimizer_step": 2, "train_sha256": "train-hash", "config_sha256": "config-hash"}
    training._write_checkpoint_manifest(checkpoint, facts)
    training._verify_checkpoint(checkpoint, expected={"train_sha256": "train-hash"}, run_id="pilot-one")
    with pytest.raises(ValueError, match="Run ID"):
        training._verify_checkpoint(checkpoint, expected={}, run_id="pilot-two")
    with pytest.raises(ValueError, match="train_sha256"):
        training._verify_checkpoint(checkpoint, expected={"train_sha256": "changed"}, run_id="pilot-one")
    (checkpoint / "optimizer.pt").write_bytes(b"changed optimizer state")
    with pytest.raises(ValueError, match="hash or size"):
        training._verify_checkpoint(checkpoint, expected={}, run_id="pilot-one")


def test_resume_accepts_only_explicitly_compatible_legacy_training_code_hash(tmp_path):
    checkpoint = tmp_path / "checkpoint-2"
    checkpoint.mkdir()
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_config.json", "adapter_model.safetensors"):
        (checkpoint / name).write_bytes(b"synthetic checkpoint fixture")
    (checkpoint / "trainer_state.json").write_text('{"global_step":2}')
    training._write_checkpoint_manifest(checkpoint, {
        "run_id": "pilot-one",
        "optimizer_step": 2,
        "training_code_sha256": "known-compatible-old-code",
    })
    training._verify_checkpoint(
        checkpoint,
        expected={"training_code_sha256": "current-code"},
        run_id="pilot-one",
        compatible_training_code_sha256=frozenset({"known-compatible-old-code"}),
    )
    with pytest.raises(ValueError, match="training_code_sha256"):
        training._verify_checkpoint(
            checkpoint,
            expected={"training_code_sha256": "current-code"},
            run_id="pilot-one",
        )


def test_resume_archives_uncheckpointed_log_tail_and_preserves_step_prefix(tmp_path):
    root = tmp_path / "pilot"
    root.mkdir()
    path = root / "train_log.jsonl"
    original = "".join(json.dumps({"optimizer_step": step, "loss": 1 / step}) + "\n" for step in range(1, 6)) + '{"incomplete"'
    path.write_text(original)
    training._prepare_resume_logs(root, 3)
    assert [row["optimizer_step"] for row in training._read_training_log(root)] == [1, 2, 3]
    assert next((root / "interrupted_attempts").glob("*/train_log.jsonl")).read_text() == original


def _tiny_trainer(root, *, stop_after=None, existing_logs=None):
    pytest.importorskip("trl")
    from peft import LoraConfig, get_peft_model
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    config, _ = training.load_config("configs/training/e01_b_pilot.yaml")
    config = copy.deepcopy(config)
    config["optimizer"].update(learning_rate=0.001, lr_scheduler="linear")
    config["training"].update(max_optimizer_steps=4, gradient_accumulation_steps=2, save_interval_steps=2, evaluation_interval_steps=2, max_training_seconds=180)
    config["evaluation"]["sft_dev_loss_rows"] = 2
    training._seed_everything(42, torch)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=32, n_embd=16, n_layer=1, n_head=2, n_positions=32, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, pad_token_id=0, eos_token_id=1))
    model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=["c_attn"], task_type="CAUSAL_LM", lora_dropout=0))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"<pad>": 0, "<eos>": 1, "<unk>": 2, **{f"t{i}": i for i in range(3, 32)}}, unk_token="<unk>")), pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
    root.mkdir(parents=True, exist_ok=True)
    counters = training._TrainingCounters(existing_logs)
    callback = training._optimizer_evidence_callback(model, training._trainable_parameters(model), config=config, samples=[], tokenizer=tokenizer, prompt="", schema=None, enums=None, run_root=root, snapshot_adapter=False, collator=counters, checkpoint_facts={"config_sha256": "cpu-fixture"}, stop_after_step=stop_after)
    trainer = training._build_sft_trainer(model, tokenizer, [_row(i) for i in range(8)], SFTDataCollator(0, max_sequence_length=32), config, max_steps=4, callback=callback, run_root=root, eval_rows=[_row(10, "dev"), _row(11, "dev")], training_counts=counters, _cpu_fixture=True)
    training._seed_everything(42, torch)
    return trainer, model, callback


def test_actual_cpu_trainer_resume_preserves_lora_lr_rng_order_and_labels(tmp_path):
    torch.set_num_threads(1)
    continuous, continuous_model, _ = _tiny_trainer(tmp_path / "continuous")
    continuous.train()
    paused_root = tmp_path / "paused"
    paused, _model, pause_callback = _tiny_trainer(paused_root, stop_after=2)
    paused.train()
    assert paused.state.global_step == 2
    assert pause_callback.stop_reason == "requested_checkpoint_pause"
    checkpoint = paused_root / "trainer/checkpoint-2"
    training._verify_checkpoint(checkpoint, expected={"config_sha256": "cpu-fixture"}, run_id=paused_root.name)
    resumed, resumed_model, _ = _tiny_trainer(paused_root, existing_logs=training._read_training_log(paused_root))
    resumed.train(resume_from_checkpoint=str(checkpoint))
    assert resumed.state.global_step == continuous.state.global_step == 4
    continuous_log = training._read_training_log(tmp_path / "continuous")
    resumed_log = training._read_training_log(paused_root)
    assert [row["optimizer_step"] for row in resumed_log] == [1, 2, 3, 4]
    assert [row["learning_rate"] for row in continuous_log] == [row["learning_rate"] for row in resumed_log]
    assert sum(row["samples_seen"] for row in resumed_log) == 8
    assert sum(row["completion_tokens"] for row in resumed_log) == 16
    assert sum(row["input_tokens_seen"] for row in resumed_log) == 32
    assert resumed.e01_data_path_facts["labels_equal"] is True
    assert resumed.e01_data_path_facts["verified_random_sample_ids"]
    expected = dict(continuous_model.named_parameters())
    for name, parameter in resumed_model.named_parameters():
        assert torch.equal(parameter, expected[name]), name


def test_peft_resume_preserves_bfloat16_adapter_parameter_objects(tmp_path):
    trainer, model, callback = _tiny_trainer(tmp_path / "dtype-resume")
    initial_trainable = training._trainable_parameters(model)
    for _name, parameter in initial_trainable:
        parameter.data = parameter.data.to(torch.bfloat16)
    initial_ids = {name: id(parameter) for name, parameter in initial_trainable}

    checkpoint = tmp_path / "bfloat16-adapter-checkpoint"
    model.save_pretrained(checkpoint, safe_serialization=True)
    trainer._load_from_checkpoint(str(checkpoint))

    restored = training._trainable_parameters(model)
    assert {name: id(parameter) for name, parameter in restored} == initial_ids
    assert {parameter.dtype for _name, parameter in restored} == {torch.bfloat16}
    callback.on_train_begin(None, None, None)
    assert {name: id(parameter) for name, parameter in callback.trainable} == initial_ids
