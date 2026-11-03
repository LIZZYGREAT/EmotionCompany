"""CPU-only invariants for the isolated E03-A1 weighted causal loss."""

from __future__ import annotations

import json

import torch
import torch.nn.functional as F

from src.data.sft_collator import SFTDataCollator
from src.training.e03_a1_loss import (
    E03A1DataCollator,
    build_alignment_cache,
    build_loss_weights,
    independently_recomputed_weighted_causal_lm_loss,
    loss_breakdown_from_logits,
    weighted_causal_lm_loss,
)
from scripts.e03_a1_loss_contrast import verify_short_gate
from scripts.e03_a1_evaluate import _retention_gate


def test_weight_builder_only_weights_requested_active_completion_regions():
    labels = [-100, -100, 5, 6, 7, 8]
    primary = ["other", "other", "user_profile", "user_profile", "boundary_mixed", "emotion_label"]
    fine = [None, None, "profile_structure", "profile_label_value", "profile_boundary_mixed", None]

    assert build_loss_weights(labels, primary, fine, "token_mean") == [0, 0, 1, 1, 1, 1]
    assert build_loss_weights(labels, primary, fine, "profile_x2") == [0, 0, 2, 2, 1, 1]
    assert build_loss_weights(labels, primary, fine, "label_value_x2") == [0, 0, 1, 2, 1, 1]


def test_weighted_causal_loss_matches_independent_token_mean_formula():
    torch.manual_seed(7)
    logits = torch.randn(1, 5, 11, dtype=torch.float32, requires_grad=True)
    labels = torch.tensor([[-100, -100, 2, 4, 7]])
    weights = torch.tensor([[0.0, 0.0, 1.0, 2.0, 2.0]])

    observed = weighted_causal_lm_loss(logits, labels, weights, token_chunk_size=2)
    audited = independently_recomputed_weighted_causal_lm_loss(logits, labels, weights, token_chunk_size=2)
    token_losses = F.cross_entropy(logits[:, :-1].reshape(-1, 11), labels[:, 1:].reshape(-1), ignore_index=-100, reduction="none")
    shifted_weights = weights[:, 1:].reshape(-1)
    active = labels[:, 1:].reshape(-1).ne(-100)
    expected = (token_losses[active] * shifted_weights[active]).sum() / shifted_weights[active].sum()

    assert torch.allclose(observed, expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(observed.detach(), audited, atol=2e-5, rtol=2e-5)
    observed.backward()
    assert logits.grad is not None and bool(torch.isfinite(logits.grad).all())


def test_weighted_causal_loss_rejects_missing_or_misaligned_targets():
    logits = torch.zeros(1, 3, 4)
    labels = torch.tensor([[-100, -100, 1]])
    with torch.no_grad():
        try:
            weighted_causal_lm_loss(logits, labels, torch.zeros(1, 3))
        except ValueError as exc:
            assert "positive loss weight" in str(exc)
        else:
            raise AssertionError("active target with zero weight must be rejected")


def test_a1_collator_pads_loss_weights_and_region_fields_on_the_right():
    base = SFTDataCollator(pad_token_id=0)
    collator = E03A1DataCollator(base)
    features = [
        {
            "sample_id": "a",
            "input_ids": [1, 2, 3],
            "labels": [-100, 2, 3],
            "prompt_length": 1,
            "completion_length": 2,
            "loss_weights": [0.0, 1.0, 2.0],
            "primary_region_ids": [0, 1, 1],
            "label_value_mask": [0, 0, 1],
        },
        {
            "sample_id": "b",
            "input_ids": [4, 5],
            "labels": [-100, 5],
            "prompt_length": 1,
            "completion_length": 1,
            "loss_weights": [0.0, 1.0],
            "primary_region_ids": [0, 2],
            "label_value_mask": [0, 0],
        },
    ]

    batch = collator(features)
    assert batch["input_ids"].tolist() == [[1, 2, 3], [4, 5, 0]]
    assert batch["loss_weights"].tolist() == [[0.0, 1.0, 2.0], [0.0, 1.0, 0.0]]
    assert batch["primary_region_ids"].tolist() == [[0, 1, 1], [0, 2, 0]]
    assert batch["label_value_mask"].tolist() == [[0, 0, 1], [0, 0, 0]]


def test_loss_breakdown_reports_global_token_means_for_nested_label_values():
    torch.manual_seed(11)
    logits = torch.randn(1, 5, 9)
    labels = torch.tensor([[-100, -100, 1, 2, 3]])
    primary = torch.tensor([[0, 0, 1, 2, 3]])
    label_values = torch.tensor([[0, 0, 0, 1, 0]])

    result = loss_breakdown_from_logits(logits, labels, primary, label_values, token_chunk_size=2)
    token_losses = F.cross_entropy(logits[:, :-1].reshape(-1, 9), labels[:, 1:].reshape(-1), ignore_index=-100, reduction="none")
    active_losses = token_losses[labels[:, 1:].reshape(-1).ne(-100)]
    assert result["all_tokens"]["tokens"] == 3
    assert abs(float(result["all_tokens"]["mean_ce"]) - float(active_losses.mean())) < 1e-6
    assert result["user_profile"]["tokens"] == 1
    assert result["label_value"]["tokens"] == 1


def test_loss_breakdown_aggregates_regions_across_batches_with_missing_local_regions():
    from types import SimpleNamespace

    from src.training.e03_a1_loss import evaluate_loss_breakdown

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(16, 8)
            self.output = torch.nn.Linear(8, 16)
            self.grad_enabled_during_forward = None

        def get_input_embeddings(self):
            return self.embedding

        def forward(self, input_ids, attention_mask=None):
            self.grad_enabled_during_forward = torch.is_grad_enabled()
            return SimpleNamespace(logits=self.output(self.embedding(input_ids)))

    rows = [
        {
            "sample_id": "response-only",
            "input_ids": [1, 2, 3],
            "labels": [-100, 2, 3],
            "prompt_length": 1,
            "completion_length": 2,
            "loss_weights": [0.0, 1.0, 1.0],
            "primary_region_ids": [0, 3, 3],
            "label_value_mask": [0, 0, 0],
        },
        {
            "sample_id": "mixed-regions",
            "input_ids": [4, 5, 6, 7],
            "labels": [-100, 5, 6, 7],
            "prompt_length": 1,
            "completion_length": 3,
            "loss_weights": [0.0, 1.0, 1.0, 1.0],
            "primary_region_ids": [0, 1, 2, 3],
            "label_value_mask": [0, 1, 0, 0],
        },
    ]
    trainer = SimpleNamespace(
        model=TinyModel(),
        data_collator=E03A1DataCollator(SFTDataCollator(pad_token_id=0)),
    )

    result = evaluate_loss_breakdown(trainer, rows, batch_size=1)

    assert result["all_tokens"]["tokens"] == 5
    assert result["user_profile"]["tokens"] == 1
    assert result["label_value"]["tokens"] == 1
    assert result["emotion_label"]["tokens"] == 1
    assert result["response_text"]["tokens"] == 3
    assert trainer.model.grad_enabled_during_forward is False

    import pytest

    with pytest.raises(ValueError, match="full Dev subset has no supervised tokens.*label_value"):
        evaluate_loss_breakdown(trainer, rows[:1], batch_size=1)


def test_e03_a1_trainer_uses_weighted_loss_and_keeps_region_fields_off_model(tmp_path):
    import copy

    import pytest
    from peft import LoraConfig, get_peft_model
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast, TrainerCallback

    from src.training import training_sft

    pytest.importorskip("trl")
    config, _path = training_sft.load_config("configs/training/e03_a1_loss_contrast.yaml")
    config = copy.deepcopy(config)
    config["training_scope"] = "short_contrast"
    config["loss_contrast"]["variant"] = "profile_x2"
    config["optimizer"]["learning_rate"] = 0.001
    config["training"].update(
        max_optimizer_steps=2,
        gradient_accumulation_steps=2,
        save_interval_steps=2,
        evaluation_interval_steps=2,
        max_training_seconds=120,
    )
    config["evaluation"]["sft_dev_loss_rows"] = 2
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=32, n_embd=16, n_layer=1, n_head=2, n_positions=32,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, pad_token_id=0, eos_token_id=1,
    ))
    model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=["c_attn"], task_type="CAUSAL_LM", lora_dropout=0.0))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"<pad>": 0, "<eos>": 1, "<unk>": 2, **{f"t{i}": i for i in range(3, 32)}}, unk_token="<unk>")),
        pad_token="<pad>", eos_token="<eos>", unk_token="<unk>",
    )
    rows = []
    for index in range(8):
        first, second = index + 5, index + 15
        ids = [3, 4, first, second]
        rows.append({
            "sample_id": f"train:fixture:{index}",
            "input_ids": ids,
            "labels": [-100, -100, first, second],
            "prompt_length": 2,
            "completion_length": 2,
            "loss_weights": [0.0, 0.0, 2.0, 1.0],
            "primary_region_ids": [0, 0, 1, 2],
            "label_value_mask": [0, 0, 1, 0],
        })
    eval_rows = [dict(rows[0], sample_id="train:fixture:dev0"), dict(rows[1], sample_id="train:fixture:dev1")]
    collator = E03A1DataCollator(SFTDataCollator(pad_token_id=0, max_sequence_length=32))
    trainer = training_sft._build_sft_trainer(
        model, tokenizer, rows, collator, config, max_steps=2,
        callback=TrainerCallback(), run_root=tmp_path, eval_rows=eval_rows,
        _cpu_fixture=True,
    )
    batch = collator(rows[:1])
    loss, outputs = trainer.compute_loss(model, dict(batch), return_outputs=True)
    loss.backward()

    assert torch.isfinite(loss)
    assert outputs.logits.shape[:2] == batch["input_ids"].shape
    assert trainer.a1_loss_formula_audit_calls == 1
    assert "loss_weights" not in trainer.e01_data_path_facts["model_batch_fields"]
    assert trainer.e01_data_path_facts["model_batch_fields"] == ["input_ids", "attention_mask", "labels"]


def test_full_dev_retention_gate_applies_absolute_and_relative_thresholds():
    baseline = {
        "profile": {"overall_33_label_micro_f1": 0.40, "overall_33_label_macro_f1": 0.25, "overall_33_label_fp": 100},
        "emotion": {"accuracy": 0.50, "macro_f1": 0.40},
        "raw_diagnostics": {"raw_schema_valid_rate": 0.90, "raw_json_parse_rate": 0.95},
        "response": {"local_corpus_bleu4": {"score_percent": 10.0}, "local_rouge_l_macro_f1": {"value": 0.20}},
    }
    candidate = {
        "profile": {"overall_33_label_micro_f1": 0.411, "overall_33_label_macro_f1": 0.26, "overall_33_label_fp": 110},
        "emotion": {"accuracy": 0.495, "macro_f1": 0.40},
        "raw_diagnostics": {"raw_schema_valid_rate": 0.895, "raw_json_parse_rate": 0.95},
        "response": {"local_corpus_bleu4": {"score_percent": 9.6}, "local_rouge_l_macro_f1": {"value": 0.191}},
    }

    result = _retention_gate(baseline, candidate)

    assert result["status"] == "passed"
    assert all(result["checks"].values())
    candidate["profile"]["overall_33_label_fp"] = 111
    assert _retention_gate(baseline, candidate)["status"] == "not_passed"


def test_short_stage_gate_requires_label_gain_without_total_ce_regression(tmp_path):
    def write_run(name: str, variant: str, total_ce: float, label_ce: float) -> Path:
        root = tmp_path / name
        root.mkdir()
        (root / "manifest.json").write_text(json.dumps({
            "experiment_id": "E03-A1",
            "training_scope": "short_contrast",
            "status": "passed",
            "optimizer_steps": 128,
            "reload_check_status": "passed",
            "base_weights_unchanged": True,
            "loss_variant": variant,
        }))
        (root / "loss_breakdown.json").write_text(json.dumps({
            "metric_scope": "unweighted_teacher_forced_sft_dev",
            "dev_rows": 128,
            "sample_ids_sha256": "same-dev-ids",
            "regions": {
                "all_tokens": {"mean_ce": total_ce},
                "label_value": {"mean_ce": label_ce},
            },
        }))
        (root / "loss_mapping_manifest.json").write_text(json.dumps({"mapping_sha256": "same-mapping"}))
        return root

    baseline = write_run("token-mean", "token_mean", 1.0, 1.0)
    candidate = write_run("weighted", "profile_x2", 1.005, 0.97)
    assert verify_short_gate(candidate, baseline, candidate_variant="profile_x2")["status"] == "passed"

    regressed = write_run("weighted-regressed", "profile_x2", 1.011, 0.97)
    try:
        verify_short_gate(regressed, baseline, candidate_variant="profile_x2")
    except ValueError as exc:
        assert "did not pass" in str(exc)
    else:
        raise AssertionError("a total Dev CE increase over 1% must fail the short-stage gate")


def test_dev_alignment_reads_matching_source_from_normalized_d03_dev(tmp_path, monkeypatch):
    from src.diagnostics import e03_a0 as a0
    from src.diagnostics import e03_a0_1_profile as a01

    dev_sample_id = "train:fixture:dev"
    normalized_train = tmp_path / "normalized_train.jsonl"
    normalized_train.write_text('{"sample_id":"train:fixture:other"}\n', encoding="utf-8")
    normalized_dev = tmp_path / "normalized_dev.jsonl"
    normalized_dev.write_text(
        json.dumps({
            "sample_id": dev_sample_id,
            "conversation_id": "fixture-conversation",
            "fixture_split": "d03_dev",
        }) + "\n",
        encoding="utf-8",
    )
    sft_train = tmp_path / "sft_train.jsonl"
    sft_train.write_text("{}\n", encoding="utf-8")
    sft_dev = tmp_path / "sft_dev.jsonl"
    sft_dev.write_text("{}\n", encoding="utf-8")

    seen_sources = []

    def fake_build_sample_mapping(**kwargs):
        seen_sources.append((kwargs["source"], kwargs["expected_sft_split"]))
        return (
            "completion",
            "rendered",
            [(0, 1), (1, 2)],
            ["other", "user_profile"],
            [None, "profile_label_value"],
            {},
            {},
            {"completion_start": 0},
        )

    monkeypatch.setattr(a0, "_build_sample_mapping", fake_build_sample_mapping)
    monkeypatch.setattr(a01, "_profile_schema", lambda _schema: {})
    monkeypatch.setattr(
        a01,
        "assign_profile_fine_regions",
        lambda **_kwargs: {"token_regions": ["other", "profile_label_value"]},
    )

    mappings, facts = build_alignment_cache(
        config={
            "data": {"train_path": str(sft_train), "dev_path": str(sft_dev)},
            "protocol": {
                "p0_sha256": "p0",
                "tokenizer_signature_sha256": "tokenizer",
                "chat_template_sha256": "template",
                "max_prompt_tokens": 128,
            },
        },
        tokenizer=object(),
        prompt_text="",
        schema={"fixture": True},
        train_samples=[],
        dev_rows=[{"sample_id": dev_sample_id, "input_ids": [1, 2], "labels": [-100, 2]}],
        normalized_train_path=normalized_train,
        normalized_dev_path=normalized_dev,
        cache_root=tmp_path / "cache",
    )

    assert len(seen_sources) == 1
    source, expected_split = seen_sources[0]
    assert source["fixture_split"] == "d03_dev"
    assert expected_split == "dev"
    assert mappings[("dev", dev_sample_id)] == {
        "primary_region_ids": [0, 1],
        "label_value_mask": [0, 1],
    }
    assert facts["normalized_dev_sha256"]


def test_train_schema_path_is_verified_against_d043_and_d042_manifests(tmp_path):
    import pytest

    from src.training import training_sft

    schema_path = tmp_path / "train_public.schema.json"
    schema_path.write_text(json.dumps({"$defs": {"emotion": {"enum": ["neutral"]}}}), encoding="utf-8")
    schema_sha = training_sft.sha256_file(schema_path)
    sft_manifest = {
        "inputs": {
            "train_schema": {"path": str(schema_path), "sha256": schema_sha},
        },
    }
    d042_manifest = {
        "input_hashes_unchanged_after_run": {"train_schema": schema_sha},
    }

    assert training_sft._verified_train_schema_path(sft_manifest, d042_manifest) == schema_path

    d042_manifest["input_hashes_unchanged_after_run"]["train_schema"] = "0" * 64
    with pytest.raises(ValueError, match="D0.4.2/D0.4.3 Manifest identities"):
        training_sft._verified_train_schema_path(sft_manifest, d042_manifest)


def test_completed_step_checkpoint_can_be_finalized_without_an_extra_optimizer_step():
    import pytest

    from src.training import training_sft

    assert training_sft._checkpoint_reached_training_budget(128, 128)
    assert not training_sft._checkpoint_reached_training_budget(96, 128)
    with pytest.raises(ValueError, match="beyond the configured optimizer-step budget"):
        training_sft._checkpoint_reached_training_budget(129, 128)


def test_failed_run_with_missing_recorded_step_count_can_resume_from_checkpoint():
    from src.training import training_sft

    assert not training_sft._manifest_records_completed_budget({"status": "failed", "optimizer_steps": None}, 128)
    assert not training_sft._manifest_records_completed_budget({"status": "interrupted", "optimizer_steps": 96}, 128)
    assert training_sft._manifest_records_completed_budget({"status": "completed", "optimizer_steps": 128}, 128)


def test_e03_a1_resume_allows_only_the_known_post_eval_failure_code_hash():
    from src.training import training_sft

    failed_run_code_sha = "9408779d9aaa5faacd191a138e4acbe808427d53627de200a79648af4071dae0"
    assert failed_run_code_sha in training_sft._resume_compatible_training_code_hashes("E03-A1")
    assert failed_run_code_sha not in training_sft._resume_compatible_training_code_hashes("E01-B")
