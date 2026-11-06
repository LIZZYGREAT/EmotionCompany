from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.training.e03_a5_multitask import (
    FinalHiddenCapture,
    combine_sft_bce_losses,
    completion_only_sft_loss,
    forward_with_prompt_end_hidden,
    gather_prompt_end_hidden,
    masked_bce_with_logits,
)
from src.training.e03_a5_probe import (
    choose_global_threshold,
    extract_prompt_features,
    iter_prompt_features,
    score_external_dev_probe,
)


class _Backbone(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)


class _TinyCausalLM(nn.Module):
    """A deterministic causal toy: each position sees only the prefix sum."""

    def __init__(self, vocab_size: int = 32, hidden_size: int = 8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.model = _Backbone(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, use_cache=False):
        hidden = self.model.norm(self.embedding(input_ids).cumsum(dim=1))
        return SimpleNamespace(logits=self.lm_head(hidden))


def _captured_prompt_hidden(model, token_ids, prompt_length):
    ids = torch.tensor([token_ids], dtype=torch.long)
    attention = torch.ones_like(ids)
    lengths = torch.tensor([prompt_length], dtype=torch.long)
    with FinalHiddenCapture(model) as capture:
        _outputs, prompt_hidden = forward_with_prompt_end_hidden(
            model, capture, ids, attention, lengths
        )
    return prompt_hidden.detach()


def test_hidden_at_prompt_end_is_independent_of_following_gold_or_response():
    torch.manual_seed(7)
    model = _TinyCausalLM().eval()
    prompt_only = _captured_prompt_hidden(model, [1, 2], 2)
    gold_a = _captured_prompt_hidden(model, [1, 2, 3, 4, 5], 2)
    gold_b = _captured_prompt_hidden(model, [1, 2, 9, 8, 7], 2)
    response_changed = _captured_prompt_hidden(model, [1, 2, 3, 4, 11], 2)
    assert torch.allclose(prompt_only, gold_a, atol=1e-6, rtol=1e-6)
    assert torch.allclose(gold_a, gold_b, atol=1e-6, rtol=1e-6)
    assert torch.allclose(gold_a, response_changed, atol=1e-6, rtol=1e-6)


def test_wrong_last_token_feature_fails_the_gold_leakage_control():
    torch.manual_seed(13)
    model = _TinyCausalLM().eval()
    ids_a = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    ids_b = torch.tensor([[1, 2, 8, 9]], dtype=torch.long)
    last_a = model.model.norm(model.embedding(ids_a).cumsum(dim=1))[:, -1, :]
    last_b = model.model.norm(model.embedding(ids_b).cumsum(dim=1))[:, -1, :]
    prompt_a = _captured_prompt_hidden(model, [1, 2, 3, 4], 2)
    prompt_b = _captured_prompt_hidden(model, [1, 2, 8, 9], 2)
    assert not torch.allclose(last_a, last_b, atol=1e-6, rtol=1e-6)
    assert torch.allclose(prompt_a, prompt_b, atol=1e-6, rtol=1e-6)


def test_right_padding_gathers_each_rows_prompt_end_position():
    hidden = torch.arange(2 * 6 * 3, dtype=torch.float32).reshape(2, 6, 3)
    attention = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0]])
    prompt_lengths = torch.tensor([2, 4])
    gathered = gather_prompt_end_hidden(hidden, prompt_lengths, attention)
    assert torch.equal(gathered[0], hidden[0, 1])
    assert torch.equal(gathered[1], hidden[1, 3])
    with pytest.raises(ValueError, match="padding"):
        gather_prompt_end_hidden(hidden, torch.tensor([4, 4]), attention)


def test_feature_extraction_uses_prompt_prefix_only_and_records_dtype():
    torch.manual_seed(19)
    model = _TinyCausalLM().eval()
    rows = [
        {"sample_id": "a", "input_ids": [1, 2, 3, 4], "prompt_length": 2},
        {"sample_id": "b", "input_ids": [1, 2, 9, 8], "prompt_length": 2},
    ]
    ids, features, dtype = extract_prompt_features(model, rows, pad_token_id=0, batch_size=2)
    assert ids == ["a", "b"]
    assert dtype == "float32"
    assert torch.allclose(features[0], features[1], atol=1e-6, rtol=1e-6)


def test_target_free_dev_prompt_rows_can_be_extracted_without_completion_suffix():
    model = _TinyCausalLM().eval()
    rows = [{"sample_id": "dev-a", "input_ids": [1, 2, 3], "prompt_length": 3}]
    with pytest.raises(ValueError, match="source shape"):
        list(iter_prompt_features(model, rows, pad_token_id=0, allow_prompt_only=False))
    ids, features, dtype = extract_prompt_features(
        model, rows, pad_token_id=0, batch_size=1, allow_prompt_only=True,
    )
    assert ids == ["dev-a"]
    assert features.shape == (1, 8)
    assert dtype == "float32"


def test_external_probe_score_applies_train_calibration_threshold_without_dev_search():
    enums = {
        "personality_traits": [f"trait_{index}" for index in range(10)],
        "interests": [f"interest_{index}" for index in range(12)],
        "style": [f"style_{index}" for index in range(11)],
    }
    targets = torch.zeros((2, 33), dtype=torch.float32)
    targets[:, 0] = 1
    masks = torch.ones_like(targets)
    supervision = {
        sample_id: {"y": targets[index].tolist(), "mask": masks[index].tolist()}
        for index, sample_id in enumerate(("dev-a", "dev-b"))
    }
    head = nn.Linear(4, 33, dtype=torch.float32)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.fill_(-5)
        head.bias[0] = 5
    features = {"dev-a": torch.zeros(4), "dev-b": torch.ones(4)}
    thresholds = {
        "linear_head": {"threshold": 0.7},
        "frequency_prior": {"threshold": 0.6},
    }
    report = score_external_dev_probe(
        features,
        supervision,
        head,
        torch.tensor([0.8] + [0.1] * 32),
        thresholds,
        enums,
    )
    assert report["status"] == "dev_224_probe_scored"
    assert report["thresholds_selected_on_dev"] is False
    assert report["threshold_source"].startswith("frozen Train-Calibration")
    assert report["linear_head"]["train_calibration_threshold"]["threshold"] == 0.7
    assert report["linear_head"]["train_calibration_threshold"]["metrics"]["overall_33_label_micro_f1"] == 1.0
    assert report["frequency_prior"]["train_calibration_threshold"]["threshold"] == 0.6
    with pytest.raises(ValueError, match="ID sets"):
        score_external_dev_probe(
            {"dev-a": torch.zeros(4)}, supervision, head,
            torch.tensor([0.8] + [0.1] * 32), thresholds, enums,
        )


def test_final_hidden_hook_preserves_bce_gradient_path():
    torch.manual_seed(23)
    model = _TinyCausalLM().train()
    labels = torch.tensor([[0.0] * 33])
    mask = torch.ones_like(labels)
    ids = torch.tensor([[1, 2, 3, 4]])
    attention = torch.ones_like(ids)
    with FinalHiddenCapture(model) as capture:
        outputs, prompt_hidden = forward_with_prompt_end_hidden(
            model, capture, ids, attention, torch.tensor([2])
        )
        head = nn.Linear(prompt_hidden.shape[-1], 33)
        loss = masked_bce_with_logits(head(prompt_hidden), labels, mask)
        loss.backward()
    assert head.weight.grad is not None and torch.count_nonzero(head.weight.grad) > 0
    assert model.embedding.weight.grad is not None
    assert torch.count_nonzero(model.embedding.weight.grad) > 0
    assert outputs.logits.shape == (1, 4, 32)


def test_masked_bce_is_fp32_and_rejects_invalid_or_empty_masks():
    logits = torch.zeros((2, 33), dtype=torch.float16, requires_grad=True)
    targets = torch.zeros((2, 33), dtype=torch.float32)
    masks = torch.zeros((2, 33), dtype=torch.float32)
    masks[0, 0] = 1
    masks[1, :2] = 1
    targets[1, 1] = 1
    loss = masked_bce_with_logits(logits, targets, masks)
    assert loss.dtype == torch.float32
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    with pytest.raises(ValueError, match="at least one"):
        masked_bce_with_logits(logits.detach(), targets, torch.zeros_like(masks))
    bad_targets = targets.clone()
    bad_targets[0, 0] = 0.5
    with pytest.raises(ValueError, match="binary"):
        masked_bce_with_logits(logits.detach(), bad_targets, masks)


def test_lambda_zero_matches_standard_completion_only_ce_gradient():
    torch.manual_seed(29)
    logits_a = torch.randn(2, 5, 17, requires_grad=True)
    logits_b = logits_a.detach().clone().requires_grad_(True)
    labels = torch.tensor([
        [-100, -100, 4, 5, 6],
        [-100, 7, 8, 9, 10],
    ])
    sft = completion_only_sft_loss(logits_a, labels)
    zero_bce = logits_a.sum() * 0
    combined = combine_sft_bce_losses(sft, zero_bce, 0.0)
    combined.backward()
    independent = torch.nn.functional.cross_entropy(
        logits_b[:, :-1, :].contiguous().reshape(-1, 17),
        labels[:, 1:].contiguous().reshape(-1),
        ignore_index=-100,
    )
    independent.backward()
    assert torch.equal(sft, independent)
    assert torch.allclose(logits_a.grad, logits_b.grad, atol=0, rtol=0)


def test_global_threshold_uses_calibration_arrays_and_is_deterministic():
    probabilities = torch.full((4, 33), 0.01)
    targets = torch.zeros((4, 33))
    masks = torch.ones_like(targets)
    probabilities[:2, 0] = torch.tensor([0.8, 0.7])
    targets[:2, 0] = 1
    first = choose_global_threshold(probabilities, targets, masks)
    second = choose_global_threshold(probabilities, targets, masks)
    assert first == second
    assert first["threshold"] <= 0.7
    assert first["micro_f1"] == 1.0
