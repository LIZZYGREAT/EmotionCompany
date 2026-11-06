from __future__ import annotations

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from src.training.e03_a5_multitask import (
    FinalHiddenCapture,
    assert_finite_nonzero_gradients,
    build_a5_train_schedule,
    combine_sft_bce_losses,
    completion_only_sft_loss,
    forward_with_prompt_end_hidden,
    masked_bce_with_logits,
)
from scripts.e03_a5_train import _head_state_sha256, _new_head


class _Backbone(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)


class _ToyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 8)
        self.model = _Backbone(8)
        self.adapter = nn.Linear(8, 8, bias=False)
        self.lm_head = nn.Linear(8, 19, bias=False)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, use_cache=False):
        hidden = self.embedding(input_ids).cumsum(dim=1)
        hidden = self.model.norm(hidden + self.adapter(hidden))
        return type("Output", (), {"logits": self.lm_head(hidden)})()


def test_fit_schedule_is_deterministic_unique_and_bounded_by_fit_ids():
    fit_ids = [f"fit-{index:04d}" for index in range(700)]
    first = build_a5_train_schedule(
        fit_ids, optimizer_steps=128, micro_batch_size=1,
        gradient_accumulation_steps=4, seed=42,
    )
    second = build_a5_train_schedule(
        list(reversed(fit_ids)), optimizer_steps=128, micro_batch_size=1,
        gradient_accumulation_steps=4, seed=42,
    )
    assert first == second
    assert first["draw_count"] == 512
    assert first["unique_draws"] is True
    assert len(first["sample_ids"]) == len(set(first["sample_ids"])) == 512
    assert set(first["sample_ids"]) <= set(fit_ids)
    assert first["source_split"] == "train_fit"
    with pytest.raises(ValueError, match="no-replacement"):
        build_a5_train_schedule(fit_ids[:10], optimizer_steps=3, gradient_accumulation_steps=4)
    with pytest.raises(ValueError, match="unique"):
        build_a5_train_schedule(["same", "same"], optimizer_steps=1, gradient_accumulation_steps=1)


def test_candidate_bce_gradient_reaches_lora_and_head_while_control_is_ce_only():
    torch.manual_seed(211)
    model = _ToyLM().train()
    labels = torch.tensor([[-100, -100, 3, 4, 5]])
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    attention = torch.ones_like(ids)
    targets = torch.zeros((1, 33), dtype=torch.float32)
    targets[0, 2] = 1
    mask = torch.ones_like(targets)
    head = nn.Linear(8, 33, dtype=torch.float32)

    with FinalHiddenCapture(model) as capture:
        output, prompt_hidden = forward_with_prompt_end_hidden(
            model, capture, ids, attention, torch.tensor([2])
        )
        sft = completion_only_sft_loss(output.logits, labels)
        bce = masked_bce_with_logits(head(prompt_hidden.float()), targets, mask)
        candidate = combine_sft_bce_losses(sft, bce, 0.1)
        assert torch.equal(sft, completion_only_sft_loss(output.logits, labels))
        candidate.backward()

    lora = assert_finite_nonzero_gradients([model.adapter.weight], description="toy LoRA")
    auxiliary = assert_finite_nonzero_gradients(list(head.parameters()), description="toy head")
    assert lora["parameters_with_nonzero_gradient"] == 1
    assert auxiliary["parameters_with_nonzero_gradient"] == 2

    model.zero_grad(set_to_none=True)
    head.zero_grad(set_to_none=True)
    with FinalHiddenCapture(model) as capture:
        output, _prompt_hidden = forward_with_prompt_end_hidden(
            model, capture, ids, attention, torch.tensor([2])
        )
        control_sft = completion_only_sft_loss(output.logits, labels)
        control = combine_sft_bce_losses(control_sft, bce, 0.0)
        assert control is control_sft
        control.backward()
    assert model.adapter.weight.grad is not None
    assert head.weight.grad is None
    assert head.bias.grad is None


def test_lambda_zero_matches_independent_microbatch_ce_accumulation():
    torch.manual_seed(257)
    vocab_size = 23
    logits_a = [
        torch.randn(1, 5, vocab_size, requires_grad=True),
        torch.randn(1, 7, vocab_size, requires_grad=True),
    ]
    logits_b = [value.detach().clone().requires_grad_(True) for value in logits_a]
    labels = [
        torch.tensor([[-100, -100, 3, 4, 5]]),
        torch.tensor([[-100, 6, 7, 8, 9, 10, 11]]),
    ]
    accumulation = len(labels)
    control_losses = [completion_only_sft_loss(value, target) for value, target in zip(logits_a, labels)]
    control = sum(loss / accumulation for loss in control_losses)
    combine_sft_bce_losses(control, torch.tensor(987.0), 0.0).backward()

    reference = sum(
        torch.nn.functional.cross_entropy(
            value[:, :-1, :].contiguous().reshape(-1, vocab_size),
            target[:, 1:].contiguous().reshape(-1),
            ignore_index=-100,
        ) / accumulation
        for value, target in zip(logits_b, labels)
    )
    reference.backward()
    assert torch.equal(control, reference)
    assert all(torch.equal(left.grad, right.grad) for left, right in zip(logits_a, logits_b))


def test_classifier_head_safetensors_roundtrip_preserves_logits(tmp_path):
    torch.manual_seed(307)
    head = nn.Linear(12, 33, dtype=torch.float32).eval()
    features = torch.randn(4, 12)
    expected = head(features).detach()
    path = tmp_path / "classifier_head.safetensors"
    save_file({key: value.detach().contiguous() for key, value in head.state_dict().items()}, str(path))
    reloaded = nn.Linear(12, 33, dtype=torch.float32).eval()
    reloaded.load_state_dict(load_file(str(path), device="cpu"))
    assert torch.equal(expected, reloaded(features))


def test_control_and_candidate_start_from_the_same_frozen_head_initialization():
    control = _new_head(16, seed=42, device=torch.device("cpu"), trainable=False)
    candidate = _new_head(16, seed=42, device=torch.device("cpu"), trainable=True)
    assert _head_state_sha256(control) == _head_state_sha256(candidate)
    assert all(torch.equal(control.state_dict()[name], candidate.state_dict()[name]) for name in control.state_dict())
    assert not any(parameter.requires_grad for parameter in control.parameters())
    assert all(parameter.requires_grad for parameter in candidate.parameters())
