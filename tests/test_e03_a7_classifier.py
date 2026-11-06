import torch
from torch import nn

from src.training.e03_a7_classifier import (
    arm_loss,
    build_heads,
    forward_prompt_end,
    tensor_state_sha256,
)


def test_profile_head_is_identical_across_arms_and_p_has_no_emotion_head():
    p = build_heads(12, "P", profile_seed=42, emotion_seed=43)
    ep = build_heads(12, "EP", profile_seed=42, emotion_seed=43)
    assert not hasattr(p, "emotion_head")
    assert hasattr(ep, "emotion_head")
    assert tensor_state_sha256(p.profile_head.state_dict()) == tensor_state_sha256(ep.profile_head.state_dict())
    assert p.profile_head.weight.dtype == torch.float32
    assert ep.emotion_head.weight.dtype == torch.float32


def test_profile_bce_and_emotion_ce_are_separate_and_both_differentiable():
    heads = build_heads(8, "EP")
    hidden = torch.randn(3, 8, requires_grad=True)
    output = heads(hidden)
    targets = torch.zeros(3, 33)
    targets[0, 1] = 1
    masks = torch.ones_like(targets)
    emotions = torch.tensor([0, 4, 15])
    losses = arm_loss(output, targets, masks, emotions, "EP", lambda_emotion=0.3)
    reference = losses["profile_bce"] + 0.3 * losses["emotion_ce"]
    assert torch.allclose(losses["total"], reference)
    losses["total"].backward()
    assert hidden.grad is not None and torch.count_nonzero(hidden.grad) > 0
    assert torch.count_nonzero(heads.profile_head.weight.grad) > 0
    assert torch.count_nonzero(heads.emotion_head.weight.grad) > 0


def test_p_loss_rejects_emotion_logits_or_targets():
    heads = build_heads(8, "P")
    output = heads(torch.randn(1, 8))
    with torch.no_grad():
        try:
            arm_loss(output, torch.zeros(1, 33), torch.ones(1, 33), torch.tensor([0]), "P", lambda_emotion=0.3)
        except ValueError as exc:
            assert "must not" in str(exc)
        else:
            raise AssertionError("P arm accepted Emotion targets")


def test_prompt_end_backbone_path_keeps_lora_gradient_and_ignores_suffix():
    class TinyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(32, 6)
            self.lora_A = nn.Parameter(torch.randn(6, 6) * 0.1)
            self.lm_head = nn.Linear(6, 32)

        def forward(self, input_ids, attention_mask, **_kwargs):
            hidden = self.embedding(input_ids) + self.embedding(input_ids) @ self.lora_A
            return type("Output", (), {"last_hidden_state": hidden})()

    class Causal(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = TinyBackbone()

    class Peft(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = Causal()

        def get_base_model(self):
            return self.base

    model = Peft()
    heads = build_heads(6, "P")
    ids = torch.tensor([[1, 2, 3, 4]])
    mask = torch.ones_like(ids)
    hidden = forward_prompt_end(model, ids, mask, torch.tensor([2]))
    assert hidden.shape == (1, 6)
    logits = heads(hidden)["profile_logits"]
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))
    loss.backward()
    assert model.base.model.lora_A.grad is not None
    assert torch.count_nonzero(model.base.model.lora_A.grad) > 0
    assert model.base.model.lm_head.weight.grad is None
